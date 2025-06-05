#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-rknn.h"
#include "ggml-backend.h"
#include "ggml-impl.h" // i
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "fp16/Float16.h"
#include "model_related_config.h"

#include <string.h>
#include <cstring>

#include <thread>
#include <vector>

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <fstream>
#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <fcntl.h>
#include <sys/sysinfo.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <arm_neon.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define GGML_COMMON_DECL_C

#include "ggml-common.h"

using namespace rknpu2;
#define UNUSED(x) (void)(x)

#define GGML_RKNPU2_INPUT_SCALE 1.7f


#define DMA_HEAP_IOCTL_ALLOC	_IOWR(DMA_HEAP_IOC_MAGIC, 0x0,\
				      struct dma_heap_allocation_data)
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_BUF_SYNC_START  (0 << 2)
#define DMA_BUF_SYNC_END    (1 << 2)
#define DMA_BUF_SYNC_READ   (1 << 0)
#define DMA_BUF_SYNC_WRITE  (2 << 0)
#define DMA_BUF_SYNC_RW     (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_BASE        'b'
#define DMA_BUF_IOCTL_SYNC   _IOW(DMA_BUF_BASE, 0, uint64_t)
#define CMA_HEAP_SIZE       (1024 * 1024)

struct matrixShape{
    int64_t row;
    int64_t col;
};
struct matrixPair{
    matrixShape src0;
    matrixShape src1;
};

std::vector<matrixPair> support_matrices;

struct pad_data{
    void * data;
    bool is_padded=false;
};

enum matrix_t{
    FLOAT16,
    INT8,
    INT4
};

std::map<matrix_t, size_t> MATRIX_SIZE = {
    {FLOAT16, sizeof(float16)},
    {INT8, sizeof(int8_t)},
    {INT4, sizeof(int8_t)}
};

struct mat_info{
    int64_t row;
    int64_t col;
    int64_t pad_row;
    int64_t pad_col;
    matrix_t matrix_type;
    void ** ori_data;
    void ** pad_data;
    size_t ori_size;
    size_t pad_size;
    char * matrix_name;
    bool is_padded=false;
    bool is_A=false;

    mat_info(int64_t row, 
        int64_t col,
        int64_t pad_row, 
        int64_t pad_col, 
        matrix_t matrix_type, 
        void ** ori_data, 
        void ** pad_data, 
        size_t ori_size, 
        size_t pad_size, 
        char * matrix_name, 
        bool is_padded, 
        bool is_A): 
        row(row), 
        col(col), 
        pad_row(pad_row), 
        pad_col(pad_col), 
        matrix_type(matrix_type), 
        ori_data(ori_data), 
        pad_data(pad_data), 
        ori_size(ori_size), 
        pad_size(pad_size), 
        matrix_name(matrix_name), 
        is_padded(is_padded), 
        is_A(is_A)
        {}

    mat_info(int64_t row_, 
        int64_t col_,
        matrix_t matrix_type_, 
        void ** data_, 
        bool is_A_, 
        char* name_)
    : mat_info(
            row_,
            col_,
            row_ < 32 ? 32 : (((row_ - 1) / 32 + 1) * 32),
            col_ < 32 ? 32 : (((col_ - 1) / 32 + 1) * 32),
            matrix_type_,
            data_,
            NULL,
            row_ * col_ * MATRIX_SIZE[matrix_type_],
            (row_ < 32 ? 32 : (((row_ - 1)/ 32 + 1) * 32)) * (col_ < 32 ? 32 : (((col_ - 1)/ 32 + 1) * 32)) * MATRIX_SIZE[matrix_type_],
            name_,
            false,
            is_A_
        )
    {
        if(is_A_){
            this->pad_row = row_;
        }
    }

    mat_info(int64_t row_, int64_t col_, matrix_t matrix_type_, void ** data_, bool is_A_)
        : mat_info( row_, col_, matrix_type_, data_, is_A_, NULL) 
    {}
};


struct matmul_ctx{
    mat_info mat_A;
    mat_info mat_B;
    rknn_matmul_type type;
    int thread_idx;
    bool matrix_B00_need_set_io = false;
    int64_t ori_n;
    matmul_ctx(mat_info mat_A, mat_info mat_B, rknn_matmul_type type, int thread_idx, int64_t ori_n): mat_A(mat_A), mat_B(mat_B), type(type), thread_idx(thread_idx), ori_n(ori_n) {}
};

void check_pad_float(const int64_t row, const int64_t col, void *pad_A01)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%2.f ", ((float*)pad_A01)[i * col + j]);
        }
        printf("\n");
    }
}
bool read_shape_pairs_from_json(
    const std::string& filename,
    std::vector<matrixPair>& out_pairs) {

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open JSON file: " << filename << std::endl;
        return false;
    }

    json j;
    try {
        ifs >> j;
        // 拿到 pairs 数组
        const auto& arr = j.at("pairs");
        for (const auto& item : arr) {
            matrixPair sp;
            sp.src0.row = item.at("src0").at("row").get<int64_t>();
            sp.src0.col = item.at("src0").at("col").get<int64_t>();
            sp.src1.row = item.at("src1").at("row").get<int64_t>();
            sp.src1.col = item.at("src1").at("col").get<int64_t>();
            out_pairs.push_back(sp);
        }
    } catch (const std::exception &e) {
        std::cerr << "Get JSON failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}


struct ggml_rknpu2_matmul_kernel{
    void* workdata;
    size_t work_size = 0;
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    std::atomic<bool> is_using = false;
    int thread_idx=0;

    rknn_tensor_mem* A;
    rknn_tensor_mem* B;
    rknn_tensor_mem* C;

    void * A_data = NULL;
    void * B_data = NULL;
    size_t A_size = 0; //0 means not allocated
    size_t B_size = 0;
    bool B_is_copied = false;
};

static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}


static uint64_t rknpu2_allocated_bytes = 0;

struct matrix_ctx{
    int64_t row;
    int64_t col;
    void* data;
    char* name;
};

struct in_kernel_time{
    double memcpy_to_kernel_time;
    double find_kernel_time;
    double set_io_time;
    double run_time;
    double memcpy_to_result_time;
    double sum_result_time;
};

void matrix_B_to_perf_layout_single_thread(int total_blocks_outer, int total_blocks_j, int32_t subN, int32_t subK, int32_t K, float16 *__restrict__ dst_ptr, const float16 *start_point, int thread_idx, int total_threads);
void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n, int offset_col, int offset_row);
void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n);
void side_matrix_multiplication(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time);
void side_matrix_multiplication(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time, bool matrix_B00_need_set_io);
void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time);
void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time, bool matrix_B00_need_set_io);

void pad_side_matrix(const int64_t A_row_01, const int64_t A_col_01, const void *A_data, const int64_t A_row_00, int64_t k, void *&pad_A01, const int64_t A_col_00);

void pad_side_matrix(const int64_t A_row_01, const int64_t A_col_01, void *&pad_A01, const int64_t A_col_00, const void *A_data, int64_t k);

#define GGML_RKNPU2_MAX_MATMUL_KERNELS 32
static ggml_rknpu2_matmul_kernel matmul_kernels[GGML_RKNPU2_MAX_MATMUL_KERNELS];

static int matmul_kernels_count = 0;

const char* rknpu2_matmul_type_to_string(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return "FLOAT16_MM_FLOAT16_TO_FLOAT32";
        case RKNN_INT8_MM_INT8_TO_INT32:
            return "INT8_MM_INT8_TO_INT32";
        case RKNN_INT4_MM_INT4_TO_INT16:
            return "INT4_MM_INT4_TO_INT16";
        default:
            GGML_ASSERT(0);
    }
}

struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, int thread_idx, const void * A_data, void * B_data, size_t A_size, size_t B_size) {
  for (int i = 0; i < matmul_kernels_count; i++) {
    ggml_rknpu2_matmul_kernel *kernel = &matmul_kernels[i];
    if (
            kernel->info.M == m &&
            kernel->info.K == k &&
            kernel->info.N == n &&
            kernel->is_using == false &&
            // kernel->thread_idx == thread_idx &&
            kernel->info.type == type 
            // kernel->A_data == A_data &&
            // kernel->B_data == B_data &&
            // kernel->A_size == A_size &&
            // kernel->B_size == B_size
        ){
            // printf("find a kernel at %d\n", i);
            return kernel;
        }
  }
  return NULL;
}

// MARK: Kernel find

static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx){
    return ggml_rknpu2_matmul_kernel_find(A.row, A.col, B.col, type, thread_idx, NULL, NULL, 0, 0);
}

static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(matmul_ctx ctx){
    return ggml_rknpu2_matmul_kernel_find(ctx.mat_A.row, ctx.mat_A.col, ctx.mat_B.col, ctx.type, ctx.thread_idx, ctx.mat_A.ori_data, ctx.mat_B.ori_data, ctx.mat_A.ori_size, ctx.mat_B.ori_size);
}

ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(const void* A_data, void* B_data, size_t A_size, size_t B_size, int m, int k, int n, rknn_matmul_type type, int core_number, int &initialized, bool is_init = false){
    ggml_rknpu2_matmul_kernel* kernel = NULL;
    if(!is_init){
        kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type, core_number, A_data, B_data, A_size, B_size);
    }

    if(kernel != NULL){
        // printf("find an existed kernel!\n");
        // initialized = 1;
        return kernel;
    }
    else{
        // printf("Creating Kernel inside the function\n");
        // printf("create kernel id: %d\n", matmul_kernels_count);
        // printf("parameters: %d, %d, %d, %d\n", m, k, n, type);
        // GGML_ASSERT(matmul_kernels_count < GGML_RKNPU2_MAX_MATMUL_KERNELS);

        kernel = &matmul_kernels[matmul_kernels_count++];
        if(matmul_kernels_count % GGML_RKNPU2_MAX_MATMUL_KERNELS == 0)
            matmul_kernels_count = 0;
        memset(kernel, 0, sizeof(ggml_rknpu2_matmul_kernel));

        kernel->thread_idx = core_number;
        kernel->info.M = m;
        kernel->info.K = k;
        kernel->info.N = n;
        kernel->info.type = type;
        kernel->info.B_layout = 1; // B use native layout (weight)
        kernel->info.AC_layout = 1; // A and C use original layout (intermediate)

        // printf("Creating RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d) %s\n", m, k, k, n, m, n, rknpu2_matmul_type_to_string(type));
        // printf("kernel->ctx: %p\n", &(kernel->ctx));
        // printf("kernel->info: %p\n", &(kernel->info));
        // printf("kernel->io_attr: %p\n", &(kernel->io_attr));

        int ret = rknn_matmul_create(&(kernel->ctx), &(kernel->info), &(kernel->io_attr));
        GGML_ASSERT(ret == 0);

        if(core_number == 0)
            rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_0);
        else if(core_number == 1)
            rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_1);
        else if(core_number == 2)
            rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_2);

        {
            auto kernel_mem_create_time = std::chrono::high_resolution_clock::now();
            kernel->A = rknn_create_mem(kernel->ctx, kernel->io_attr.A.size);
            kernel->B = rknn_create_mem(kernel->ctx, kernel->io_attr.B.size);
            kernel->C = rknn_create_mem(kernel->ctx, kernel->io_attr.C.size);
            auto kernel_mem_create_time_end = std::chrono::high_resolution_clock::now();   
            auto kernel_mem_create_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_mem_create_time_end - kernel_mem_create_time).count();
            printf("kernel_mem_create_duration: %ld us\n", kernel_mem_create_duration);

        }
    }

    // printf("created\n");
    // printf("Created RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d) %s\n", m, k, k, n, m, n, rknpu2_matmul_type_to_string(type));

    {
        // kernel->A_data = (void*)A_data;
        // kernel->B_data = (void*)B_data;
        kernel->A_size = A_size;
        kernel->B_size = B_size;
    }

    return kernel;
}

static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, rknpu2::float16 * A_data, rknpu2::float16 * B_data, size_t A_size, size_t B_size, int &initialized) {
    return ggml_rknpu2_matmul_kernel_create(A_data, B_data, A_size, B_size, m,k,n,type,1, initialized);
}

void transposed_matrix_to_perf_layout(const void *src, void *dst, int32_t K, int32_t N, int32_t subK, int32_t subN)
{
    // subN = 16 subK = 32 on 3588
    const void * start_point = src;
    int dst_offset = 0;
    for(int outer = 0; outer < N / subN; outer++){
        for(int j = 0 ; j < K / subK; j++){
            int start_offset = outer * subN * K + j * subK;
            for(int i = 0 ; i < subN; i++){
                memcpy((float16 *)dst + dst_offset, (float16 *)start_point + start_offset, subK * sizeof(float16));
                dst_offset += subK;
                start_offset += K;
            }
        }
    }
}


static void process_range(
    const float16* src, float16* dst, int K,
    int start_outer, int end_outer,
    int subK, int subN, int total_j)
{
    const int block_size = subN * total_j * subK; // 每个outer块的大小
    
    for (int outer = start_outer; outer < end_outer; ++outer) {
        // 计算当前outer块的目标起始位置
        float16* block_dst = dst + outer * block_size;
        
        // 源矩阵的起始行
        const float16* outer_src = src + outer * subN * K;
        
        // 遍历j维度（K方向分块）
        for (int j = 0; j < total_j; ++j) {
            // 当前j块的起始位置
            const float16* j_src = outer_src + j * subK;
            
            // 遍历subN行
            for (int i = 0; i < subN; ++i) {
                // 目标位置：block + j块偏移 + 行内偏移
                float16* dst_pos = block_dst + (j * subN + i) * subK;
                
                // 源位置：当前outer块的i行，j列
                const float16* src_pos = j_src + i * K;
                
                // 32个float16=64字节，正好一个cacheline
                // static_assert(sizeof(float16)*32 == 64, "Cache line size mismatch");
                memcpy(dst_pos, src_pos, subK * sizeof(float16));
            }
        }
    }
}

void transposed_matrix_to_perf_layout_multi_threads(
    const void* src, void* dst, 
    int32_t K, int32_t N,
    int32_t subK, int32_t subN) 
{
    const float16* src_ptr = static_cast<const float16*>(src);
    float16* dst_ptr = static_cast<float16*>(dst);

    const int total_outer = N / subN;  // 外层循环次数
    const int total_j = K / subK;      // j方向分块数
    
    // 根据物理核心数设置线程数（建议4-6）
    const int n_threads = 1; 
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    // 计算每个线程处理的outer范围
    const int min_blocks_per_thread = total_outer / n_threads;
    const int remainder = total_outer % n_threads;

    for (int t = 0; t < n_threads; ++t) {
        // 带余数的均衡分配
        const int start_outer = t * min_blocks_per_thread + std::min(t, remainder);
        const int end_outer = start_outer + min_blocks_per_thread + (t < remainder ? 1 : 0);

        threads.emplace_back([=]() {
            process_range(
                src_ptr, dst_ptr, K,
                start_outer, end_outer,
                subK, subN, total_j
            );
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

void matrix_B_to_perf_layout_single_thread(
    int total_outer, int total_j, int subN, int subK, int K,
    float16* dst, const float16* src,
    int thread_id, int num_threads)
{
    const int blocks_per_thread = (total_outer + num_threads - 1) / num_threads;
    const int start_outer = thread_id * blocks_per_thread;
    const int end_outer = std::min((thread_id + 1) * blocks_per_thread, total_outer);
    
    int dst_offset = start_outer * subN * (K / subK) * subK;
    
    for (int outer = start_outer; outer < end_outer; ++outer) {
        for (int j = 0; j < total_j; ++j) {
            const int src_offset = outer * subN * K + j * subK;
            for (int i = 0; i < subN; ++i) {
                const int src_pos = src_offset + i * K;
                const int dst_pos = dst_offset + (j * subN + i) * subK;
                memcpy(&dst[dst_pos], &src[src_pos], subK * sizeof(float16));
            }
        }
        dst_offset += subN * total_j * subK; // 移动到下一个连续区域
    }
}
// void matrix_B_to_perf_layout_single_thread(int total_blocks_outer, int total_blocks_j, int32_t subN, int32_t subK, int32_t K, float16 *__restrict__ dst_ptr, const float16 *start_point, int thread_idx, const int total_threads)
// {
//     int outer_begin = thread_idx * (total_blocks_outer / total_threads);
//     int outer_end = (thread_idx + 1) * (total_blocks_outer / total_threads);
//     // for (int outer = 0; outer < total_blocks_outer; outer++)
//     for (int outer = outer_begin; outer < outer_end; outer++)
//     {
//         for (int j = 0; j < total_blocks_j; j++)
//         {
//             int local_offset = (outer * total_blocks_j + j) * subN * subK;
//             int start_offset = outer * subN * K + j * subK;
//             for (int i = 0; i < subN; i++)
//             {
//                 // printf("Thread %d dst_ptr = %p\n", thread_idx, dst_ptr + local_offset);
//                 memcpy(dst_ptr + local_offset + i * subK,
//                        start_point + start_offset,
//                        subK * sizeof(float16));
//                 start_offset += K;
//             }
//         }
//     }
// }
void perf_matrixC_to_norm_layout(void *src, void *&dst, int32_t M, int32_t N){
    if(M == 1){
        dst = src;
        return;
    }

    const int mem_unit = 4;
    int dst_offset = 0;
    for(int i = 0; i < M; i++){
        for(int outer = 0; outer < N / mem_unit ; outer++){
            memcpy((float*)dst + dst_offset, (float*)src + outer * mem_unit * M + i * mem_unit, mem_unit * sizeof(float));
            dst_offset += mem_unit;
        }
    }
}
void matrixA_to_perf_layout(const void* src, void *&dst, int32_t M, int32_t K){
    dst = malloc(M * K * sizeof(float16));
    const int mem_unit = 8;
    int dst_offset = 0;
    for(int outer = 0; outer < K / mem_unit ; outer++){
        for(int i = 0; i < M; i++){
            int src_offset = outer * mem_unit + i *K;
            memcpy((float16 *)dst + dst_offset, (float16 *)src + src_offset, mem_unit * sizeof(float16));
            dst_offset += mem_unit;
        }
    }
}

void perf_matrixC_to_norm_transposed_layout(void *src, void *&dst, int32_t M, int32_t N){
    if(M == 1){
        dst = src;
        return;
    }
    const int mem_unit = 4;
    int dst_offset = 0;
    for(int outer = 0; outer < N / mem_unit ; outer++){
        for(int j = 0 ; j < mem_unit; j++){
            for(int i = 0; i < M; i++){
                ((float*)dst)[dst_offset] = ((float*) src)[outer * M * mem_unit + i * mem_unit + j];
                // printf("dst_offset: %d, src_offset: %d, dst: %5.f, src: %5.f\n", dst_offset, outer * M * mem_unit + i * mem_unit + j, ((float*)dst)[dst_offset], ((float*) src)[outer * M * mem_unit + i * mem_unit + j]);
                dst_offset++;
            }
        }
    }
}

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);
// prototypes
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type);
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type);
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type);
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type);
const char* rknpu2_matmul_type_to_string(rknn_matmul_type type);
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type);
size_t get_type_size(rknn_matmul_type type);
void dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k) ;

void compute_submat_mul(int64_t m, int64_t k, const void * A_data, void * B_data, ggml_tensor * dst, int64_t row_start, int64_t row_end, int thread_idx, rknn_matmul_type type, const ggml_tensor * src0, const ggml_tensor * src1) ;


struct ggml_backend_rknn_context {
    int n_threads = 1;
};


// rknn tensor type -> rknn matmul type
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_INT8_MM_INT8_TO_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_INT4_MM_INT4_TO_INT16;
        default:
            GGML_ASSERT(0);
    }
}

// rknn matmul type -> string


// rknn tensor type -> string
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT32:
            return "FLOAT32";
        case RKNN_TENSOR_FLOAT16:
            return "FLOAT16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        case RKNN_TENSOR_INT32:
            return "INT32";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_UINT16:
            return "UINT16";
        default:
            GGML_ASSERT(0);
    }
}

// ggml type -> rknn tensor type
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type) {
    switch(type) {
        case GGML_TYPE_F32:
            return RKNN_TENSOR_FLOAT32;
        case GGML_TYPE_F16:
            return RKNN_TENSOR_FLOAT16;
        case GGML_TYPE_I8:
            return RKNN_TENSOR_INT8;
        case GGML_TYPE_I16:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

// rknn_matmul_type -> rknn_tensor_type
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return RKNN_TENSOR_FLOAT16;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return RKNN_TENSOR_INT8;
        case RKNN_INT4_MM_INT4_TO_INT16:
            return RKNN_TENSOR_INT4;
        default:
            GGML_ASSERT(0);
    }
}

// rknn_tensor_type -> rknn_tensor_type
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_TENSOR_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_TENSOR_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

static inline struct timespec * timespec_sub(const struct timespec *ts_a, const struct timespec *ts_b, struct timespec * ts_out){
    ts_out->tv_sec = ts_a->tv_sec - ts_b->tv_sec;
    ts_out->tv_nsec = ts_a->tv_nsec - ts_b->tv_nsec;
    if (ts_out->tv_nsec < 0) {
        ts_out->tv_sec--;
        ts_out->tv_nsec += 1000000000;
    }
    return ts_out;
}

static inline unsigned long long timespec_ns(const struct timespec * ts){
    return (unsigned long long)ts->tv_sec * 1000000000ull + (unsigned long long)ts->tv_nsec;
}

static ggml_status ggml_backend_rknn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // printf("rknn graph compute!!!!!!!!\n");
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        struct timespec start_compute_forward;
        clock_gettime(CLOCK_MONOTONIC, &start_compute_forward);
        bool ok = ggml_rk_compute_forward(backend, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        struct timespec end_compute_forward;
        clock_gettime(CLOCK_MONOTONIC, &end_compute_forward);

        printf("Node %d: %s (%s) compute time: %llu ns\n", i, node->name, ggml_op_name(node->op), timespec_ns(timespec_sub(&end_compute_forward, &start_compute_forward, &end_compute_forward)));


        GGML_ASSERT(ok);


    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_rknn2_free(ggml_backend_t backend) {
    //ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend->context;
    // if(ctx != nullptr) delete ctx;
    for(int i = 0 ; i < matmul_kernels_count; i++){
        ggml_rknpu2_matmul_kernel *kernel = &matmul_kernels[i];
        rknn_destroy_mem(kernel->ctx, kernel->A);
        rknn_destroy_mem(kernel->ctx, kernel->B);
        rknn_destroy_mem(kernel->ctx, kernel->C);
        rknn_matmul_destroy(kernel->ctx);
    }

    delete backend;
}

static const char * ggml_backend_rknn_name(ggml_backend_t backend) {
    return "RKNN";

    UNUSED(backend);
}
static void ggml_backend_rknn_free(ggml_backend_t backend) {
    ggml_rknn2_free(backend);

    GGML_UNUSED(backend);
}

static bool has_init_kernel_from_file = false;

void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads){
    GGML_ASSERT(ggml_backend_is_rknn(backend_rknn));
    ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend_rknn -> context;
    ctx->n_threads = n_threads;
   // printf("n_threads: %d\n", n_threads);
    if(!has_init_kernel_from_file){

        std::vector<matrixPair> matrix_pairs;
        bool status = read_shape_pairs_from_json(std::string(CONFIG_DIR) + "/mat_kernel_size.json", matrix_pairs);
        // bool status = true;
        if(!status){
            printf("read shape pairs from json failed!\n");
            exit(-1);
        }
    
        for(matrixPair &matrix_pair : matrix_pairs){
            printf("matrix_pair: (%d, %d), (%d, %d)\n", matrix_pair.src0.row, matrix_pair.src0.col, matrix_pair.src1.row, matrix_pair.src1.col);
            matrix_ctx A = {matrix_pair.src0.row, matrix_pair.src0.col, NULL, "A"};
            matrix_ctx B = {matrix_pair.src1.row, matrix_pair.src1.col, NULL, "B"};
            size_t matrix_A_size = A.row * A.col * sizeof(float16);
            size_t matrix_B_size = B.row * B.col * sizeof(float16);
            int initialized = 0;

            int mod_number = 32 * n_threads;
            for(int i = 0 ; i < n_threads;i++){
                    int split_B_col= B.col / 32 / 3 * 32;
                    if(i == n_threads - 1)
                        split_B_col = B.col - (n_threads - 1) * split_B_col;
                    printf("split_B_col: %d, i: %d\n", split_B_col, i);
                        ggml_rknpu2_matmul_kernel * kernel = ggml_rknpu2_matmul_kernel_create(
                        A.data, 
                        B.data, 
                        matrix_A_size, 
                        matrix_B_size, 
                        A.row, 
                        A.col, 
                        split_B_col, 
                        RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, 
                        i, 
                        initialized,
                        true);
            }
        }
        has_init_kernel_from_file = true;
        for(int i = 0 ; i < matmul_kernels_count; i++){
            printf("kernel %d:\n", i);
            printf("dims: %d, %d, %d\n", matmul_kernels[i].info.M, matmul_kernels[i].info.K, matmul_kernels[i].info.N);
        }
    }
    // printf("set n threads done\n");
}

static ggml_backend_i ggml_backend_rknn_i = {
    /* .get_name                = */ ggml_backend_rknn_name,
    /* .free                    = */ ggml_backend_rknn_free,
    /* .set_tensor_async        = */ NULL,  /* ggml_backend_opencl_set_tensor_async */
    /* .get_tensor_async        = */ NULL,  /* ggml_backend_opencl_get_tensor_async */
    /* .cpy_tensor_async        = */ NULL,  /* ggml_backend_opencl_cpy_tensor_async */
    /* .synchronize             = */ NULL,  /* ggml_backend_opencl_synchronize */
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rknn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};
static int ggml_backend_rknn_n_devices = 1;
static const char * ggml_backend_rknn_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNN";

    GGML_UNUSED(reg);
}
static size_t ggml_backend_rknn_reg_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_rknn_n_devices;

    GGML_UNUSED(reg);
}
static const char * ggml_backend_rknn_device_get_name(ggml_backend_dev_t dev) {
    return "RKNN";
    GGML_UNUSED(dev);
}
static const char * ggml_backend_rknn_device_get_description(ggml_backend_dev_t dev) {
    #if defined(GGML_BLAS_USE_ACCELERATE)
        return "Accelerate";
    #elif defined(GGML_BLAS_USE_MKL)
        return "MKL";
    #elif defined(GGML_BLAS_USE_BLIS)
        return "BLIS";
    #elif defined(GGML_BLAS_USE_NVPL)
        return "NVPL";
    #elif defined(OPENBLAS_VERSION)
        return "OpenBLAS";
    #else
        return "RKNN";
    #endif

    GGML_UNUSED(dev);
}
static void ggml_backend_rknn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        *free = info.freeram;
        *total = info.totalram;
        *free = *free * info.mem_unit;
        *total = *total * info.mem_unit;
    } else {
        std::cout<< "sysinfo failed" << "\n";
    }

    GGML_UNUSED(dev);
}
static enum ggml_backend_dev_type ggml_backend_rknn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}
static void ggml_backend_rknn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknn_device_get_name(dev);
    props->description = ggml_backend_rknn_device_get_description(dev);
    props->type        = ggml_backend_rknn_device_get_type(dev);
    ggml_backend_rknn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { // check src/llama-context.cpp:276 
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}
static ggml_backend_t ggml_backend_rknn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_rknn_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}
static ggml_backend_buffer_type_t ggml_backend_rknn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}
static ggml_backend_buffer_t ggml_backend_rknn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}
static bool ggml_backend_rknn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            // printf("op->name: %s\n", op->name);
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            const struct ggml_tensor * dst = op;
            const int64_t ne00 = src0->ne[0]; // k
            const int64_t ne01 = src0->ne[1]; // m
            const int64_t ne10 = src1->ne[0]; // k
            const int64_t ne11 = src1->ne[1]; // n
            const int64_t ne0 = dst->ne[0]; // m
            const int64_t ne1 = dst->ne[1]; // n


            bool result = true;

            if(dst->type != GGML_TYPE_F32){
                result = false;
            }
            result = false;


            for(matrixPair &matrix_pair : support_matrices){
                matrix_ctx A = {matrix_pair.src0.row, matrix_pair.src0.col, NULL, "A"};
                matrix_ctx B = {matrix_pair.src1.row, matrix_pair.src1.col, NULL, "B"};
                if(A.row == ne11 && A.col == ne10 && B.row == ne00 && B.col == ne01){
                    result = true;
                    break;
                }
            }
            if(result == true){
                printf("ne00: %ld, ne01: %ld, ne10: %ld, ne11: %ld, ne0: %ld, ne1: %ld\n", ne00, ne01, ne10, ne11, ne0, ne1);

                if(ne01 == 8192){
                        // printf("op->name: %s\n", op->name);
                        // printf("match: ffn_up-0: %d\n", std::strcmp(op->name, "ffn_up-0") );
                    // const char * pos = std::strstr(op->name, "ffn_up-1");
                    // printf("op->name: %s\n", op->name);
                    // printf("pos: %s\n", pos);
                    if(op->name != NULL && std::strcmp(op->name, "ffn_out-1") == 0) {
                    // if(op->name != NULL && pos) {
                        return true;
                    }
                    else{
                        return false;
                    }
                }
                // printf("op->name: %s\n", op->name);
            }

            return result;

        }

        default:
            return false;

    }

    GGML_UNUSED(dev);
}
static bool ggml_backend_rknn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}
static const struct ggml_backend_device_i ggml_backend_rknn_device_i = {
    /* .get_name             = */ ggml_backend_rknn_device_get_name,
    /* .get_description      = */ ggml_backend_rknn_device_get_description,
    /* .get_memory           = */ ggml_backend_rknn_device_get_memory,
    /* .get_type             = */ ggml_backend_rknn_device_get_type,
    /* .get_props            = */ ggml_backend_rknn_device_get_props,
    /* .init_backend         = */ ggml_backend_rknn_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rknn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_rknn_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_rknn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rknn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
static ggml_backend_dev_t ggml_backend_rknn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_rknn_device = {
        /* .iface   = */ ggml_backend_rknn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,

    };

    // return &g_ggml_backend_rknn_device;
    return &ggml_backend_rknn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_rknn_reg_i = {
    /* .get_name         = */ ggml_backend_rknn_reg_get_name,
    /* .device_count     = */ ggml_backend_rknn_reg_device_count,
    /* .device_get       = */ ggml_backend_rknn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_rknn_get_proc_address,
};

static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_rknn_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}

ggml_backend_reg_t ggml_backend_rknn_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    if (!initialized) {
         reg = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface   = */ ggml_backend_rknn_reg_i,
            /* .context = */ NULL,
         };

        initialized = true;
    }

    return &reg;
}
static const char * ggml_backend_rknn_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "RKNN";
    GGML_UNUSED(buffer_type);
}


static ggml_guid_t ggml_backend_rknn_guid() {
    //c9bdb702-4936-4212-af35-a287d8c02920
    static ggml_guid guid = { 0xc9, 0xbd, 0xb7, 0x02, 0x49, 0x36, 0x42, 0x12, 0xaf, 0x35, 0xa2, 0x87, 0xd8, 0xc9, 0x29, 0x20 };
    return &guid;
}

bool ggml_backend_is_rknn(ggml_backend_t backend){
    return backend != NULL && ggml_guid_matches(backend -> guid, ggml_backend_rknn_guid());
}

// MARK: rknn INIT

ggml_backend_t ggml_backend_rknn_init(void) {
    printf("@ggml-rknn.cpp\n");
    printf("start rknn init!\n");
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_rknn_reg(), 0);
    printf("register the rknn!\n");
    ggml_backend_rknn_context * context = (ggml_backend_rknn_context *) malloc(sizeof(ggml_backend_rknn_context));
    printf("creating the backend!\n");
    

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */  ggml_backend_rknn_guid(),
        /* .interface = */  ggml_backend_rknn_i,
        /* .device    = */  dev,
        /* .context   = */  context
    };
    printf("done!\n");

    bool status = read_shape_pairs_from_json(std::string(CONFIG_DIR) + "/mat_kernel_size.json", support_matrices);
    // bool status = true;
    if(!status){
        printf("read shape pairs from json failed!\n");
        return NULL;
    }
    // printf("ne00: %d, ne01: %d, ne10: %d, ne11: %d, ne0: %d, ne1: %d\n", (int)ne00, (int)ne01, (int)ne10, (int)ne11, (int)ne0, (int)ne1);

    


    return backend;
}

// if the mul mat type is f16 x f16 = f32
// static bool ggml_rk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {

//     const int64_t ne00 = src0->ne[0]; // k
//     const int64_t ne01 = src0->ne[1]; // m
//     const int64_t ne10 = src1->ne[0]; // k
//     const int64_t ne11 = src1->ne[1]; // n
//     const int64_t ne0 = dst->ne[0]; // m
//     const int64_t ne1 = dst->ne[1]; // n

//     // printf("ne00: %d, ne01: %d, ne10: %d, ne11: %d, ne0: %d, ne1: %d\n", (int)ne00, (int)ne01, (int)ne10, (int)ne11, (int)ne0, (int)ne1);
//     //ne00: 960, ne01: 1, ne10: 960, ne11: 2880, ne0: 1, ne1: 2880


//     // if(ne00 %32 != 0 || ne11%32 != 0){
//     //     return false;
//     // }

//     if(dst->type != GGML_TYPE_F32){
//         return false;
//     }
//         std::vector<matrixPair> matrix_pairs;
//     bool status = read_shape_pairs_from_json(std::string(CONFIG_DIR) + "/deepseek-r1-qwen2-1.5B.json", matrix_pairs);
//     if(!status){
//         printf("read shape pairs from json failed!\n");
//         return NULL;
//     }

//     bool pre_created = false;
//     for(matrixPair &matrix_pair : matrix_pairs){
//         // printf("can mul mat matrix_pair: (%d, %d), (%d, %d)\n", matrix_pair.src0.row, matrix_pair.src0.col, matrix_pair.src1.row, matrix_pair.src1.col);
//         matrix_ctx A = {matrix_pair.src0.row, matrix_pair.src0.col, NULL, "A"};
//         matrix_ctx B = {matrix_pair.src1.row, matrix_pair.src1.col, NULL, "B"};
//         if(A.row == ne11 && A.col == ne10 && B.row == ne00 && B.col == ne01){
//             pre_created = true;
//             break;
//         }
//     }
//     // if(pre_created){printf("running on rknn\n");}
//     // else{printf("running on cpu\n");}
//     return pre_created;
//     // return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
//     //         (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
//     //          dst->type == GGML_TYPE_F32 &&
//     //          //(ne0 % 32 == 0 && ne1 % 32 == 0) &&
//     //         //  (ne1 % 32 == 0) &&
//     //         // (ne0 >= 1 && ne1 >= 32 && ne10 >= 32);
//     //         (ne0 >= 1 && ne1 >= 1 && ne10 >= 1);
//     //         //(ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
//     return true;

// }


// static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, int thread_idx, rknpu2::float16 * A_data, rknpu2::float16 * B_data, size_t A_size, size_t B_size) {





// static ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(rknpu2::float16* A_data, rknpu2::float16* B_data, size_t A_size, size_t B_size, int m, int k, int n, rknn_matmul_type type, int core_number, int &initialized){



struct ggml_rknn_data_pack{
    rknn_tensor_type type;
    void* ordered_data;
    int initialized;

    rknn_tensor_mem* B;
};


void copy_submatrix_A(bool is_last,
                    int A_block_row,
                    int A_block_column,
                    int A_row_ele_cnt,
                    int A_column_ele_cnt,
                    rknpu2::float16* sub_A_data,
                    rknpu2::float16* A_block_start,
                    int64_t ori_k,
                    int64_t A_row_cnt){
    if(is_last){
        for(int i = 0 ; i < A_block_row; i++){
            for(int j = 0 ; j < A_block_column; j++){
                if(i < A_row_ele_cnt && j < A_column_ele_cnt){
                    ((rknpu2::float16*)sub_A_data)[i * A_block_column + j] = ((rknpu2::float16*)A_block_start)[i * ori_k + j];
                }
                else{
                    ((rknpu2::float16*)sub_A_data)[i * A_block_column + j] = 0;
                }
            }
        }
    }
    else{
        for(int i = 0; i < A_row_cnt; i++){
            memcpy((rknpu2::float16*)sub_A_data + i * A_block_column, 
                    (rknpu2::float16*)A_block_start + i * ori_k , 
                    A_block_column * sizeof(rknpu2::float16));
        }
    }
}

void copy_submatrix_B(bool is_last,
                    int B_block_row,
                    int B_block_column,
                    int B_row_ele_cnt,
                    int B_column_ele_cnt,
                    rknpu2::float16* sub_B_data,
                    rknpu2::float16* B_block_start,
                    int64_t sub_n){
    if(is_last){
        for(int i = 0 ; i < B_block_row; i++){
            for(int j = 0 ; j < B_block_column; j++){
                if(i < B_row_ele_cnt && j < B_column_ele_cnt){
                    ((rknpu2::float16*)sub_B_data)[i * B_block_column + j] = ((rknpu2::float16*)B_block_start)[i * sub_n + j];
                }
                else{
                    ((rknpu2::float16*)sub_B_data)[i * B_block_column + j] = 0;
                }
            }
        }
    }
    else
    {
        for(int i = 0; i < B_block_row; i++){
            memcpy((rknpu2::float16*)sub_B_data + i * B_block_column , 
                    (rknpu2::float16*)B_block_start + i * sub_n, 
                    B_block_column * sizeof(rknpu2::float16));
        }
    }
}


size_t get_type_size(rknn_matmul_type type){
    switch(type){
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return sizeof(rknpu2::float16);
        case RKNN_INT8_MM_INT8_TO_INT32:
            return sizeof(int8_t);
        default:
            GGML_ASSERT(0);
    }
}

void debug_A_block_start_B_block_start_data(int A_row_cnt,
                                            int B_row_ele_cnt,
                                            int B_column_ele_cnt,
                                            int A_block_column,
                                            int B_block_column,
                                            int B_block_row,
                                            int A_block_row,
                                            void * A_block_start,
                                            void * B_block_start,
                                            rknpu2::float16* sub_A_data,
                                            rknpu2::float16* sub_B_data,
                                            int ori_k,
                                            int sub_n){
    // check A_block_start and B_block_start data

    printf("A_block_start:\n");
    for(int i = 0; i < A_row_cnt; i++){
        for(int j = 0; j < ori_k; j++){
            printf("%4.f ", (float)((rknpu2::float16*)A_block_start)[i * ori_k + j]);
        }
        printf("\n");
    }
    printf("B_block_start:\n");
    for(int i = 0; i < B_row_ele_cnt; i++){
        for(int j = 0; j < B_column_ele_cnt; j++){
            printf("%4.f ", (float)((rknpu2::float16*)B_block_start)[i * sub_n + j]);
        }
        printf("\n");
    }
    // copy the data to the new matrix
    // use memcpy to save time



    printf("sub_A_data:\n");
    for(int i = 0; i < A_row_cnt; i++){
        for(int j = 0; j < A_block_column; j++){
            printf("%4.f ", (float)((rknpu2::float16*)sub_A_data)[i * A_block_column + j]);
        }
        printf("\n");
    }
    printf("sub_B_data:\n");
    for(int i = 0; i < B_block_row; i++){
        for(int j = 0; j < B_block_column; j++){
            printf("%4.f ",(float)((rknpu2::float16*)sub_B_data)[i * B_block_column + j]);
        }
        printf("\n");
    }

}

void debug_A_data_B_data(const void * A_data,
                        void * B_data,
                        int m,
                        int k,
                        int n){
    printf("A_data:\n");
    for(int i = 0 ; i < m; i++){
        for(int j = 0 ; j < k; j++){
            printf("%4.f ", (float)((rknpu2::float16*)A_data)[i * k + j]);
        }
        printf("\n");
    }
    printf("B_data:\n");
    for(int i = 0 ; i < k; i++){
        for(int j = 0 ; j < n; j++){
            printf("%4.f ", (float)((rknpu2::float16*)B_data)[i * n + j]);
        }
        printf("\n");
    }
}
void debug_C_block(void * C_block,
                    int m,
                    int n){
    printf("C_block:\n");
    for(int i = 0 ; i < m; i++){
        for(int j = 0 ; j < n; j++){
            printf("%4.f ", (float)((rknpu2::float16*)C_block)[i * n + j]);
        }
        printf("\n");
    }
}

void dump_time_usage(double prepare_data_time, in_kernel_time &kernel_time, double total_run_time);
void dump_matrix_shape(const int64_t A_row_00, const int64_t A_col_00, const int64_t A_row_01, const int64_t A_col_01, const int64_t A_row_10, const int64_t A_col_10, const int64_t A_row_11, const int64_t A_col_11, const int64_t B_row_00, const int64_t B_col_00, const int64_t B_row_01, const int64_t B_col_01, const int64_t B_row_10, const int64_t B_col_10, const int64_t B_row_11, const int64_t B_col_11);
void second_split(int64_t dst_n, int64_t m, int64_t k, const void *A_data, int64_t ori_k, void *B_data, rknn_matmul_type type, int thread_idx, ggml_tensor *dst);
void naive_split(int64_t row_end, int64_t row_start, int64_t m, int64_t k, const void *A_data, void *B_data, rknn_matmul_type type, int thread_idx, int64_t dst_n, ggml_tensor *dst);
void side_matrix_mulmat_process(void *pad_A00, void *pad_B00, int &C_tile, const matrix_ctx &A00_ctx, const matrix_ctx &B00_ctx, rknn_matmul_type type, int thread_idx, ggml_tensor *dst, int64_t n, in_kernel_time &kernel_time, int id, int offset_col, int offset_row);
void side_matrix_mulmat_process(void *pad_A00, void *pad_B00, int &C_tile, const matrix_ctx &A00_ctx, const matrix_ctx &B00_ctx, rknn_matmul_type type, int thread_idx, ggml_tensor *dst, int64_t n, in_kernel_time &kernel_time, int id, int offset_col, int offset_row, bool matrix_B00_need_set_io);
void side_matrix_mulmat_process(matmul_ctx &A00_B00, ggml_tensor *dst, in_kernel_time &kernel_time, int offset_col, int offset_row, int & C_tile);

// MARK: Matmul


void compute_submat_mul(int64_t m, // matrix A row
                        int64_t k, // matrix B row
                        void *A_data,
                        void *B_data,
                        ggml_tensor *dst,
                        int64_t row_start,
                        int64_t row_end,
                        int thread_idx,
                        rknn_matmul_type type,
                        int64_t dst_n,
                        int64_t ori_k,
                        ggml_tensor *src0,
                        ggml_tensor *src1)
{
    bool split_matrix= false;
    bool second_split_flag = false;
    // printf("row_end: %ld, row_start: %ld, m: %ld, k: %ld, dst_n: %ld\n", row_end, row_start, m, k, dst_n);
    int64_t n = row_end - row_start;
    // int64_t k = ori_k;
    k = ori_k;

    // printf("m: %d, k: %d, n: %d\n",  (int)m, (int)k, (int)n);
    // memset(dst->data, 0, m * n * sizeof(float));

    // >>>>>>>>>>>>>>>Debug>>>>>>>>>>>>>>>.
    // check_A_B_data(m, k, A_data, n, B_data);
    // <<<<<<<<<<<<<<<<<<<<<done<<<<<<<<<<<<<<<<<<<<<<<<<

    // const int64_t A_row_00 = m / 32 * 32;
    const int64_t A_row_00 = m;
    const int64_t A_row_01 = A_row_00;
    // const int64_t A_row_10 = m - A_row_00;
    // const int64_t A_row_11 = A_row_10;

    const int64_t A_col_00 = k / 32 * 32;
    const int64_t A_col_01 = k - A_col_00;
    // const int64_t A_col_10 = A_col_00;
    // const int64_t A_col_11 = A_col_01;

    const int64_t B_row_00 = k / 32 * 32;
    const int64_t B_row_01 = B_row_00;
    const int64_t B_row_10 = k - B_row_00;
    const int64_t B_row_11 = B_row_10;

    const int64_t B_col_00 = n / 32 * 32;
    const int64_t B_col_01 = n - B_col_00;
    const int64_t B_col_10 = B_col_00;
    const int64_t B_col_11 = B_col_01;

    // dump_matrix_shape(A_row_00, A_col_00, A_row_01, A_col_01, A_row_10, A_col_10, A_row_11, A_col_11, B_row_00, B_col_00, B_row_01, B_col_01, B_row_10, B_col_10, B_row_11, B_col_11);
    // pad A01, A10, A11, B01, B10, B11

    double prepare_data_time = 0; 
    double total_run_time = 0;
    in_kernel_time kernel_time;
    memset(&kernel_time, 0, sizeof(in_kernel_time));

    auto total_run_start = std::chrono::high_resolution_clock::now();

    void * pad_A00 = nullptr; int fixed_A00 = 0;
    void * pad_A01 = nullptr;
    void * pad_A10 = nullptr;
    void * pad_A11 = nullptr;
    void * pad_B00 = nullptr; int fixed_B00 = 0;
    void * pad_B01 = nullptr;
    void * pad_B10 = nullptr;
    void * pad_B11 = nullptr;

    auto prepare_data_start = std::chrono::high_resolution_clock::now();
    bool mat_A_mat_B_in_kernel = false;

    matrix_ctx A_ctx = {A_row_00, A_col_00, pad_A00, "A"};
    matrix_ctx B_ctx = {B_row_00, B_col_00, pad_B00, "B"};

    // mat_info mat_A = {A_row_00, A_col_00, FLOAT16, pad_A00, true};
    // mat_info mat_B = {B_row_00, B_col_00, FLOAT16, pad_B00, false};
    void ** ptr_pad_A00 = &pad_A00;
    void ** ptr_pad_B00 = &pad_B00;
    mat_info mat_A = mat_info(A_row_00, A_col_00, FLOAT16, ptr_pad_A00, true);
    mat_info mat_B = mat_info(B_row_00, B_col_00, FLOAT16, ptr_pad_B00, false);

    
    matmul_ctx A00_B00 = matmul_ctx(mat_A, mat_B, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, thread_idx, dst_n);

    // ggml_rknpu2_matmul_kernel * tmp_kernel = ggml_rknpu2_matmul_kernel_find(A_ctx, B_ctx, type, thread_idx);
    ggml_rknpu2_matmul_kernel * tmp_kernel = ggml_rknpu2_matmul_kernel_find(A00_B00);


    if(tmp_kernel != NULL) mat_A_mat_B_in_kernel = true;
    // printf("mat_A_mat_B_in_kernel: %d\n", mat_A_mat_B_in_kernel);   


    if(A_row_00 != 0 && A_col_00 != 0){
        matrixA_to_perf_layout(A_data, pad_A00, A_row_00, A_col_00);
    }
    // printf("pad_A00:\n");
    // check_pad(A_row_00, A_col_00, pad_A00);
    // printf("Check ori_data \n");
    // check_pad(A_row_00, A_col_00, *(mat_A.ori_data));

    bool matrix_B00_need_set_io = true;
    if(B_row_00 != 0 && B_col_00 != 0){
        // goals in this if condition:
        // 1. check if need set io
        // 2. pad_B00 is ready
        pad_B00 = B_data;
        // printf("pad_B00: %p\n", pad_B00);
        // check_pad(B_row_00, B_col_00, pad_B00);
        if(mat_A_mat_B_in_kernel){// make sure tmp_kernel != NULL
            fixed_B00 = 1; //B00's data should not be released after running
            if(tmp_kernel->B_data == pad_B00){
                // pad_B00 is already in kernel, do not need to set io
                matrix_B00_need_set_io = false;
                tmp_kernel->B_is_copied = true;
            }else{
                matrix_B00_need_set_io = true;
                tmp_kernel->B_is_copied = false;
            }
        }
    }
    // printf("thread_idx: %d\n", thread_idx);
    // printf("pad_A00: %p\n", pad_A00);
    // check_pad(A_row_00, A_col_00, pad_A00);
    // printf("pad_B00:\n");
    // check_pad(B_row_00, B_col_00, pad_B00);

    A00_B00.matrix_B00_need_set_io = matrix_B00_need_set_io;
        
        // take A01 as an example
        // TODO: Abstract the code
        // process A01
    pad_side_matrix(A_row_01, A_col_01, A_data, A_row_00, k, pad_A01, A_col_00);
    
    bool is_A = true;

    if(B_row_01 != 0 && B_col_01 != 0){
        void * B01_start = (rknpu2::float16*)B_data + B_row_00 * n;
        // pad_B01 = pad_sub_matrix(B_row_01, B_col_01, B_col_00, B_data, n, false);
        void * tmp_ptr = pad_sub_matrix(B_row_01, B_col_01, B_col_00, B_data, n, false);
        int pad_row = B_row_01 < 32 ? 32 : B_row_01 /32 * 32;
        int pad_col = B_col_01 < 32 ? 32 : B_col_01 /32 * 32;
        pad_B01 = malloc(pad_row * pad_col * sizeof(rknpu2::float16));
        transposed_matrix_to_perf_layout_multi_threads(tmp_ptr, pad_B01, pad_row, pad_col, 32, 16);
        // printf("pad_B01:\n");
        // check_pad(B_row_01, 32, pad_B01);
    }

    if(B_row_10 != 0 && B_col_10 != 0){
        void * B10_start = (rknpu2::float16*)B_data + B_row_00 * n;
        // pad_B10 = pad_sub_matrix(B_row_10, B_col_10, 0, B10_start, n, false);
        void * tmp_ptr = pad_sub_matrix(B_row_10, B_col_10, 0, B_data, n, false);
        int pad_row = B_row_10 < 32 ? 32 : B_row_10 /32 * 32;
        int pad_col = B_col_10 < 32 ? 32 : B_col_10 /32 * 32;
        pad_B10 = malloc(pad_row * pad_col * sizeof(rknpu2::float16));
        transposed_matrix_to_perf_layout_multi_threads(tmp_ptr, pad_B10, pad_row, pad_col, 32, 16);
        // printf("pad_B10:\n");
        // check_pad(32, B_col_10, pad_B10);
    }

    if(B_row_11 != 0 && B_col_11 != 0){
        void * B11_start = (rknpu2::float16*)B_data + B_row_00 * n;
        // pad_B11 = pad_sub_matrix(B_row_11, B_col_11, B_col_10, B11_start, n, false);
        void * tmp_ptr = pad_sub_matrix(B_row_11, B_col_11, B_col_10, B_data, n, false);
        int pad_row = B_row_11 < 32 ? 32 : B_row_11 /32 * 32;
        int pad_col = B_col_11 < 32 ? 32 : B_col_11 /32 * 32;
        pad_B11 = malloc(pad_row * pad_col * sizeof(rknpu2::float16));
        transposed_matrix_to_perf_layout_multi_threads(tmp_ptr, pad_B11, pad_row, pad_col, 32, 16);
        // printf("pad_B11:\n");
        // check_pad(32, 32, pad_B11);
    }

    auto prepare_data_end = std::chrono::high_resolution_clock::now();
    auto prepare_data_duration = std::chrono::duration_cast<std::chrono::microseconds>(prepare_data_end - prepare_data_start);
    prepare_data_time = prepare_data_duration.count();
    // printf("1357: prepare_data_duration: %ld us\n", prepare_data_duration.count());

    matrix_ctx A00_ctx = {A_row_00, A_col_00, pad_A00, "A00"};
    matrix_ctx A01_ctx = {A_row_01, A_col_01, pad_A01, "A01"};
    matrix_ctx B00_ctx = {B_row_00, B_col_00, pad_B00, "B00"};
    matrix_ctx B01_ctx = {B_row_01, B_col_01, pad_B01, "B01"};
    matrix_ctx B10_ctx = {B_row_10, B_col_10, pad_B10, "B10"};
    matrix_ctx B11_ctx = {B_row_11, B_col_11, pad_B11, "B11"};

    int C_tile = 0;
    // side_matrix_mulmat_process(pad_A00, pad_B00, C_tile, A00_ctx, B00_ctx, type, thread_idx, dst, n, kernel_time, 1, 0, 0, matrix_B00_need_set_io);

    // printf("A00_B00: %d, %d, %d, %d\n", (int)A_row_00, (int)A_col_00, (int)B_row_00, (int)B_col_00);


    side_matrix_mulmat_process(A00_B00, dst, kernel_time, row_start, 0, C_tile);

    {
        // side_matrix_mulmat_process(pad_A01, pad_B10, C_tile, A01_ctx, B10_ctx, type, thread_idx, dst, n, kernel_time, 2, 0, 0);

        // C_tile = 1;
        // side_matrix_mulmat_process(pad_A00, pad_B01, C_tile, A00_ctx, B01_ctx, type, thread_idx, dst, n, kernel_time, 3, B_col_00, 0);
        // side_matrix_mulmat_process(pad_A01, pad_B11, C_tile, A01_ctx, B11_ctx, type, thread_idx, dst, n, kernel_time, 4, B_col_00, 0);
    }


    auto total_run_end = std::chrono::high_resolution_clock::now();
    auto total_run_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_run_end - total_run_start);
    total_run_time = total_run_duration.count();


    // dump_time_usage(prepare_data_time, kernel_time, total_run_time);

    if(pad_A00!= nullptr && fixed_A00 == 0)
    {
        free(pad_A00);
    }
    if(pad_A01!= nullptr)
    {
        // free(pad_A01);
    }
    if(pad_B00!= nullptr && fixed_B00 == 0)
    {
        free(pad_B00);
    }
    if(pad_B01!= nullptr)
    {
        free(pad_B01);
    }
    if(pad_B10!= nullptr)
    {
        free(pad_B10);
    }
    if(pad_B11!= nullptr)
    {
        free(pad_B11);
    }

}


void side_matrix_mulmat_process(matmul_ctx &A00_B00, ggml_tensor *dst, in_kernel_time &kernel_time, int offset_col, int offset_row, int & C_tile){
    void *pad_A00 = *(A00_B00.mat_A.ori_data);
    void *pad_B00 = *(A00_B00.mat_B.ori_data);
    // printf("check pad_A00:\n");
    // check_pad(A00_B00.mat_A.row, A00_B00.mat_A.col, pad_A00);

    int thread_idx = A00_B00.thread_idx;

    mat_info mat_A = A00_B00.mat_A;
    mat_info mat_B = A00_B00.mat_B;

    int64_t A_row_00 = mat_A.row;
    int64_t A_col_00 = mat_A.col;
    int64_t B_row_00 = mat_B.row;
    int64_t B_col_00 = mat_B.col;

    int64_t A_pad_row_00 = mat_A.pad_row;
    int64_t A_pad_col_00 = mat_A.pad_col;
    int64_t B_pad_row_00 = mat_B.pad_row;
    int64_t B_pad_col_00 = mat_B.pad_col;
    // printf("A_pad_row_00: %d, A_pad_col_00: %d, B_pad_row_00: %d, B_pad_col_00: %d\n", (int)A_pad_row_00, (int)A_pad_col_00, (int)B_pad_row_00, (int)B_pad_col_00);

    int n = A00_B00.ori_n;

    size_t A_size = mat_A.pad_size;
    size_t B_size = mat_B.pad_size;

    rknn_matmul_type type = A00_B00.type;
    bool matrix_B00_need_set_io = A00_B00.matrix_B00_need_set_io;

    int initialized = 0;
    int ret = 0;

    // TODO: change the thread_idx 
    auto create_kernel_start = std::chrono::high_resolution_clock::now();
    // printf("start create kernel inside side_matrix_multiplication\n");
    ggml_rknpu2_matmul_kernel *sub_kernel = ggml_rknpu2_matmul_kernel_create(pad_A00, pad_B00, A_size, B_size, A_pad_row_00, A_pad_col_00, B_pad_col_00, type, thread_idx, initialized);
    sub_kernel->is_using = true;
    // printf("end create kernel inside side_matrix_multiplication\n");
    auto create_kernel_end = std::chrono::high_resolution_clock::now();
    auto create_kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(create_kernel_end - create_kernel_start).count();
    kernel_time.find_kernel_time+= create_kernel_duration;
    if (initialized == 0)
    {
        auto copy_to_mem_start = std::chrono::high_resolution_clock::now();
        // for(int i = 0; i < A_pad_row_01; i++){
        //     for(int j = 0; j < A_pad_col_01; j++){
        //         ((rknpu2::float16*)sub_kernel->A->virt_addr)[i * A_pad_col_01 + j] = ((rknpu2::float16*)pad_A01)[i * A_pad_col_01 + j];
        //         // printf("copying matrix A to sub_kernel->A->virt_addr: %d, %d\n", i, j);
        //     }
        // }
        // printf("start memcpy A\n");
        memcpy(sub_kernel->A->virt_addr, pad_A00, A_pad_row_00 * A_pad_col_00 * sizeof(rknpu2::float16));
        // printf("A is copied\n");
        // printf("sub_kernel->B_is_copied: %d\n", sub_kernel->B_is_copied);
        if(!sub_kernel->B_is_copied){
            // for(int i = 0; i < B_pad_row_10; i++){
            //     for(int j = 0; j < B_pad_col_10; j++){
            //         ((rknpu2::float16*)sub_kernel->B->virt_addr)[i * B_pad_col_10 + j] = ((rknpu2::float16*)pad_B10)[i * B_pad_col_10 + j];
            //         // printf("copying matrix B to sub_kernel->B->virt_addr: %d, %d\n", i, j);
            //     }
            // }
            // printf("pad_b00\n");
            // check_pad(B_row_00, B_col_00, pad_B00);
            // printf("B_pad_row_00: %d, B_pad_col_00: %d\n", (int)B_pad_row_00, (int)B_pad_col_00);
            memcpy(sub_kernel->B->virt_addr, pad_B00, B_pad_row_00 * B_pad_col_00 * sizeof(rknpu2::float16));
                        sub_kernel->B_is_copied = true;
            sub_kernel->B_data = pad_B00;
        }
        // printf("A->virt_addr\n");
        // check_pad(A_row_00, A_col_00, sub_kernel->A->virt_addr);
        // printf("B->virt_addr\n");
        // check_pad(B_row_00, B_col_00, sub_kernel->B->virt_addr);

        auto copy_to_mem_end = std::chrono::high_resolution_clock::now();
        auto copy_to_mem_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_mem_end - copy_to_mem_start).count();
        kernel_time.memcpy_to_kernel_time += copy_to_mem_duration;
    }
    // printf("is initialized: %d\n", initialized);    

    {
        auto set_io_start = std::chrono::high_resolution_clock::now();
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->A, &sub_kernel->io_attr.A);
        if(matrix_B00_need_set_io)
        {
            rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
        }
        // rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->C, &sub_kernel->io_attr.C);
        auto set_io_end = std::chrono::high_resolution_clock::now();
        auto set_io_duration = std::chrono::duration_cast<std::chrono::microseconds>(set_io_end - set_io_start).count();
        kernel_time.set_io_time += set_io_duration;

    }
    // printf("set io done\n");

    {
        // printf("subkernel->A->virt_addr:\n");
        // check_pad(A_row_00, A_col_00, sub_kernel->A->virt_addr);
        // printf("subkernel->B->virt_addr:\n");
        // check_pad(B_row_00, B_col_00, sub_kernel->B->virt_addr);
        int64_t run_start = getCurrentTimeUs();
        ret = rknn_matmul_run(sub_kernel->ctx);
        int64_t run_end = getCurrentTimeUs();
        int64_t run_duration = run_end - run_start;
        kernel_time.run_time += run_duration;
    }

    {
        void* norm_layout_C = malloc(A_row_00 * B_col_00 * sizeof(float));
        perf_matrixC_to_norm_layout(sub_kernel->C->virt_addr, norm_layout_C, A_row_00, B_col_00);
        // printf("norm_layout_C: thread_idx: %d\n",thread_idx);
        // check_pad_float(A_row_00, B_col_00, norm_layout_C);
        // printf("sub_kernel->C->virt_addr thread_idx: %d\n", thread_idx);
        // check_pad_float(A_row_00, B_col_00, sub_kernel->C->virt_addr);
        auto sum_result_start = std::chrono::high_resolution_clock::now();
        // printf("thread_idx: %d, offset_col: %d, offset_row: %d\n", thread_idx, offset_col, offset_row);
        // printf("n: %d\n", n);
        // printf("B_pad_col_00: %d\n", B_pad_col_00);
        if (C_tile == 0)
        {
            // printf("offset_col: %d", offset_col);
            // printf("dst->data: %p\n", dst->data);
            // printf("dst->ne[0]: %d, dst->ne[1]: %d\n", dst->ne[0], dst->ne[1]);
            for (int i = 0; i < A_row_00; i++)
            {
                for (int j = 0; j < B_col_00; j++)
                {
                    float* dst_data = (float *)dst->data + offset_col + offset_row * n;
                    dst_data[i * n + j] += ((float *)norm_layout_C)[i * B_pad_col_00 + j];
                }
            }
            // printf("id: %d\n", id);
            // if(id == 2){
            //     printf("A01 x B10\n");
            // }
            // A00xB00(A_row_01, B_col_10, dst, n);
            // check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
        }
        if (C_tile == 1){
            float * dst_data = (float *)dst->data + offset_col + offset_row * n;
            for (int i = 0; i < A_row_00; i++)
            {
                for (int j = 0; j < B_col_00; j++)
                {
                    dst_data[i * n + j] += ((float *)norm_layout_C)[i * B_pad_col_00 + j];
                }
            }
            // if(id == 3){
            //     printf("A00 x B01\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }else if (id == 4) {
            //     printf("A01 x B11\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }
        }
        sub_kernel->is_using = false;
        auto sum_result_end = std::chrono::high_resolution_clock::now();
        auto sum_result_duration = std::chrono::duration_cast<std::chrono::microseconds>(sum_result_end - sum_result_start).count();
        kernel_time.sum_result_time += sum_result_duration;
    }
}
void side_matrix_mulmat_process(void *pad_A00, void *pad_B00, int &C_tile, const matrix_ctx &A00_ctx, const matrix_ctx &B00_ctx, rknn_matmul_type type, int thread_idx, ggml_tensor *dst, int64_t n, in_kernel_time &kernel_time, int id, int offset_col, int offset_row, bool matrix_B00_need_set_io){
    if (pad_A00 != nullptr && pad_B00 != nullptr)
    {
        side_matrix_multiplication(A00_ctx, B00_ctx, type, thread_idx, C_tile, dst, n, id, offset_col, offset_row, kernel_time, matrix_B00_need_set_io);
    }
}
void side_matrix_mulmat_process(void *pad_A00, void *pad_B00, int &C_tile, const matrix_ctx &A00_ctx, const matrix_ctx &B00_ctx, rknn_matmul_type type, int thread_idx, ggml_tensor *dst, int64_t n, in_kernel_time &kernel_time, int id, int offset_col, int offset_row) {
    side_matrix_mulmat_process(pad_A00, pad_B00, C_tile, A00_ctx, B00_ctx, type, thread_idx, dst, n, kernel_time, id, offset_col, offset_row, true);
}
void dump_matrix_shape(const int64_t A_row_00, const int64_t A_col_00, const int64_t A_row_01, const int64_t A_col_01, const int64_t A_row_10, const int64_t A_col_10, const int64_t A_row_11, const int64_t A_col_11, const int64_t B_row_00, const int64_t B_col_00, const int64_t B_row_01, const int64_t B_col_01, const int64_t B_row_10, const int64_t B_col_10, const int64_t B_row_11, const int64_t B_col_11)
{
    {

        printf("A_row_00: %d, A_col_00: %d, A_row_01: %d, A_col_01: %d\n", (int)A_row_00, (int)A_col_00, (int)A_row_01, (int)A_col_01);
        printf("A_row_10: %d, A_col_10: %d, A_row_11: %d, A_col_11: %d\n", (int)A_row_10, (int)A_col_10, (int)A_row_11, (int)A_col_11);
        printf("B_row_00: %d, B_col_00: %d, B_row_01: %d, B_col_01: %d\n", (int)B_row_00, (int)B_col_00, (int)B_row_01, (int)B_col_01);
        printf("B_row_10: %d, B_col_10: %d, B_row_11: %d, B_col_11: %d\n", (int)B_row_10, (int)B_col_10, (int)B_row_11, (int)B_col_11);
    }
}

void dump_time_usage(double prepare_data_time, in_kernel_time &kernel_time, double total_run_time)
{
    printf("prepare data time: %.f\n", prepare_data_time);
    printf("memcpy to kernel time: %.f\n", kernel_time.memcpy_to_kernel_time);
    printf("find kernel time: %.f\n", kernel_time.find_kernel_time);
    printf("set io time: %.f\n", kernel_time.set_io_time);
    printf("run time: %.f\n", kernel_time.run_time);
    printf("sum result time: %.f\n", kernel_time.sum_result_time);
    printf("total_run_time: %.f\n", total_run_time);
}

void pad_side_matrix(const int64_t A_row_01, const int64_t A_col_01, const void *A_data, const int64_t A_row_00, int64_t k, void *&pad_A01, const int64_t A_col_00)
{
    if (A_row_01 != 0 && A_col_01 != 0)
    {
        void *A01_start = (rknpu2::float16 *)A_data + A_row_00 * k;
        // pad_A01 = pad_sub_matrix(A_row_01, A_col_01, A_col_00, A_data, k, true);
        void * tmp_mat= pad_sub_matrix(A_row_01, A_col_01, A_col_00, A_data, k, true);
        // printf("A_row_01: %d, A_col_01: %d, A_col_00: %d\n", (int)A_row_01, (int)A_col_01, (int)A_col_00);
        int pad_col_01 = A_col_01 < 32 ? 32 : A_col_01 / 32 * 32;
        // printf("pad_col_01: %d\n", (int)pad_col_01);
        matrixA_to_perf_layout(tmp_mat, pad_A01, A_row_01, pad_col_01);
        // printf("pad_A01 done\n");
        // check_pad(A_row_01, 32, pad_A01);
    }
}

// MARK: side matmul

void side_matrix_multiplication(const int64_t A_row_00, const int64_t A_col_00, const int64_t B_row_00, const int64_t B_col_00, void *pad_A00, void *pad_B00, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time, bool matrix_B00_need_set_io)
{
    // int A_pad_row_01 = A_row_01 < 32 ? 32 : A_row_01 / 32 * 32;
    int A_pad_row_00 = A_row_00;
    int A_pad_col_00 = A_col_00 < 32 ? 32 : A_col_00 / 32 * 32;
    int B_pad_row_00 = B_row_00 < 32 ? 32 : B_row_00 / 32 * 32;
    int B_pad_col_00 = B_col_00 < 32 ? 32 : B_col_00 / 32 * 32;

    size_t A_size = A_pad_row_00 * A_pad_col_00 * sizeof(rknpu2::float16);
    size_t B_size = B_pad_row_00 * B_pad_col_00 * sizeof(rknpu2::float16);

    int initialized = 0;
    int ret = 0;

    // TODO: change the thread_idx 
    auto create_kernel_start = std::chrono::high_resolution_clock::now();
    // printf("start create kernel inside side_matrix_multiplication\n");
    ggml_rknpu2_matmul_kernel *sub_kernel = ggml_rknpu2_matmul_kernel_create(pad_A00, pad_B00, A_size, B_size, A_pad_row_00, A_pad_col_00, B_pad_col_00, type, 1, initialized);
    // printf("end create kernel inside side_matrix_multiplication\n");
    auto create_kernel_end = std::chrono::high_resolution_clock::now();
    auto create_kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(create_kernel_end - create_kernel_start).count();
    kernel_time.find_kernel_time+= create_kernel_duration;
    if (initialized == 0)
    {
        auto copy_to_mem_start = std::chrono::high_resolution_clock::now();
        // for(int i = 0; i < A_pad_row_01; i++){
        //     for(int j = 0; j < A_pad_col_01; j++){
        //         ((rknpu2::float16*)sub_kernel->A->virt_addr)[i * A_pad_col_01 + j] = ((rknpu2::float16*)pad_A01)[i * A_pad_col_01 + j];
        //         // printf("copying matrix A to sub_kernel->A->virt_addr: %d, %d\n", i, j);
        //     }
        // }
        memcpy(sub_kernel->A->virt_addr, pad_A00, A_pad_row_00 * A_pad_col_00 * sizeof(rknpu2::float16));
        // printf("A is copied\n");
        // printf("sub_kernel->B_is_copied: %d\n", sub_kernel->B_is_copied);
        if(!sub_kernel->B_is_copied){
            printf("B_is_copied is false, copying B to sub_kernel->B->virt_addr\n");
            // for(int i = 0; i < B_pad_row_10; i++){
            //     for(int j = 0; j < B_pad_col_10; j++){
            //         ((rknpu2::float16*)sub_kernel->B->virt_addr)[i * B_pad_col_10 + j] = ((rknpu2::float16*)pad_B10)[i * B_pad_col_10 + j];
            //         // printf("copying matrix B to sub_kernel->B->virt_addr: %d, %d\n", i, j);
            //     }
            // }
            memcpy(sub_kernel->B->virt_addr, pad_B00, B_pad_row_00 * B_pad_col_00 * sizeof(rknpu2::float16));
            sub_kernel->B_is_copied = true;
        }

        auto copy_to_mem_end = std::chrono::high_resolution_clock::now();
        auto copy_to_mem_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_mem_end - copy_to_mem_start).count();
        kernel_time.memcpy_to_kernel_time += copy_to_mem_duration;
    }

    {
        auto set_io_start = std::chrono::high_resolution_clock::now();
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->A, &sub_kernel->io_attr.A);
        if(matrix_B00_need_set_io)
        {
            rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
        }
        // rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->C, &sub_kernel->io_attr.C);
        auto set_io_end = std::chrono::high_resolution_clock::now();
        auto set_io_duration = std::chrono::duration_cast<std::chrono::microseconds>(set_io_end - set_io_start).count();
        kernel_time.set_io_time += set_io_duration;

    }

    {
        int64_t run_start = getCurrentTimeUs();
        ret = rknn_matmul_run(sub_kernel->ctx);
        int64_t run_end = getCurrentTimeUs();
        int64_t run_duration = run_end - run_start;
        kernel_time.run_time += run_duration;
    }

    {
        void* norm_layout_C = malloc(A_row_00 * B_col_00 * sizeof(float));
        perf_matrixC_to_norm_layout(sub_kernel->C->virt_addr, norm_layout_C, A_row_00, B_col_00);
        auto sum_result_start = std::chrono::high_resolution_clock::now();
        if (C_tile == 0)
        {
            for (int i = 0; i < A_row_00; i++)
            {
                for (int j = 0; j < B_col_00; j++)
                {
                    float* dst_data = (float *)dst->data + offset_col + offset_row * n;
                    dst_data[i * n + j] += ((float *)norm_layout_C)[i * B_pad_col_00 + j];
                }
            }

            // printf("id: %d\n", id);
            // if(id == 2){
            //     printf("A01 x B10\n");
            // }
            // A00xB00(A_row_01, B_col_10, dst, n);
            // check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
        }
        if (C_tile == 1){
            float * dst_data = (float *)dst->data + offset_col + offset_row * n;
            for (int i = 0; i < A_row_00; i++)
            {
                for (int j = 0; j < B_col_00; j++)
                {
                    dst_data[i * n + j] += ((float *)norm_layout_C)[i * B_pad_col_00 + j];
                }
            }
            // if(id == 3){
            //     printf("A00 x B01\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }else if (id == 4) {
            //     printf("A01 x B11\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }
        }
        if (C_tile == 2){
            float * dst_data = (float *)dst->data + offset_col + offset_row * n;
            for (int i = 0; i < A_row_00; i++)
            {
                for (int j = 0; j < B_col_00; j++)
                {
                    dst_data[i * n + j] += ((float *)norm_layout_C)[i * B_pad_col_00 + j];
                }
            }
            // if(id == 5){
            //     printf("A10 x B00\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }else if(id == 6) {
            //     printf("A11 x B01\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }
        }

        if (C_tile == 3){
            float * dst_data = (float *)dst->data + offset_col + offset_row * n;
            for (int i = 0; i < A_row_00; i++)
            {
                for (int j = 0; j < B_col_00; j++)
                {
                    dst_data[i * n + j] += ((float *)norm_layout_C)[i * B_pad_col_00 + j];
                }
            }
            // if(id == 7){
            //     printf("A10 x B01\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }else if (id == 8) {
            //     printf("A11 x B11\n");
            //     A00xB00(A_row_01, B_col_10, dst, n, offset_col, offset_row);
            //     check_A00xB00_CPU(A_pad_row_01, B_pad_col_10, A_pad_col_01, pad_A01, pad_B10, (float *)sub_kernel->C->virt_addr, B_pad_col_10);
            // }
        }
        auto sum_result_end = std::chrono::high_resolution_clock::now();
        auto sum_result_duration = std::chrono::duration_cast<std::chrono::microseconds>(sum_result_end - sum_result_start).count();
        kernel_time.sum_result_time += sum_result_duration;
    }
}

void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time){
    side_matrix_multiplication(A_row_01, A_col_01, B_row_10, B_col_10, pad_A01, pad_B10, type, thread_idx, C_tile, dst, n, id, offset_col, offset_row, kernel_time, true);
}

void side_matrix_multiplication(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time)
{
    side_matrix_multiplication(A.row, A.col, B.row, B.col, A.data, B.data, type, thread_idx, C_tile, dst, n, id, offset_col, offset_row, kernel_time, true);
}

void side_matrix_multiplication(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time, bool matrix_B00_need_set_io)
{
    side_matrix_multiplication(A.row, A.col, B.row, B.col, A.data, B.data, type, thread_idx, C_tile, dst, n, id, offset_col, offset_row, kernel_time, matrix_B00_need_set_io);
}


void check_pad(const int64_t row, const int64_t col, void *pad_A01)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%2.f ", (float)((rknpu2::float16 *)pad_A01)[i * col + j]);
        }
        printf("\n");
    }
}

void * pad_sub_matrix(const int64_t A_row_01, const int64_t A_col_01, int64_t offset, const void *A_data, int64_t k, bool is_A)
{
    int64_t ori_row = A_row_01;
    int64_t ori_col = A_col_01;
    int64_t pad_row = A_row_01 < 32 ? 32 : A_row_01 / 32 * 32;
    int64_t pad_col = A_col_01 < 32 ? 32 : A_col_01 / 32 * 32;

    if(is_A) pad_row = ori_row;
    int64_t pad_size = pad_row * pad_col * sizeof(rknpu2::float16);
    void *pad_matrix = malloc(pad_size);
    memset(pad_matrix, 0, pad_size);
    for (int i = 0; i < ori_row; i++)
    {
        memcpy((rknpu2::float16 *)pad_matrix + i * pad_col,
               (rknpu2::float16 *)A_data + offset + i * k,
               ori_col * sizeof(rknpu2::float16));
    }
    return pad_matrix;
}

void check_A00xB00_CPU(const int64_t A_row_00, const int64_t B_col_00, const int64_t A_col_00, void *sub_A_data, void *sub_B_data, float *dst, int64_t n)
{

    const float eps = 0.1f;
    for (int i = 0; i < A_row_00; i++)
    {
        for (int j = 0; j < B_col_00; j++)
        {
            float sum = 0;
            for (int k = 0; k < A_col_00; k++)
            {
                rknpu2::float16 a_val = ((rknpu2::float16 *)sub_A_data)[i * A_col_00 + k];
                rknpu2::float16 b_val = ((rknpu2::float16 *)sub_B_data)[k * B_col_00 + j];
                sum += (float)a_val * (float)b_val;
                // ugly code
                // sum += (float)(((rknpu2::float16 *)sub_A_data)[i * A_col_00 + k]) * (float)(((rknpu2::float16 *)sub_B_data)[k * B_col_00 + j]);
            }
            if (fabs((dst[i * n + j] - sum) > eps))
            {
                printf("result is wrong, i: %d, j: %d, dst: %f, cpu: %f\n", (int)i, (int)j, dst[i * n + j], sum);
            }
        }
    }
    printf("checked!\n");
}


void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n)
{
    for (int i = 0; i < A_row_00; i++)
    {
        for (int j = 0; j < B_col_00; j++)
        {
            printf("%4.f ", ((float *)dst->data)[i * n + j]);
        }
        printf("\n");
    }
}
void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n, int offset_col, int offset_row)
{
    float *dst_data = (float *)dst->data + offset_col + offset_row * n;
    for (int i = 0; i < A_row_00; i++)
    {
        for (int j = 0; j < B_col_00; j++)
        {
            printf("%2.f ", dst_data[i * n + j]);
        }
        printf("\n");
    }
}

void check_A_B_data(int64_t m, int64_t k, const void *A_data, int64_t n, void *B_data)
{
    {
        printf("A_data:\n");
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                printf("%4.f ", (float)((rknpu2::float16 *)A_data)[i * k + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("B_data:\n");
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                printf("%4.f ", (float)((rknpu2::float16 *)B_data)[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void copy_submatrix_B_process(bool &is_last, int B_row_block, const int B_block_row_cnt, int &B_row_ele_cnt, int64_t ori_k, const int B_block_row, int B_column_block, const int B_block_column_cnt, int &B_column_ele_cnt, int64_t sub_n, const int B_block_column, void *sub_B_data, void *B_block_start, double &memcpy_duration)
{
    auto B_copy_start = std::chrono::high_resolution_clock::now();
    is_last = false;
    if (B_row_block == B_block_row_cnt - 1)
    { // || B_column_block == B_block_column_cnt - 1){
        // last block
        // printf("B_row_block: %d, B_block_row_cnt: %d\n", (int)B_row_block, (int)B_block_row_cnt);
        B_row_ele_cnt = ori_k % B_block_row;
        if (B_row_ele_cnt == 0)
            B_row_ele_cnt = B_block_row;
        is_last = true;
    }
    if (B_column_block == B_block_column_cnt - 1)
    {
        B_column_ele_cnt = sub_n % B_block_column;
        if (B_column_ele_cnt == 0)
            B_column_ele_cnt = B_block_column;
        is_last = true;
    }
    // printf("B_row_ele_cnt: %d, B_column_ele_cnt: %d\n", (int)B_row_ele_cnt, (int)B_column_ele_cnt);

    copy_submatrix_B(is_last, B_block_row, B_block_column, B_row_ele_cnt, B_column_ele_cnt, (rknpu2::float16 *)sub_B_data, (rknpu2::float16 *)B_block_start, sub_n);

    auto B_copy_end = std::chrono::high_resolution_clock::now();
    auto B_copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(B_copy_end - B_copy_start);
    // printf("B_copy time: %lld\n", B_copy_duration.count());
    memcpy_duration += B_copy_duration.count();
}

void copy_submatrix_A_process(int A_row_block, const int A_block_row_cnt, int &A_row_ele_cnt, int64_t m, const int A_block_row, bool &is_last, int A_column_block, const int A_block_column_cnt, int &A_column_ele_cnt, int64_t ori_k, const int A_block_column, void *sub_A_data, void *A_block_start, int A_row_cnt, double &memcpy_duration)
{
    auto A_copy_start = std::chrono::high_resolution_clock::now();
    if (A_row_block == A_block_row_cnt - 1)
    {
        // last block
        A_row_ele_cnt = m % A_block_row;
        if (A_row_ele_cnt == 0)
            A_row_ele_cnt = A_block_row;
        is_last = true;
    }
    if (A_column_block == A_block_column_cnt - 1)
    {
        A_column_ele_cnt = ori_k % A_block_column;
        if (A_column_ele_cnt == 0)
            A_column_ele_cnt = A_block_column;
        is_last = true;
    }

    copy_submatrix_A(is_last, A_block_row, A_block_column, A_row_ele_cnt, A_column_ele_cnt, (rknpu2::float16 *)sub_A_data, (rknpu2::float16 *)A_block_start, ori_k, A_row_cnt);
    auto A_copy_end = std::chrono::high_resolution_clock::now();
    auto A_copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(A_copy_end - A_copy_start);
    // printf("A_copy time: %lld\n", A_copy_duration.count());
    memcpy_duration += A_copy_duration.count();
}

void pad_matrix_A(
    const void * A_data,
    void * A_pad_data,
    int m,
    int k,
    int pad_k
){
    for(int i = 0; i < m; i++){
        for(int j = 0 ; j < pad_k ; j++){
            if(j < k){
                ((rknpu2::float16*)A_pad_data)[i * pad_k + j] = ((rknpu2::float16*)A_data)[i * k + j];
            }else{
                ((rknpu2::float16*)A_pad_data)[i * pad_k + j] = 0;
            }
        }
    }

}

void pad_matrix_B(
    void * B_data,
    void * B_pad_data,
    int n,
    int pad_n,
    int k,
    int pad_k
){
    for(int i = 0; i < pad_n; i++){
        for(int j = 0 ; j < pad_k ; j++){
            if(i < n && j < k){
                ((rknpu2::float16*)B_pad_data)[i * pad_k + j] = ((rknpu2::float16*)B_data)[i * k + j];
            }else{
                ((rknpu2::float16*)B_pad_data)[i * pad_k + j] = 0;
            }
        }
    }
}
void transpose_matrix_A(
    void * A_transposed_data,
    void * A_pad_data,
    int m,
    int k
){
    for(int i = 0; i < m; i++){
        for(int j = 0 ; j < k; j++){
            ((rknpu2::float16*)A_transposed_data)[j * m + i] = ((rknpu2::float16*)A_pad_data)[i * k + j];
        }
    }
}
void transpose_matrix_B(
    void * B_transposed_data,
    void * B_pad_data,
    int pad_n,
    int pad_k
){
    for(int i = 0; i < pad_n; i++){
        for(int j = 0 ; j < pad_k ; j++){
            ((rknpu2::float16*)B_transposed_data)[j * pad_n + i] = ((rknpu2::float16*)B_pad_data)[i * pad_k + j];
        }
    }
}

// MARK: ggml_rk_mul_mat

static void ggml_rk_mul_mat(ggml_backend_t backend, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type inference_type) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const int64_t n = src1->ne[1]; // matrix B columns
    int n_threads = (int)((ggml_backend_rknn_context * )backend->context)->n_threads;
    threads.reserve(n_threads);

    int64_t k = src1->ne[0];
    int64_t m = src0->ne[1];

    int64_t pad_n = n;
    int64_t pad_k = k;

    // if m or k is not aligned to 32, we need to pad the matrix
    if(n % 32 != 0){
        pad_n = (n / 32 + 1) * 32;
    }
    if(k % 32 != 0){
        pad_k = (k / 32 + 1) * 32;
    }

    start = std::chrono::high_resolution_clock::now();

    void * A_data = src0->data;
    void * B_data_f32 = src1->data;
    void * B_data = malloc(n * k * sizeof(rknpu2::float16));
    for(int i = 0 ; i < n * k ; i++)
        ((rknpu2::float16 *)B_data)[i] = GGML_FP32_TO_FP16(((float *)B_data_f32)[i]);


    // printf("A_data: %p, B_data: %p\n", A_data, B_data);
    // check_pad(m, k, A_data);
    void * A_perf_data;
    if(src0->extra == NULL){
        A_perf_data= malloc(m * pad_k * sizeof(rknpu2::float16));
        transposed_matrix_to_perf_layout_multi_threads(A_data, A_perf_data, k, m, 32, 16);
        src0->extra = A_perf_data;
    }else{
        A_perf_data = src0->extra;
    }
    // printf("A_perf_data: %p\n", A_perf_data);
    // check_pad(pad_k, m, A_perf_data);


    memset(dst->data, 0, dst->ne[0] * dst->ne[1] * sizeof(float));

    void * A_transposed_data = A_data;

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    int threads_number = n_threads;

    for(int i = n_threads; i >= 1; i--){
        if(m % (16 * i) != 0){
            threads_number = i;
        }else{
            break;
        }
    }

    for(int t = 0; t < threads_number; t++){
        // int64_t col_start = t * n / n_threads;
        // int64_t col_end = (t + 1) * n / n_threads;
        int64_t col_start = t * m / threads_number / 32 * 32;
        int64_t col_end = (t + 1) * m / threads_number / 32 * 32;
        if (col_end > m){
            col_end = m;
        }
        int64_t sub_n = col_end - col_start;

        // void * A_compute_data = A_data;
        // void * B_compute_data = (rknpu2::float16*)B_data + col_start * k;
        void * A_compute_data = (rknpu2::float16*)A_perf_data+ col_start * k;
        void * B_compute_data = B_data;

        // run the thread;
        threads.emplace_back([m, pad_k, A_compute_data, A_transposed_data, B_compute_data, dst, col_start, col_end, t, inference_type, k, n, src0, src1](){

            //TODO: Change dst_n to the real value
            int64_t dst_n = dst->ne[1];
            // printf("dst_n: %d\n", dst_n);

            compute_submat_mul(dst_n,pad_k, B_compute_data, A_compute_data, dst, col_start, col_end, t, inference_type, m, k, src1, src0);
        });
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // printf("thread creation time: %lld\n", duration.count());


    start = std::chrono::high_resolution_clock::now();
    // wait for all threads to finish
    for (auto & th : threads) {
        th.join();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

typedef void (*ggml_rk_func_t)(ggml_backend_t backend, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type type);

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    // printf("Timestamp: %lld, start ggml_rk_compute_forward called for tensor\n",getCurrentTimeUs());
    ggml_rk_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    if(tensor->op == GGML_OP_MUL_MAT){
        // if(!any_on_device && !ggml_rk_can_mul_mat(tensor->src[0], tensor->src[1], tensor)){
        // if(!any_on_device ){
        //     return false;
        // }
        func = ggml_rk_mul_mat;
    }
    else{
        return false;
    }

    rknn_matmul_type matmul_type;
    matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    // printf("src0->type: %d, src1->type: %d\n", src0->type, src1->type);
    if(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16){
        matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    }else if(src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0){
        matmul_type = RKNN_INT8_MM_INT8_TO_INT32;
    }
    auto start = std::chrono::high_resolution_clock::now();
    func(backend, tensor->src[0], tensor->src[1], tensor, matmul_type);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // printf("Timestamp: %lld, end ggml_rk_compute_forward called for tensor\n",getCurrentTimeUs());
    // printf("total time: %lld\n", duration.count());
    return true;
}
