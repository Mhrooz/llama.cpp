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

#include <string.h>
#include <cstring>

#include <thread>
#include <vector>

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <fstream>
#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <fcntl.h>
#include <sys/sysinfo.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

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




static uint64_t rknpu2_allocated_bytes = 0;

void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n, int offset_col, int offset_row);
void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n);
void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row);

void pad_side_matrix(const int64_t A_row_01, const int64_t A_col_01, const void *A_data, const int64_t A_row_00, int64_t k, void *&pad_A01, const int64_t A_col_00);

void pad_side_matrix(const int64_t A_row_01, const int64_t A_col_01, void *&pad_A01, const int64_t A_col_00, const void *A_data, int64_t k);

void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id);


bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);
// prototypes
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type);
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type);
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type);
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type);
const char* rknpu2_matmul_type_to_string(rknn_matmul_type type);
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type);

void compute_submat_mul(int64_t m, int64_t k, const void * A_data, void * B_data, ggml_tensor * dst, int64_t row_start, int64_t row_end, int thread_idx, rknn_matmul_type type) ;




struct ggml_backend_rknn_context {
    int n_threads = 1;
};

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
};




#define GGML_RKNPU2_MAX_MATMUL_KERNELS 16
static ggml_rknpu2_matmul_kernel matmul_kernels[GGML_RKNPU2_MAX_MATMUL_KERNELS];

static int matmul_kernels_count = 0;
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
static ggml_status ggml_backend_rknn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        bool ok = ggml_rk_compute_forward(backend, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
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

void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads){
    GGML_ASSERT(ggml_backend_is_rknn(backend_rknn));
    ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend_rknn -> context;
    ctx->n_threads = n_threads;
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
        std::cerr << "sysinfo failed" << "\n";
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
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            const struct ggml_tensor * dst = op;
            const int64_t ne00 = src0->ne[0]; // k
            const int64_t ne01 = src0->ne[1]; // m
            const int64_t ne10 = src1->ne[0]; // k
            const int64_t ne11 = src1->ne[1]; // n
            const int64_t ne0 = dst->ne[0]; // m
            const int64_t ne1 = dst->ne[1]; // n

            //ne00: 960, ne01: 1, ne10: 960, ne11: 2880, ne0: 1, ne1: 2880

            bool result = true;

            // if(ne00 %32 != 0 || ne11%32 != 0){
            //     printf("ne00 %d %% 32 != 0 || ne11 %d %% 32 != 0\n", (int)ne00, (int)ne11);
            //     result = false;
            // }

            if(dst->type != GGML_TYPE_F32){
                result = false;
            }

            return result;


            // BLAS usually is only faster for large matrices
            // const struct ggml_tensor * src0 = op->src[0];
            // const struct ggml_tensor * src1 = op->src[1];

            // const int64_t ne10 = src1->ne[0];

            // const int64_t ne0 = op->ne[0];
            // const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            // const int64_t min_batch = 32;

            // bool result = ggml_is_contiguous(src0) &&
            //        ggml_is_contiguous(src1) &&
            //        src1->type == GGML_TYPE_F16 &&
            //        (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
            //        (src0->type == GGML_TYPE_F16 || ggml_get_type_traits(src0->type)->to_float != NULL);
            // printf("result = %d\n", result);
            // return result;
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

    return backend;
}

// if the mul mat type is f16 x f16 = f32
static bool ggml_rk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {

    const int64_t ne00 = src0->ne[0]; // k
    const int64_t ne01 = src0->ne[1]; // m
    const int64_t ne10 = src1->ne[0]; // k
    const int64_t ne11 = src1->ne[1]; // n
    const int64_t ne0 = dst->ne[0]; // m
    const int64_t ne1 = dst->ne[1]; // n

    // printf("ne00: %d, ne01: %d, ne10: %d, ne11: %d, ne0: %d, ne1: %d\n", (int)ne00, (int)ne01, (int)ne10, (int)ne11, (int)ne0, (int)ne1);
    //ne00: 960, ne01: 1, ne10: 960, ne11: 2880, ne0: 1, ne1: 2880

    // if(ne00 %32 != 0 || ne11%32 != 0){
    //     return false;
    // }

    if(dst->type != GGML_TYPE_F32){
        return false;
    }
    // return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
    //         (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
    //          dst->type == GGML_TYPE_F32 &&
    //          //(ne0 % 32 == 0 && ne1 % 32 == 0) &&
    //         //  (ne1 % 32 == 0) &&
    //         // (ne0 >= 1 && ne1 >= 32 && ne10 >= 32);
    //         (ne0 >= 1 && ne1 >= 1 && ne10 >= 1);
    //         //(ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
    return true;
}


// static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, int thread_idx, rknpu2::float16 * A_data, rknpu2::float16 * B_data, size_t A_size, size_t B_size) {
static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, int thread_idx, const void * A_data, void * B_data, size_t A_size, size_t B_size) {
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
        )
      return kernel;
  }
  return NULL;
}




// static ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(rknpu2::float16* A_data, rknpu2::float16* B_data, size_t A_size, size_t B_size, int m, int k, int n, rknn_matmul_type type, int core_number, int &initialized){
static ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(const void* A_data, void* B_data, size_t A_size, size_t B_size, int m, int k, int n, rknn_matmul_type type, int core_number, int &initialized){
    ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type, core_number, A_data, B_data, A_size, B_size);
    if(kernel != NULL){
        // printf("find an existed kernel!\n");
        // initialized = 1;
        // return kernel;
    }

    else{
        printf("Creating Kernel inside the function\n");
        printf("parameters: %d, %d, %d, %d\n", m, k, n, type);
        GGML_ASSERT(matmul_kernels_count < GGML_RKNPU2_MAX_MATMUL_KERNELS);

        kernel = &matmul_kernels[matmul_kernels_count++];
        if(matmul_kernels_count % GGML_RKNPU2_MAX_MATMUL_KERNELS == 0)
            matmul_kernels_count = 0;
        memset(kernel, 0, sizeof(ggml_rknpu2_matmul_kernel));

        kernel->thread_idx = core_number;
        kernel->info.M = m;
        kernel->info.K = k;
        kernel->info.N = n;
        kernel->info.type = type;
        kernel->info.B_layout = 0; // B use native layout (weight)
        kernel->info.AC_layout = 0; // A and C use original layout (intermediate)

        printf("Creating RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d) %s\n", m, k, k, n, m, n, rknpu2_matmul_type_to_string(type));
        printf("kernel->ctx: %p\n", &(kernel->ctx));
        printf("kernel->info: %p\n", &(kernel->info));
        printf("kernel->io_attr: %p\n", &(kernel->io_attr));

        int ret = rknn_matmul_create(&(kernel->ctx), &(kernel->info), &(kernel->io_attr));
        GGML_ASSERT(ret == 0);
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
    if(core_number == 0)
        rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_0);
    else if(core_number == 1)
        rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_1);
    else if(core_number == 2)
        rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_2);

    // printf("Created RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d) %s\n", m, k, k, n, m, n, rknpu2_matmul_type_to_string(type));

    // create memory for A, B, C
    // but need to memcpy to Matrix->virt_addr
    // {
    //     auto kernel_mem_create_time = std::chrono::high_resolution_clock::now();
    //     kernel->A = rknn_create_mem(kernel->ctx, kernel->io_attr.A.size);
    //     kernel->B = rknn_create_mem(kernel->ctx, kernel->io_attr.B.size);
    //     kernel->C = rknn_create_mem(kernel->ctx, kernel->io_attr.C.size);
    //     auto kernel_mem_create_time_end = std::chrono::high_resolution_clock::now();   
    //     auto kernel_mem_create_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_mem_create_time_end - kernel_mem_create_time).count();
    //     printf("kernel_mem_create_duration: %ld us\n", kernel_mem_create_duration);

    // }

    {
        kernel->A_data = (void*)A_data;
        kernel->B_data = (void*)B_data;
        kernel->A_size = A_size;
        kernel->B_size = B_size;
    }

    return kernel;
}

static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, rknpu2::float16 * A_data, rknpu2::float16 * B_data, size_t A_size, size_t B_size, int &initialized) {
    return ggml_rknpu2_matmul_kernel_create(A_data, B_data, A_size, B_size, m,k,n,type,1, initialized);
}

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
void compute_submat_mul(int64_t m, // matrix A row
                        int64_t k, // matrix B row
                        const void * A_data,
                        void * B_data,
                        ggml_tensor * dst,
                        int64_t row_start,
                        int64_t row_end,
                        int thread_idx,
                        rknn_matmul_type type,
                        int64_t dst_n,
                        int64_t ori_k) {
    bool split_matrix= false;
    bool second_split = false;
    if(split_matrix) {
        printf("partition B\n");

        // columns of the sub_matrix of B
        int64_t sub_n = row_end - row_start;
        size_t A_size = m * k * sizeof(rknpu2::float16);
        size_t B_size = sub_n * k * sizeof(rknpu2::float16);

        printf("m: %d, k: %d, sub_n: %d\n",  (int)m, (int)k, (int)sub_n);
    
        int initialized = 0;
        // measure the overhead of creating the kernel
        auto start = std::chrono::high_resolution_clock::now();
        ggml_rknpu2_matmul_kernel* sub_kernel = ggml_rknpu2_matmul_kernel_create(A_data, B_data, A_size, B_size, m, k, sub_n, type, thread_idx, initialized);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("kernel creation time: %lld\n", duration.count());

        // sub_kernel->is_using = true;
        
        printf("m*k = %d, k*sub_n = %d\n", (int)(m * k), (int)(k * sub_n));
        start = std::chrono::high_resolution_clock::now();
        if(initialized == 0){
            memcpy(sub_kernel->A->virt_addr, A_data, m * k * sizeof(rknpu2::float16));
            memcpy(sub_kernel->B->virt_addr, B_data, sub_n * k * sizeof(rknpu2::float16));
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("memcpy time: %lld\n", duration.count());

        if(initialized == 0){
            rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->A, &sub_kernel->io_attr.A);
            rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
            rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->C, &sub_kernel->io_attr.C);
        }

        start = std::chrono::high_resolution_clock::now();
        int ret = rknn_matmul_run(sub_kernel->ctx);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("run time: %lld\n", duration.count());

        GGML_ASSERT(ret == 0);

        // print the result of sub_kernel
        printf("result of sub_kernel:\n");
        for(int i = 0 ; i < sub_n; i++){
            for(int j = 0 ; j < m; j++){
                printf("%4.f ", ((float*)sub_kernel->C->virt_addr)[i * m + j]);
            }
            printf("\n");
        }
        printf("\n");

        start = std::chrono::high_resolution_clock::now();



        // write back the result to dst
        // need transposed
        for(int i = 0 ; i < dst_n ; i++){ //row
            for(int j = 0 ; j < m; j++){ //column
                ((float *)dst->data)[i * m + j] = ((float*)sub_kernel->C->virt_addr)[j * sub_n + i];
            }
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("write back time: %lld\n", duration.count());
    }else if(second_split){
        // So now we have B^T x A^T = C^T
        // In the parameter of the function, A is the matrix B in the external program, B is matrix A in the external program
        // In such way, we can send them to the kernel directly, we don't need to transpose the matrix in this function.
        double run_duration = 0;
        double memcpy_duration = 0;
        double kernel_duration = 0;
        double set_io_duration = 0;
        double memcpy_to_result = 0;
        double sum_result_duration = 0;
        double total_duration = 0;
        double create_mem_duration = 0;

        auto total_start = std::chrono::high_resolution_clock::now();

        printf("split the original matrix\n");
        int64_t sub_n = dst_n;
        size_t A_size = m * k * sizeof(rknpu2::float16);
        size_t B_size = sub_n * k * sizeof(rknpu2::float16);

        printf("m: %d, k: %d, sub_n: %d\n",  (int)m, (int)k, (int)sub_n);

        const int A_block_column    = 32;
        const int A_block_row       = 32;
        const int A_block_ele_size  = A_block_column * A_block_row;
        const int B_block_column    = 32;
        const int B_block_row       = 32;
        const int B_block_ele_size  = B_block_column * B_block_row;
        const int C_block_column    = 32;
        const int C_block_row       = 32;
        const int C_block_ele_size  = C_block_column * C_block_row;

        const int A_block_column_cnt    = (k - 1) / A_block_column + 1;
        const int A_block_row_cnt       = (m - 1) / A_block_row + 1;
        const int B_block_column_cnt    = (sub_n - 1) / B_block_column + 1;
        const int B_block_row_cnt       = (k - 1) / B_block_row + 1;

        const int C_block_column_cnt    = B_block_column_cnt;
        const int C_block_row_cnt       = A_block_row_cnt;

        const int block_size            = A_block_ele_size * sizeof(rknpu2::float16);

        // prepare for the computation. alloc once and use multiple times
        auto calloc_C_start = std::chrono::high_resolution_clock::now();
        void * C_data  = calloc(m * sub_n , sizeof(float)); //waiting
        auto calloc_C_end = std::chrono::high_resolution_clock::now();
        auto calloc_C_duration = std::chrono::duration_cast<std::chrono::microseconds>(calloc_C_end - calloc_C_start).count();
        // printf("calloc_C_duration: %lld\n", calloc_C_duration);
        create_mem_duration += calloc_C_duration;
        

        for(int C_row_block = 0; C_row_block < C_block_row_cnt; C_row_block++){
            for(int C_column_block = 0; C_column_block < C_block_column_cnt; C_column_block++){
                int A_row_cnt = A_block_row;

                int A_last_block_row = m % A_block_row;
                if(A_last_block_row == 0)
                    A_last_block_row = A_block_row;
                if(C_row_block == C_block_row_cnt - 1)
                    A_row_cnt = A_last_block_row;

                std::vector<float*> results(A_block_column_cnt, nullptr); 
                std::vector<float*> results_float(A_block_column_cnt, nullptr); 


                int A_row_ele_cnt = A_block_row;
                int A_column_ele_cnt = A_block_column;

                int B_row_ele_cnt = B_block_row;
                int B_column_ele_cnt = B_block_column;

                for(int A_column_block = 0; A_column_block < A_block_column_cnt; A_column_block++){
                    // printf("C_row_block: %d, C_column_block: %d, A_clolumn_block: %d \n", (int)C_row_block, (int)C_column_block, (int)A_column_block);

                    void * sub_A_data = malloc(A_block_row * A_block_column * sizeof(rknpu2::float16)); //Released
                    void * sub_B_data = malloc(B_block_row * B_block_column * sizeof(rknpu2::float16)); //Released
                    int A_row_block     = C_row_block;
                    int B_row_block     = A_column_block;
                    int B_column_block  = C_column_block;

                    void * A_block_start;
                    A_block_start = (void*)((rknpu2::float16*)A_data  // point starter
                                            + A_row_block * ori_k * A_block_row  // row block offset
                                            + A_column_block * A_block_column); // column block offset
                    void * B_block_start;
                    B_block_start = (void*)((rknpu2::float16*)B_data 
                                            + B_row_block * sub_n * B_block_row 
                                            + B_column_block * B_block_column);

                    bool is_last = false;

                    copy_submatrix_A_process(A_row_block, A_block_row_cnt, A_row_ele_cnt, m, A_block_row, is_last, A_column_block, A_block_column_cnt, A_column_ele_cnt, ori_k, A_block_column, sub_A_data, A_block_start, A_row_cnt, memcpy_duration);

                    copy_submatrix_B_process(is_last, B_row_block, B_block_row_cnt, B_row_ele_cnt, ori_k, B_block_row, B_column_block, B_block_column_cnt, B_column_ele_cnt, sub_n, B_block_column, sub_B_data, B_block_start, memcpy_duration);



                    // >>>>>>>>>>>>>>>>>>>>>>>debug>>>>>>>>>>>>>>>>>>>>>>>
                    // debug_A_block_start_B_block_start_data(A_row_cnt, B_row_ele_cnt, B_column_ele_cnt, A_block_column, B_block_column, B_block_row, A_block_row, A_block_start, B_block_start, (rknpu2::float16*)sub_A_data, (rknpu2::float16*)sub_B_data, ori_k, sub_n);
                    // // >>>>>>>>>>>>>>>>>>>>>>>done>>>>>>>>>>>>>>>>>>>>>>>

                    int initialized = 0;
                    /*
                        ////////////////////////////////////////////////////////////////////////
                        * Step 1:
                        * create matmul info and io_attr
                        ////////////////////////////////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////
                        * Step 2:
                        * set core mask
                        ////////////////////////////////////////////////////////////////////////
                    */
                    auto kernel_find_start = std::chrono::high_resolution_clock::now();
                    ggml_rknpu2_matmul_kernel* sub_kernel = ggml_rknpu2_matmul_kernel_create(sub_A_data, sub_B_data, A_size, B_size, A_row_cnt, A_block_column, B_block_row, type, thread_idx, initialized);
                    auto kernel_find_end = std::chrono::high_resolution_clock::now();
                    auto kernel_find_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_find_end - kernel_find_start);
                    // printf("kernel creation time: %lld\n", kernel_find_duration.count());
                    kernel_duration += kernel_find_duration.count();

                    if(initialized == 0){
                    // printf("A_row_cnt: %d, A_block_column: %d \n", (int)A_row_cnt, (int)A_block_column);
                    // printf("B_block_row: %d, B_block_column: %d \n", (int)B_block_row, (int)B_block_column);
                        /*
                            ////////////////////////////////////////////////////////////////////////
                            * Step 3:
                            * create A, B, C float data
                            ////////////////////////////////////////////////////////////////////////
                        */                        
                        /*
                            ////////////////////////////////////////////////////////////////////////
                            * Step 4:
                            * create A, B, C rknn_tensor mem
                            ////////////////////////////////////////////////////////////////////////
                        */
                        /*
                            ////////////////////////////////////////////////////////////////////////
                            * Step 5:
                            * data cpy to A, B
                            * Assume A and B both use normal layout
                            ////////////////////////////////////////////////////////////////////////
                        */
                        auto copy_to_mem_start = std::chrono::high_resolution_clock::now();
                        memcpy(sub_kernel->A->virt_addr, sub_A_data, A_row_cnt * A_block_column * sizeof(rknpu2::float16));
                        // printf("sub_kernel->A->virt_addr: %p\n", sub_kernel->A->virt_addr);
                        memcpy(sub_kernel->B->virt_addr, sub_B_data, B_block_row* A_block_column * sizeof(rknpu2::float16));
                        auto copy_to_mem_end = std::chrono::high_resolution_clock::now();
                        auto copy_to_mem_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_mem_end - copy_to_mem_start);
                        // printf("copy to mem time: %lld\n", copy_to_mem_duration.count());
                        memcpy_duration += copy_to_mem_duration.count();

                        /*
                            ////////////////////////////////////////////////////////////////////////
                            * Step 6:
                            * set input and output mem
                            ////////////////////////////////////////////////////////////////////////
                        */
                        auto set_io_start = std::chrono::high_resolution_clock::now();
                        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->A, &sub_kernel->io_attr.A);
                        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
                        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->C, &sub_kernel->io_attr.C);
                        auto set_io_end = std::chrono::high_resolution_clock::now();
                        auto set_io_duration_s = std::chrono::duration_cast<std::chrono::microseconds>(set_io_end - set_io_start);
                        // printf("set io time: %lld\n", set_io_duration_s.count());
                        set_io_duration += set_io_duration_s.count();
                    }

                    auto run_start = std::chrono::high_resolution_clock::now();
                    int ret = rknn_matmul_run(sub_kernel->ctx);
                    auto run_end = std::chrono::high_resolution_clock::now();
                    auto r_duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end- run_start);
                    run_duration += r_duration.count();
                    // printf("run time: %lld\n", r_duration.count());


                    // int ret = rknn_matmul_run(sub_kernel->ctx);

                    results[A_column_block] = (float*)sub_kernel->C->virt_addr;
                    // printf("result of sub_kernel:\n");
                    // for(int i = 0 ; i < A_row_cnt; i++){
                    //     for(int j = 0 ; j < A_block_column; j++){
                    //         printf("%4.f ", ((float*)sub_kernel->C->virt_addr)[i * A_block_column + j]);
                    //     }
                    //     printf("\n");
                    // }
                    // printf("done\n");

                    auto memcpy_to_result_start = std::chrono::high_resolution_clock::now();
                    float * float_result = (float*)malloc(A_row_cnt * A_block_column * sizeof(float));
                    memcpy(float_result, results[A_column_block], A_row_cnt * A_block_column * sizeof(float));
                    auto memcpy_to_result_end = std::chrono::high_resolution_clock::now();
                    auto memcpy_to_result_duration = std::chrono::duration_cast<std::chrono::microseconds>(memcpy_to_result_end - memcpy_to_result_start);
                    // printf("memcpy to result time: %lld\n", memcpy_to_result_duration.count());
                    memcpy_duration+= memcpy_to_result_duration.count();

                    results_float[A_column_block] = float_result;
                    if(initialized){
                        free(sub_A_data);
                        free(sub_B_data);
                    }
                }
                int C_offset =  C_row_block * C_block_column_cnt * A_row_cnt * A_block_column
                                + C_column_block * A_block_column;
                // printf("C_offset: %d, dst->ne[0]: %d, dst->ne[1]: %d\n", (int)C_offset, (int)dst->ne[0], (int)dst->ne[1]);
                
                auto create_tmp_result_start = std::chrono::high_resolution_clock::now();
                float * C_block_start = (float*)C_data
                                            + C_row_block * sub_n * C_block_row
                                            + C_column_block * A_block_column;
                float * tmp_result = (float*)malloc(A_row_cnt * A_block_column* sizeof(float)); // waiting 
                memset(tmp_result, 0, A_row_cnt * A_block_column * sizeof(float));
                auto create_tmp_result_end = std::chrono::high_resolution_clock::now();
                auto create_tmp_result_duration = std::chrono::duration_cast<std::chrono::microseconds>(create_tmp_result_end - create_tmp_result_start);
                // printf("create tmp result time: %lld\n", create_tmp_result_duration.count());
                create_mem_duration += create_tmp_result_duration.count();
                // printf("tmp_result1\n");

                auto sum_result_start = std::chrono::high_resolution_clock::now();
                for(int block = 0 ; block < A_block_column_cnt ;block++){
                    for(int i = 0 ; i < A_row_cnt ; i++){
                        for(int j = 0 ; j < A_block_column; j++){
                            tmp_result[i * A_block_column + j] += results_float[block][i * A_block_column + j];
                            // printf("%4.f ", results[block][i * A_block_column + j]);
                        }
                        // printf("\n");
                    }
                }
                auto sum_result_end = std::chrono::high_resolution_clock::now();
                auto sum_result_duration_t = std::chrono::duration_cast<std::chrono::microseconds>(sum_result_end - sum_result_start);
                // printf("sum result time: %lld\n", sum_result_duration_t.count());
                sum_result_duration += sum_result_duration_t.count();

                // >>>>>>>>>>>>>>>>>>>>debug>>>>>>>>>>>>>>>>>>
                // printf("tmp_result\n");
                // for(int i = 0 ; i < A_row_cnt;i++){
                //     for(int j = 0 ; j < A_block_column; j++){
                //         printf("%4.f ", tmp_result[i * A_block_column + j]);
                //     }
                //     printf("\n");
                // }
                // <<<<<<<<<<<<<<<<<<<<<done<<<<<<<<<<<<<<<<<<<<<<

                auto memcpy_to_result_start = std::chrono::high_resolution_clock::now();
                for(int i = 0 ; i < A_row_cnt ; i++){
                    memcpy(C_block_start + i * sub_n, 
                            tmp_result + i * A_block_column , 
                            B_column_ele_cnt * sizeof(float));
                }
                auto memcpy_to_result_end = std::chrono::high_resolution_clock::now();
                auto memcpy_to_result_duration = std::chrono::duration_cast<std::chrono::microseconds>(memcpy_to_result_end - memcpy_to_result_start);
                // printf("memcpy to result time: %lld\n", memcpy_to_result_duration.count());
                memcpy_to_result += memcpy_to_result_duration.count();

                // >>>>>>>>>>>>>>>>>debug>>>>>>>>>>>>>>>>>>>>>>
                // debug_C_block(C_block_start, m, sub_n);
                // <<<<<<<<<<<<<<<<<done<<<<<<<<<<<<<<<<<<<<<<<<<
                // printf("free sub_A\n");
                free(tmp_result);
                // printf("freed\n");

            }

        }
        auto memcpy_to_result_start = std::chrono::high_resolution_clock::now();
        for(int i = 0 ; i < m;i++){
            for(int j = 0 ; j < sub_n ;j++){
                ((float *)dst->data)[i * sub_n + j] = ((float*)C_data)[i * sub_n + j];
            }
        }
        auto memcpy_to_result_end = std::chrono::high_resolution_clock::now();
        auto memcpy_to_result_duration = std::chrono::duration_cast<std::chrono::microseconds>(memcpy_to_result_end - memcpy_to_result_start);
        // printf("memcpy to result time: %lld\n", memcpy_to_result_duration.count());
        memcpy_to_result += memcpy_to_result_duration.count();

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration_end = std::chrono::duration_cast<std::chrono::microseconds>(total_end- total_start);
        // printf("run time: %lld\n", total_duration_end.count());
        total_duration += total_duration_end.count();

        printf("write back the result to dst\n");
        printf("run_duration: %f\n", run_duration);
        printf("memcpy_duration: %f\n", memcpy_duration);
        printf("kernel_duration: %f\n", kernel_duration);
        printf("set_io_duration: %f\n", set_io_duration);
        printf("memcpy_to_result: %f\n", memcpy_to_result);
        printf("sum_result_duration: %f\n", sum_result_duration);
        printf("create_mem_duration: %f\n", create_mem_duration);
        printf("total_duration: %f\n", total_duration);

        printf("dst->data copied\n");
        free(C_data);
    }
    else{
        // int64_t m = m;
        int64_t n = dst_n;
        int64_t k = ori_k;

        printf("m: %d, k: %d, n: %d\n",  (int)m, (int)k, (int)n);
        memset(dst->data, 0, m * n * sizeof(float));

        // >>>>>>>>>>>>>>>Debug>>>>>>>>>>>>>>>.
        // check_A_B_data(m, k, A_data, n, B_data);
        // <<<<<<<<<<<<<<<<<<<<<done<<<<<<<<<<<<<<<<<<<<<<<<<

        const int64_t A_row_00 = m / 32 * 32;
        const int64_t A_row_01 = A_row_00;
        const int64_t A_row_10 = m - A_row_00;
        const int64_t A_row_11 = A_row_10;

        const int64_t A_col_00 = k / 32 * 32;
        const int64_t A_col_01 = k - A_col_00;
        const int64_t A_col_10 = A_col_00;
        const int64_t A_col_11 = A_col_01;

        const int64_t B_row_00 = k / 32 * 32;
        const int64_t B_row_01 = B_row_00;
        const int64_t B_row_10 = k - B_row_00;
        const int64_t B_row_11 = B_row_10;

        const int64_t B_col_00 = n / 32 * 32;
        const int64_t B_col_01 = n - B_col_00;
        const int64_t B_col_10 = B_col_00;
        const int64_t B_col_11 = B_col_01;

        {

            printf("A_row_00: %d, A_col_00: %d, A_row_01: %d, A_col_01: %d\n", (int)A_row_00, (int)A_col_00, (int)A_row_01, (int)A_col_01);
            printf("A_row_10: %d, A_col_10: %d, A_row_11: %d, A_col_11: %d\n", (int)A_row_10, (int)A_col_10, (int)A_row_11, (int)A_col_11);
            printf("B_row_00: %d, B_col_00: %d, B_row_01: %d, B_col_01: %d\n", (int)B_row_00, (int)B_col_00, (int)B_row_01, (int)B_col_01);
            printf("B_row_10: %d, B_col_10: %d, B_row_11: %d, B_col_11: %d\n", (int)B_row_10, (int)B_col_10, (int)B_row_11, (int)B_col_11);
        }
        // pad A01, A10, A11, B01, B10, B11

        double prepare_data_time = 0; 
        double total_run_time = 0;

        auto total_run_start = std::chrono::high_resolution_clock::now();

        void * pad_A00 = nullptr;
        void * pad_A01 = nullptr;
        void * pad_A10 = nullptr;
        void * pad_A11 = nullptr;
        void * pad_B00 = nullptr;
        void * pad_B01 = nullptr;
        void * pad_B10 = nullptr;
        void * pad_B11 = nullptr;



        auto prepare_data_start = std::chrono::high_resolution_clock::now();
        if(A_row_00 != 0 && A_col_00 != 0){
            pad_A00 = malloc(A_row_00 * A_col_00 * sizeof(rknpu2::float16));
            for(int i = 0 ; i < A_row_00; i++){
                memcpy((rknpu2::float16*)pad_A00 + i * A_col_00, 
                        (rknpu2::float16*)A_data + i * k , 
                        A_col_00 * sizeof(rknpu2::float16));
            }
            printf("pad_A00:\n");
            check_pad(A_row_00, A_col_00, pad_A00);
        }

        if(B_row_00 != 0 && B_col_00 != 0){
            pad_B00 = malloc(B_row_00 * B_col_00 * sizeof(rknpu2::float16));
            for(int i = 0 ; i < B_row_00; i++){
                memcpy((rknpu2::float16*)pad_B00 + i * B_col_00, 
                        (rknpu2::float16*)B_data + i * n , 
                        B_col_00 * sizeof(rknpu2::float16));
            }
            printf("pad_B00:\n");
            check_pad(B_row_00, B_col_00, pad_B00);
        }
        
        // take A01 as an example
        // TODO: Abstract the code
        pad_side_matrix(A_row_01, A_col_01, A_data, A_row_00, k, pad_A01, A_col_00);
        
        if(A_row_10 != 0 && A_col_10 != 0){
            void * A10_start = (rknpu2::float16*)A_data + A_row_00 * k;
            pad_A10 = pad_sub_matrix(A_row_10, A_col_10, 0, A10_start, k);
            printf("pad_A10:\n");
            check_pad(32, A_col_10, pad_A10);
        }

        if(A_row_11 != 0 && A_col_11 != 0){
            void * A11_start = (rknpu2::float16*)A_data + A_row_00 * k;
            pad_A11 = pad_sub_matrix(A_row_11, A_col_11, A_col_10, A11_start, k);
            printf("pad_A11: \n");
            check_pad(32,32,pad_A11);
        }

        if(B_row_01 != 0 && B_col_01 != 0){
            void * B01_start = (rknpu2::float16*)B_data + B_row_00 * n;
            pad_B01 = pad_sub_matrix(B_row_01, B_col_01, B_col_00, B_data, n);
            printf("pad_B01:\n");
            check_pad(B_row_01, 32, pad_B01);
        }

        if(B_row_10 != 0 && B_col_10 != 0){
            void * B10_start = (rknpu2::float16*)B_data + B_row_00 * n;
            pad_B10 = pad_sub_matrix(B_row_10, B_col_10, 0, B10_start, n);
            printf("pad_B10:\n");
            check_pad(32, B_col_10, pad_B10);
        }

        if(B_row_11 != 0 && B_col_11 != 0){
            void * B11_start = (rknpu2::float16*)B_data + B_row_00 * n;
            pad_B11 = pad_sub_matrix(B_row_11, B_col_11, B_col_10, B11_start, n);
            printf("pad_B11:\n");
            check_pad(32, 32, pad_B11);
        }

        auto prepare_data_end = std::chrono::high_resolution_clock::now();
        auto prepare_data_duration = std::chrono::duration_cast<std::chrono::microseconds>(prepare_data_end - prepare_data_start);
        prepare_data_time = prepare_data_duration.count();


        

        int C_tile = 0;
        // A00 x B00 
        if(A_row_00!=0 && B_row_00!=0 && A_col_00!=0 && B_col_00!=0)
        {
            side_matrix_multiplication(A_row_00, A_col_00, B_row_00, B_col_00, pad_A00, pad_B00, type, thread_idx, C_tile, dst, n, 2, 0, 0);
        }


        // A01 x B10
        if(pad_A01 != nullptr && pad_B10 != nullptr){
            side_matrix_multiplication(A_row_01, A_col_01, B_row_10, B_col_10, pad_A01, pad_B10, type, thread_idx, C_tile, dst, n, 2, 0, 0);
        }

        // A00 x B01
        if(A_row_00 != 0 && A_col_00 != 0 && pad_B01 != nullptr)
        {
            C_tile = 1;
            side_matrix_multiplication(A_row_00, A_col_00, B_row_01, B_col_01, pad_A00, pad_B01, type, thread_idx, C_tile, dst, n, 3, B_col_00, 0);

        }

        // A01 x B11
        if(pad_A01 != nullptr && pad_B11 != nullptr)
        {
            C_tile = 1;
            side_matrix_multiplication(A_row_01, A_col_01, B_row_11, B_col_11, pad_A01, pad_B11, type, thread_idx, C_tile, dst, n, 4, B_col_00, 0);
        }

        // A10 x B00
        if(pad_A10 != nullptr && pad_B00 != nullptr){
            C_tile = 2;
            side_matrix_multiplication(A_row_10, A_col_10, B_row_00, B_col_00, pad_A10, pad_B00, type, thread_idx, C_tile, dst, n, 5, 0, A_row_00);
        }

        // A11 x B10
        if(pad_A11 != nullptr && pad_B10 != nullptr){
            C_tile = 2;
            side_matrix_multiplication(A_row_11, A_col_11, B_row_10, B_col_10, pad_A11, pad_B10, type, thread_idx, C_tile, dst, n, 6, 0, A_row_00);
        }

        // A10 x B01
        if(pad_A10 != nullptr && pad_B01 != nullptr){
            C_tile = 3;
            side_matrix_multiplication(A_row_10, A_col_10, B_row_01, B_col_01, pad_A10, pad_B01, type, thread_idx, C_tile, dst, n, 7, B_col_00, A_row_00);
        }
        // A11 x B11
        if(pad_A11 != nullptr && pad_B01 != nullptr){
            C_tile = 3;
            side_matrix_multiplication(A_row_11, A_col_11, B_row_11, B_col_11, pad_A11, pad_B11, type, thread_idx, C_tile, dst, n, 8, B_col_00, A_row_00);
        }

        auto total_run_end = std::chrono::high_resolution_clock::now();
        auto total_run_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_run_end - total_run_start);
        total_run_time = total_run_duration.count();

        printf("prepare data time: %.f\n", prepare_data_time);
        printf("total_run_time: %.f\n", total_run_time);
        if(pad_A00!= nullptr)
        {
            free(pad_A00);
        }
        if(pad_A01!= nullptr)
        {
            free(pad_A01);
        }
        if(pad_A10!= nullptr)
        {
            free(pad_A10);
        }
        if(pad_A11!= nullptr)
        {
            free(pad_A11);
        }
        if(pad_B00!= nullptr)
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
}

void pad_side_matrix(const int64_t A_row_01, const int64_t A_col_01, const void *A_data, const int64_t A_row_00, int64_t k, void *&pad_A01, const int64_t A_col_00)
{
    if (A_row_01 != 0 && A_col_01 != 0)
    {
        void *A01_start = (rknpu2::float16 *)A_data + A_row_00 * k;
        pad_A01 = pad_sub_matrix(A_row_01, A_col_01, A_col_00, A_data, k);
        printf("pad_A01:\n");
        check_pad(A_row_01, 32, pad_A01);
    }
}

void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row)
{
    int A_pad_row_01 = A_row_01 < 32 ? 32 : A_row_01 / 32 * 32;
    int A_pad_col_01 = A_col_01 < 32 ? 32 : A_col_01 / 32 * 32;
    int B_pad_row_10 = B_row_10 < 32 ? 32 : B_row_10 / 32 * 32;
    int B_pad_col_10 = B_col_10 < 32 ? 32 : B_col_10 / 32 * 32;

    size_t A_size = A_pad_row_01 * A_pad_col_01 * sizeof(rknpu2::float16);
    size_t B_size = B_pad_row_10 * B_pad_col_10 * sizeof(rknpu2::float16);

    int initialized = 0;
    int ret = 0;

    // TODO: change the thread_idx 
    ggml_rknpu2_matmul_kernel *sub_kernel = ggml_rknpu2_matmul_kernel_create(pad_A01, pad_B10, A_size, B_size, A_pad_row_01, A_pad_col_01, B_pad_col_10, type, 1, initialized);

    if (initialized == 0)
    {
        memcpy(sub_kernel->A->virt_addr, pad_A01, A_pad_row_01 * A_pad_col_01 * sizeof(rknpu2::float16));
        memcpy(sub_kernel->B->virt_addr, pad_B10, B_pad_row_10 * B_pad_col_10 * sizeof(rknpu2::float16));
    }

    {
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->A, &sub_kernel->io_attr.A);
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->C, &sub_kernel->io_attr.C);
    }

    {
        ret = rknn_matmul_run(sub_kernel->ctx);
    }

    {
        if (C_tile == 0)
        {
            for (int i = 0; i < A_row_01; i++)
            {
                for (int j = 0; j < B_col_10; j++)
                {
                    float* dst_data = (float *)dst->data + offset_col + offset_row * n;
                    dst_data[i * n + j] += ((float *)sub_kernel->C->virt_addr)[i * B_pad_col_10 + j];
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
            for (int i = 0; i < A_row_01; i++)
            {
                for (int j = 0; j < B_col_10; j++)
                {
                    dst_data[i * n + j] += ((float *)sub_kernel->C->virt_addr)[i * B_pad_col_10 + j];
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
            for (int i = 0; i < A_row_01; i++)
            {
                for (int j = 0; j < B_col_10; j++)
                {
                    dst_data[i * n + j] += ((float *)sub_kernel->C->virt_addr)[i * B_pad_col_10 + j];
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
            for (int i = 0; i < A_row_01; i++)
            {
                for (int j = 0; j < B_col_10; j++)
                {
                    dst_data[i * n + j] += ((float *)sub_kernel->C->virt_addr)[i * B_pad_col_10 + j];
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
    }
}

void check_pad(const int64_t row, const int64_t col, void *pad_A01)
{
    // for (int i = 0; i < row; i++)
    // {
    //     for (int j = 0; j < col; j++)
    //     {
    //         printf("%2.f ", (float)((rknpu2::float16 *)pad_A01)[i * col + j]);
    //     }
    //     printf("\n");
    // }
}

void * pad_sub_matrix(const int64_t A_row_01, const int64_t A_col_01, int64_t offset, const void *A_data, int64_t k)
{
    int64_t ori_row = A_row_01;
    int64_t ori_col = A_col_01;
    int64_t pad_row = A_row_01 < 32 ? 32 : A_row_01 / 32 * 32;
    int64_t pad_col = A_col_01 < 32 ? 32 : A_col_01 / 32 * 32;
    int64_t pad_size = pad_row * pad_col * sizeof(rknpu2::float16);
    void *pad_matrix = malloc(pad_size);
    memset(pad_matrix, 0, pad_size);
    for (int i = 0; i < ori_row; i++)
    {
        memcpy((rknpu2::float16 *)pad_matrix + i * pad_col,
               (rknpu2::float16 *)A_data + offset + i * k,
               ori_col * sizeof(rknpu2::float16));
    }
    // >>>>>>>debug>>>>>>>>>>>
    
    // <<<<<<<<<done<<<<<<<<<<<<<
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
    void * A_data,
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

static void ggml_rk_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type inference_type) {


    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("initilized threads time: %lld\n", duration.count());

    printf("using multi threads B \n");

    // matrix B has transposed -> matrix B is column major
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

    printf("pad_n: %d, pad_k: %d\n", (int)pad_n, (int)pad_k);
    // pad the matrix
    start = std::chrono::high_resolution_clock::now();

    void * A_data = src0->data;
    void * B_data = src1->data;


    memset(dst->data, 0, dst->ne[0] * dst->ne[1] * sizeof(rknpu2::float16));

    void * A_transposed_data;
    if(m != 1){
         A_transposed_data = malloc(m * k * sizeof(rknpu2::float16));
        transpose_matrix_A(A_transposed_data, A_data, m, k);
        // printf("A_transposed_data:\n");
        // for(int i = 0; i < k; i++){
        //     for(int j = 0; j < m; j++){
        //         printf("%4.f ", (float)((rknpu2::float16*)A_transposed_data)[i * m + j]);
        //     }
        //     printf("\n");
        // }
    }
    else{
        A_transposed_data = A_data;
    }
    // void * B_transposed_data = malloc(k * n * sizeof(rknpu2::float16));
    // transpose_matrix_B(B_transposed_data, B_data, n, k);

    printf("zero padding done!\n");
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("padding time: %lld\n", duration.count());


    //measure the overhead of creating the threads
    for(int t = 0; t < n_threads; t++){

        int64_t col_start = t * n / n_threads;
        int64_t col_end = (t + 1) * n / n_threads;

        int64_t sub_n = col_end - col_start;

        void * A_compute_data = A_data;
        void * B_compute_data = (rknpu2::float16*)B_data + col_start * k;

        // run the thread;
        threads.emplace_back([m, pad_k, A_compute_data, A_transposed_data, B_compute_data, dst, col_start, col_end, t, inference_type, k, n](){

            //TODO: Change dst_n to the real value
            int64_t dst_n = dst->ne[1];
            printf("dst_n: %d\n", dst_n);

            compute_submat_mul(dst_n,pad_k, B_compute_data, A_transposed_data, dst, col_start, col_end, t, inference_type, m, k);
        });
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("thread creation time: %lld\n", duration.count());


    start = std::chrono::high_resolution_clock::now();
    // wait for all threads to finish
    for (auto & th : threads) {
        th.join();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("waiting computation time: %lld\n", duration.count());

    free(A_transposed_data);

    // free(B_transposed_data);

}

typedef void (*ggml_rk_func_t)(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type type);

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggml_rk_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    // save src0 src1's shape to a external file

    //check src0 and src1 data
    // printf("src0 data, src0->ne[1]: %d, src0->ne[0]: %d\n", (int)src0->ne[1], (int)src0->ne[0]);
    // for(int i = 0; i < src0->ne[1]; i++){
    //     for(int j = 0; j < src0->ne[0]; j++){
    //         printf("%2.f ", float(((rknpu2::float16*)src0->data)[i * src0->ne[0] + j]));
    //     }
    //     printf("\n");
    // }

    // printf("src1 data, src1->ne[1]: %d, src1->ne[0]: %d\n", src1->ne[1], src1->ne[0]);
    // for(int i = 0; i < src1->ne[1]; i++){
    //     for(int j = 0; j < src1->ne[0]; j++){
    //         printf("%2.f ", float(((rknpu2::float16*)src1->data)[i * src1->ne[0] + j]));
    //     }
    //     printf("\n");
    // }

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    if(tensor->op == GGML_OP_MUL_MAT){
        if(!any_on_device && !ggml_rk_can_mul_mat(tensor->src[0], tensor->src[1], tensor)){
            return false;
        }
        func = ggml_rk_mul_mat;
    }
    else{
        return false;
    }

    rknn_matmul_type matmul_type;
    matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    if(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16){
        matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    }else if(src0->type == GGML_TYPE_I8 && src1->type == GGML_TYPE_I8){
        matmul_type = RKNN_INT8_MM_INT8_TO_INT32;
    }
    auto start = std::chrono::high_resolution_clock::now();
    func(backend, tensor->src[0], tensor->src[1], tensor, matmul_type);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("total time: %lld\n", duration.count());
    return true;
}
