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
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/sysinfo.h>
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

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);
// prototypes
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type);
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type);
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type);
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type);
const char* rknpu2_matmul_type_to_string(rknn_matmul_type type);
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type);
static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) ;

void compute_submat_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int64_t row_start, int64_t row_end, int thread_idx, rknn_matmul_type type, bool partition_A) ;




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
    printf("computing graph\n");
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
    printf("deleting backends\n");
    // if(ctx != nullptr) delete ctx;
    for(int i = 0 ; i < matmul_kernels_count; i++){
        ggml_rknpu2_matmul_kernel *kernel = &matmul_kernels[i];
        printf("kernels: %d\n", (int)i);
        rknn_destroy_mem(kernel->ctx, kernel->A);
        printf("kernels: %d\n", (int)i);
        // rknn_destroy_mem(kernel->ctx, kernel->B);
        printf("kernels: %d\n", (int)i);
        rknn_destroy_mem(kernel->ctx, kernel->C);
        printf("kernels: %d\n", (int)i);
        rknn_matmul_destroy(kernel->ctx);
        printf("kernels: %d\n", (int)i);
    }

    printf("deleting backends\n");
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

    printf("%s: op = %s\n", __func__, ggml_op_name(op->op));
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

            printf("ne00: %d, ne01: %d, ne10: %d, ne11: %d, ne0: %d, ne1: %d\n", (int)ne00, (int)ne01, (int)ne10, (int)ne11, (int)ne0, (int)ne1);
            //ne00: 960, ne01: 1, ne10: 960, ne11: 2880, ne0: 1, ne1: 2880

            bool result = true;

            // if(ne00 %32 != 0 || ne11%32 != 0){
            //     printf("ne00 %d %% 32 != 0 || ne11 %d %% 32 != 0\n", (int)ne00, (int)ne11);
            //     result = false;
            // }

            if(dst->type != GGML_TYPE_F32){
                printf("dst->type != GGML_TYPE_F32\n");
                result = false;
            }

            printf("result = %d\n", result);
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

    printf("ne00: %d, ne01: %d, ne10: %d, ne11: %d, ne0: %d, ne1: %d\n", (int)ne00, (int)ne01, (int)ne10, (int)ne11, (int)ne0, (int)ne1);
    //ne00: 960, ne01: 1, ne10: 960, ne11: 2880, ne0: 1, ne1: 2880

    // if(ne00 %32 != 0 || ne11%32 != 0){
    //     return false;
    // }

    if(dst->type != GGML_TYPE_F32){
        return false;
    }

    return true;
    // TODO: find the optimal values for these
    // return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
    //         (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
    //          dst->type == GGML_TYPE_F32 &&
    //          //(ne0 % 32 == 0 && ne1 % 32 == 0) &&
    //          (ne1 % 32 == 0) &&
    //         (ne0 >= 1 && ne1 >= 32 && ne10 >= 32);
            //(ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
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
                kernel->thread_idx == thread_idx &&
                kernel->info.type == type &&
                kernel->A_data == A_data &&
                kernel->B_data == B_data &&
                kernel->A_size == A_size &&
                kernel->B_size == B_size
            )
        return kernel;
    }
    return NULL;
}


// static ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(rknpu2::float16* A_data, rknpu2::float16* B_data, size_t A_size, size_t B_size, int m, int k, int n, rknn_matmul_type type, int core_number, int &initialized){
    static ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(const void* A_data, void* B_data, size_t A_size, size_t B_size, int m, int k, int n, rknn_matmul_type type, int core_number, int &initialized){
        ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type, core_number, A_data, B_data, A_size, B_size);
        if(kernel != NULL){
            printf("find an existed kernel!\n");
            initialized = 1;
            return kernel;
        }
    
        printf("Creating Kernel inside the function\n");
        printf("parameters: %d, %d, %d, %d\n", m, k, n, type);
        GGML_ASSERT(matmul_kernels_count < GGML_RKNPU2_MAX_MATMUL_KERNELS);
        kernel = &matmul_kernels[matmul_kernels_count++];
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
        printf("created\n");
        if(core_number == 0)
            rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_0);
        else if(core_number == 1)
            rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_1);
        else if(core_number == 2)
            rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_2);
    
        printf("Created RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d) %s\n", m, k, k, n, m, n, rknpu2_matmul_type_to_string(type));
    
        // create memory for A, B, C
        // but need to memcpy to Matrix->virt_addr
        {
            kernel->A = rknn_create_mem(kernel->ctx, kernel->io_attr.A.size);
            kernel->B = rknn_create_mem(kernel->ctx, kernel->io_attr.B.size);
            kernel->C = rknn_create_mem(kernel->ctx, kernel->io_attr.C.size);
        }
    
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


void compute_submat_mul(int64_t m, // matrix A row
    int64_t k, // matrix B row
    const void * A_data,
    void * B_data,
    ggml_tensor * dst,
    int64_t row_start,
    int64_t row_end,
    int thread_idx,
    rknn_matmul_type type) {
    printf("partition B\n");

    // columns of the sub_matrix of B
    int64_t sub_n = row_end - row_start;
    size_t A_size = m * k * sizeof(rknpu2::float16);
    size_t B_size = sub_n * k * sizeof(rknpu2::float16);

    printf("m: %d, k: %d, sub_n: %d\n",  (int)m, (int)k, (int)sub_n);
    // check the A_data and B_data
    printf("Adata:\n");
    for(int i = 0 ; i < m; i++){
        for(int j = 0 ; j < k; j++){
            printf("%2.f ", float(((rknpu2::float16*)A_data)[i * k + j]));
        }
        printf("\n");
    }

    printf("Bdata:\n");
    for(int i = 0 ; i < sub_n; i++){
        for(int j = 0 ; j < k; j++){
            printf("%2.f ", float(((rknpu2::float16*)B_data)[i * k + j]));
        }
        printf("\n");
    }

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
    for(int i = 0; i < sub_n; i++){
        for(int j = 0; j < m; j++){
            ((float*)dst->data)[(row_start + i) * m + j] = ((float*)sub_kernel->C->virt_addr)[i * m + j];
        }
    }

    // memcpy((float*)dst->data + row_start * dst->ne[0],
    //         sub_kernel->C->virt_addr,
    //         m * sub_n * sizeof(float));
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("write back time: %lld\n", duration.count());

// sub_kernel->is_using = false;

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
    start = std::chrono::high_resolution_clock::now();
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
    // first, we need to allocate the memory for the matrix
    void * A_pad_data = malloc(m * pad_k * sizeof(rknpu2::float16));
    void * B_pad_data = malloc(pad_n * pad_k * sizeof(rknpu2::float16));
    // then, we need to copy the data to the new matrix

    void * A_data = src0->data;
    void * B_data = src1->data;
    for(int i = 0; i < m; i++){
        for(int j = 0 ; j < pad_k ; j++){
            if(j < k){
                ((rknpu2::float16*)A_pad_data)[i * pad_k + j] = ((rknpu2::float16*)A_data)[i * k + j];
            }else{
                ((rknpu2::float16*)A_pad_data)[i * pad_k + j] = 0;
            }
        }
    }
    for(int i = 0; i < pad_n; i++){
        for(int j = 0 ; j < pad_k ; j++){
            if(i < n && j < k){
                ((rknpu2::float16*)B_pad_data)[i * pad_k + j] = ((rknpu2::float16*)B_data)[i * k + j];
            }else{
                ((rknpu2::float16*)B_pad_data)[i * pad_k + j] = 0;
            }
        }
    }

    void * B_transposed_data = malloc(pad_n * pad_k * sizeof(rknpu2::float16));
    // transpose the matrix B
    for(int i = 0; i < pad_n; i++){
        for(int j = 0 ; j < pad_k ; j++){
            ((rknpu2::float16*)B_transposed_data)[j * pad_n + i] = ((rknpu2::float16*)B_pad_data)[i * pad_k + j];
        }
    }

    printf("zero padding done!\n");


    //measure the overhead of creating the threads
    for(int t = 0; t < n_threads; t++){

        int64_t col_start = t * pad_n / n_threads;
        int64_t col_end = (t + 1) * pad_n / n_threads;

        int64_t sub_n = col_end - col_start;

        void * A_compute_data = A_pad_data;
        printf("col_start: %d, col_end: %d\n", (int)col_start, (int)col_end);
        void * B_compute_data = (rknpu2::float16*)B_pad_data + col_start * pad_k;
        printf("B_data: %p\n", B_data);

        printf("m: %d\n", (int)m);
        printf("pad_k: %d\n", (int)pad_k);
        printf("A_compute_data: %p\n", A_compute_data);
        printf("B_compute_data: %p\n", B_compute_data);
        printf("dst: %p\n", dst);
        printf("col_start: %d\n", (int)col_start);
        printf("col_end: %d\n", (int)col_end);
        printf("t: %d\n", t);
        printf("inference_type: %d\n", inference_type);
        // run the thread
        threads.emplace_back([m, pad_k, A_compute_data, B_transposed_data, dst, col_start, col_end, t, inference_type](){
        //compute_submat_mul(m,k, A_data, B_data, dst, col_start, col_end, t, inference_type);
        printf("thread %d: col_start: %d, col_end: %d\n", t, (int)col_start, (int)col_end);

        compute_submat_mul(m,pad_k, A_compute_data, B_transposed_data, dst, col_start, col_end, t, inference_type);
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


}

typedef void (*ggml_rk_func_t)(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type type);


typedef void (*ggml_rk_func_t)(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type type);

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggml_rk_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    printf("ggml_rk_can_mul_mat: %d\n", ggml_rk_can_mul_mat(src0, src1, tensor));

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
