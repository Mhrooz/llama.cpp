## 1. Overview

This document describes the integration of the `RKNPU` backend into the `llama.cpp` inference framework. The integration enables inference acceleration using NPU with RKNN-Toolkit.

The scope includes:

- Backend registration and initialization
- Operator offloading to the backend
- Model loading and execution interface
- Build instructions and runtime usage



## 2. Architecture Overview

- `llama.cpp` uses the `ggml` tensor library to perform low-level operations.
- Backends in `ggml` are modular and support operator-specific offloading.

### Components Added or Modified:

| File                                | Description |
| ----------------------------------- | ----------- |
| `ggml/src/ggml-rknn/ggml-rknn.cpp   |             |
| `ggml/src/ggml-rknn/CMakeLists.cpp` |             |
| `ggml/src/ggml-backend-reg.cpp`     |             |
| `ggml/src/CMakeLists.txt`           |             |
| `ggml/include/ggml-rknn.h`          |             |
| `ggml/include/fp16`                 |             |
| `ggml/include/rknn_api.h`           |             |
| `ggml/include/rknn_custom_op.h`     |             |
| `ggml/include_matmul_api.h`         |             |
| `scripts/mat_kernel_size.json`      |             |
| `lib/librknnrt.so`                  |             |
| `CMakeLists.txt`                    |             |
| `model_related_config.h`            |             |

## 3. Implementation Details

### 3.1 Compilation Files - CMakeLists.txt

For compiling the program, llama.cpp uses CMakeLists to organise files. In order to integrate the API of  `RKNN-Toolkit` to the llama.cpp, we need to modify `CMakeLists.txt` in each folder layer.

###  `ggml/CMakeLists.txt`

@line 197:

```cmake
option(GGML_RKNN 	"ggml: use RKNN"	OFF)
```

This option tells `cmake` to use `ggml_rknn` generate compile files.

@line 268:

```cmake
set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-blas.h
    include/ggml-cann.h
    include/ggml-cpp.h
    include/ggml-rknn.h
    include/ggml-cuda.h
    include/ggml-kompute.h
    include/ggml-opt.h
    include/ggml-metal.h
    include/ggml-rpc.h
    include/ggml-sycl.h
    include/ggml-vulkan.h
    include/gguf.h)
```

Need to tell compiler where the header file is.

### `ggml/src/CMakeLists.txt`

@line 310:

```cmake
ggml_add_backend(rknn)
```

Tell optimizer `rknn` backend should be registered.

### `ggml/src/ggml-rknn/CMakeLists.txt`

This file does following things:

1. add one configure file(for reading specific matrix kernel dimension in the program).
2. add one custom library json from github.
3. set the backend name to `ggml-rknn` then find the right header and source file.

```cmake
configure_file(
    ../../include/model_related_config.h.in
    ${CMAKE_BINARY_DIR}/model_related_config.h
    @ONLY
)
include_directories(${CMAKE_BINARY_DIR})

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

set(TARGET_NAME ggml-rknn)

ggml_add_backend_library(${TARGET_NAME}
                            ggml-rknn.cpp
                            ../../include/ggml-rknn.h)

if(DEFINED GGML_DEBUGING)
    add_compile_definitions(GGML_RKNN_DEBUGING)
endif()

target_link_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
${CMAKE_SOURCE_DIR}/lib/librknnrt.so nlohmann_json::nlohmann_json)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/lib)
```



### 3.2 Backend Registration

### `ggml/src/ggml-backend-reg.cpp`

To register `ggml-rknn` backend, we need to add several lines inside this file:

@line 53:

```c
#ifdef GGML_USE_RKNN
#include "ggml-rknn.h"
#endif 
```

@line 187

```c
 struct ggml_backend_registry {

#ifdef GGML_USE_RKNN
        register_backend(ggml_backend_rknn_reg());
#endif
```

@line 567 - find the ggml-rknn lib.

```c
ggml_backend_load_best("rknn", silent, dir_path);
```

### 3.4 Must-implement data structures

This section will introduce in order to add `ggml-rknn` backend to llama.cpp, what data structures we need to implement.  The codes in this section can be found in `ggml/ggml-rknn/ggml-rknn.cpp`.

#### `ggml_backend_rknn_reg_i`

This struct shows the basic attribution of the backend.  It will be used when the 

 The variables in this struct are function pointers.

```c
static struct ggml_backend_reg_i ggml_backend_rknn_reg_i = {
    /* .get_name         = */ ggml_backend_rknn_reg_get_name,
    /* .device_count     = */ ggml_backend_rknn_reg_device_count,
    /* .device_get       = */ ggml_backend_rknn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_rknn_get_proc_address,
};
```

From top to bottom, the functions are:

##### `ggml_backend_rknn_reg_get_name`

```c
static const char * ggml_backend_rknn_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNN";
    GGML_UNUSED(reg);
}
```

##### `ggml_backend_rknn_reg_device_count`

`ggml_backend_rknn_n_devices` is a global constants. It always be 1 since on rk3588 we only have one 3-core NPU.

```c
static size_t ggml_backend_rknn_reg_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_rknn_n_devices; 
    GGML_UNUSED(reg);
}
```

##### `ggml_backend_rknn_reg_device_get`

In this function, we need another detailed backend descriptor, which is implement by another sturct `ggml_backend_rknn_device_i`. This descriptor is written in the next section.

```c
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

```

#### `ggml_backend_rknn_device_i`

```c
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
```

##### `ggml_backend_rknn_device_get_name`

```c
static const char * ggml_backend_rknn_device_get_name(ggml_backend_dev_t dev) {
    return "RKNN";
    GGML_UNUSED(dev);
}
```

##### `ggml_backend_rknn_device_get_description`

This function returns the name of the device.

```c
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
```

##### `ggml_backend_rknn_device_get_memory`

This API needs to return the memory the device have. Since NPU uses the DRAM, we just simply return s current system's DRAM remains. The DRAM could be seen as unified memory in such context.

```c
static void ggml_backend_rknn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
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
```

##### `ggml_backend_rknn_device_get_type`

This function needs to return the type of device. In llama.cpp, there are three different kinds of `dev_type`:

```c
// @ ggml-backend.h  
enum ggml_backend_dev_type {
      // CPU device using system memory
      GGML_BACKEND_DEVICE_TYPE_CPU,
      // GPU device using dedicated memory
      GGML_BACKEND_DEVICE_TYPE_GPU,
      // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
      GGML_BACKEND_DEVICE_TYPE_ACCEL
  };
```

Since the `RKNPU` needs its own memory inside the DRAM, I chose to return `TYPE_GPU` in this function.

```c
static enum ggml_backend_dev_type ggml_backend_rknn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}
```

##### `ggml_backend_rknn_device_get_props`

This function need to return the properties of the backend. `props->name`, `props->description`, `props->type` has been introduced before. 

The only problem is how should we decide the attributions inside `caps`.

In `src/llama-context.cpp:276`

```c
ggml_backend_dev_get_props(dev, &props);
if (!props.caps.async || !props.caps.events) {
    // device does not support async compute or events
    pipeline_parallel = false;
    break;
}
```

So the `props.caps` determines if the backend supports `pipeline_parallel`. For better debugging, I set the `async` and `events` to false.

 `host_buffer` and `buffer_from_host_ptr` are set to true because the NPU is using DRAM.

```c
static void ggml_backend_rknn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknn_device_get_name(dev);
    props->description = ggml_backend_rknn_device_get_description(dev);
    props->type        = ggml_backend_rknn_device_get_type(dev);
    ggml_backend_rknn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { 
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}
```

##### `ggml_backend_rknn_device_init_backend`

This function describes how backend init the backend. The `ggml_backend_rknn_init()` function is introduced in section 3.3.

```c
static ggml_backend_t ggml_backend_rknn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_rknn_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}
```

#### `ggml_backend_rknn_i`

```c
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
```

## 3.3 Must-implement APIs

The content in this section are mainly about the functions in the `ggml-rknn.h` header file.

###  `GGML_BACKEND_API ggml_backend_t ggml_backend_rknn_init ()`

`ggml_backend_rknn_reg()` will be introduced later.

```c
ggml_backend_t ggml_backend_rknn_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_rknn_reg(), 0);
    ggml_backend_rknn_context * context = (ggml_backend_rknn_context *) malloc(sizeof(ggml_backend_rknn_context));
    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */  ggml_backend_rknn_guid(),
        /* .interface = */  ggml_backend_rknn_i,
        /* .device    = */  dev,
        /* .context   = */  context
    };
    return backend;
}
```



#### `ggml_backend_rknn_guid()`

GUID should be an identifier. This guid is generated by a linux tool.

```c
static ggml_guid_t ggml_backend_rknn_guid() {
    //c9bdb702-4936-4212-af35-a287d8c02920
    static ggml_guid guid = { 0xc9, 0xbd, 0xb7, 0x02, 0x49, 0x36, 0x42, 0x12, 0xaf, 0x35, 0xa2, 0x87, 0xd8, 0xc9, 0x29, 0x20 };
    return &guid;
}
```



### `GGML_BACKEND_API bool ggml_backend_is_rknn(ggml_backend_t backend)`

Use `guid` to check if backend is `ggml-rknn`.

```c
bool ggml_backend_is_rknn(ggml_backend_t backend){
    return backend != NULL && ggml_guid_matches(backend -> guid, ggml_backend_rknn_guid());
}
```

### `GGML_BACKEND_API void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads);`

After checking the backend is `ggml_rknn`, we could 

```c
void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads){
    GGML_ASSERT(ggml_backend_is_rknn(backend_rknn));
    ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend_rknn -> context;
    ctx->n_threads = n_threads;
}
```



### `GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rknn_reg(void);`

This function mainly is used to check if the backend has been initialized. If it has not initialized, return the backend's register information.

```c
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
```



###  `static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name)`

`get_proc_address` usually used in **getting the function or variable address in DLL during the runtime dynamically**. Refernce to [ggml_blas](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-blas/ggml-blas.cpp), I just return the `ggml_backend_set_n_threads` function. Otherwise it returns with `NULL`.

```c
static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_rknn_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}
```

## 4. Matmul implementation & Optimization

## 4.1 Move the copy weights to warm-up stage

```c
ggml_tensor *ggml_backend_backend_name_tensor(...);
```

### 4.4 Operator Coverage

| Operator | Supported | Comments             |
| -------- | --------- | -------------------- |
| MatMul   | ✅         | Offloaded to backend |

## 6. Build Instructions

The build has been tested on the orangepi oultra/pro.

### CMake Build

```bash
mkdir build
cmake -DGGML_RKNN=ON -B build
cmake --build build --config Debug -j 4
```

## 7. Usage Instructions

```bash
sudo chrt -f 90 taskset -c 4-6 build/bin/llama-cli -m /mnt/playground/gguf_models/llama-3.2-1B-Instruct.gguf -n 100 -t 3 -p "Once upon a time" -no-cnv -ngl 20
```

## 8. Known Issues / TODOs

[] 