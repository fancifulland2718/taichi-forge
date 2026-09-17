// Init
PER_AMDGPU_FUNCTION(init, hipInit, unsigned int);

// Device management
PER_AMDGPU_FUNCTION(device_get_count, hipGetDeviceCount, int *);
PER_AMDGPU_FUNCTION(device_get_attribute,
                    hipDeviceGetAttribute,
                    int *,
                    uint32,
                    int);
PER_AMDGPU_FUNCTION(device_get_name, hipDeviceGetName, char *, int, int);
PER_AMDGPU_FUNCTION(device_get_current, hipGetDevice, int *);
PER_AMDGPU_FUNCTION(device_set_current, hipSetDevice, int);

// Stream management
PER_AMDGPU_FUNCTION(stream_create, hipStreamCreateWithFlags, void **, uint32);

// Memory management
PER_AMDGPU_FUNCTION(memcpy_host_to_device,
                    hipMemcpyHtoD,
                    void *,
                    void *,
                    std::size_t);
PER_AMDGPU_FUNCTION(memcpy_device_to_host,
                    hipMemcpyDtoH,
                    void *,
                    void *,
                    std::size_t);
PER_AMDGPU_FUNCTION(memcpy_device_to_device,
                    hipMemcpyDtoD,
                    void *,
                    void *,
                    std::size_t);
PER_AMDGPU_FUNCTION(memcpy,
                    hipMemcpy,
                    void *,
                    void *,
                    std::size_t,
                    unsigned int);
PER_AMDGPU_FUNCTION(memcpy_async,
                    hipMemcpyAsync,
                    void *,
                    void *,
                    std::size_t,
                    unsigned int,
                    void *);
PER_AMDGPU_FUNCTION(memcpy_host_to_device_async,
                    hipMemcpyHtoDAsync,
                    void *,
                    void *,
                    std::size_t,
                    void *);
PER_AMDGPU_FUNCTION(memcpy_device_to_host_async,
                    hipMemcpyDtoHAsync,
                    void *,
                    void *,
                    std::size_t,
                    void *);
PER_AMDGPU_FUNCTION(malloc, hipMalloc, void **, std::size_t);
PER_AMDGPU_FUNCTION(malloc_managed,
                    hipMallocManaged,
                    void **,
                    std::size_t,
                    uint32);
PER_AMDGPU_FUNCTION(memset, hipMemset, void *, int, std::size_t);
PER_AMDGPU_FUNCTION(mem_free, hipFree, void *);
PER_AMDGPU_FUNCTION(mem_get_info, hipMemGetInfo, std::size_t *, std::size_t *);

// Module and kernels
PER_AMDGPU_FUNCTION(module_get_function,
                    hipModuleGetFunction,
                    void **,
                    void *,
                    const char *);
PER_AMDGPU_FUNCTION(module_load_data, hipModuleLoadData, void **, const void *);
PER_AMDGPU_FUNCTION(launch_kernel,
                    hipModuleLaunchKernel,
                    void *,
                    uint32,
                    uint32,
                    uint32,
                    uint32,
                    uint32,
                    uint32,
                    uint32,
                    void *,
                    void **,
                    void **);
PER_AMDGPU_FUNCTION(kernel_get_attribute,
                    hipFuncGetAttribute,
                    int *,
                    uint32,
                    void *);
PER_AMDGPU_FUNCTION(kernel_get_occupancy,
                    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
                    int *,
                    void *,
                    int,
                    size_t);

// Stream management
PER_AMDGPU_FUNCTION(stream_synchronize, hipStreamSynchronize, void *);

// Event management
PER_AMDGPU_FUNCTION(event_create, hipEventCreateWithFlags, void **, uint32);
PER_AMDGPU_FUNCTION(event_destroy, hipEventDestroy, void *);
PER_AMDGPU_FUNCTION(event_record, hipEventRecord, void *, void *);
PER_AMDGPU_FUNCTION(event_synchronize, hipEventSynchronize, void *);
PER_AMDGPU_FUNCTION(event_elapsed_time,
                    hipEventElapsedTime,
                    float *,
                    void *,
                    void *);
