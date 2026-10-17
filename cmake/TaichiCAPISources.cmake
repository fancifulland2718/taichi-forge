# Shared source and header inventory for standalone and runtime-hosted C APIs.
include_guard(GLOBAL)

function(ti_get_c_api_sources OUT_SOURCES OUT_HEADERS)
    set(_sources c_api/src/taichi_core_impl.cpp c_api/src/aot_bindings.cpp
        c_api/src/argument_binding.cpp c_api/src/prepared_execution.cpp
        c_api/src/execution_resources.cpp c_api/src/execution_binding_family.cpp
        c_api/src/native_resource_registry.cpp)
    if(TI_WITH_RUNTIME_C_API)
        list(APPEND _sources c_api/src/aot_runtime_client.cpp)
    endif()
    set(_headers
        c_api/include/taichi/taichi_platform.h
        c_api/include/taichi/taichi_core.h
        c_api/include/taichi/taichi.h
        c_api/include/taichi/taichi_unity.h)
    if(TI_WITH_LLVM)
        list(APPEND _sources c_api/src/taichi_llvm_impl.cpp)
        list(APPEND _headers c_api/include/taichi/taichi_cpu.h)
        if(TI_WITH_CUDA)
            list(APPEND _headers c_api/include/taichi/taichi_cuda.h)
        endif()
    endif()
    if(TI_WITH_OPENGL OR TI_WITH_VULKAN OR TI_WITH_METAL)
        list(APPEND _sources c_api/src/taichi_gfx_impl.cpp)
    endif()
    if(TI_WITH_OPENGL)
        list(APPEND _sources c_api/src/taichi_opengl_impl.cpp)
        list(APPEND _headers c_api/include/taichi/taichi_opengl.h)
    endif()
    if(TI_WITH_METAL)
        list(APPEND _sources c_api/src/taichi_metal_impl.mm)
        list(APPEND _headers c_api/include/taichi/taichi_metal.h)
    endif()
    if(TI_WITH_VULKAN)
        list(APPEND _sources c_api/src/taichi_vulkan_impl.cpp)
        list(APPEND _headers c_api/include/taichi/taichi_vulkan.h)
    endif()
    set(${OUT_SOURCES} ${_sources} PARENT_SCOPE)
    set(${OUT_HEADERS} ${_headers} PARENT_SCOPE)
endfunction()

function(ti_configure_c_api_target TARGET)
    if(TI_WITH_RUNTIME_C_API)
        target_include_directories(${TARGET} PRIVATE
            ${PROJECT_SOURCE_DIR}/external/eigen)
    endif()
    target_include_directories(${TARGET}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c_api/include>
            $<INSTALL_INTERFACE:c_api/include>
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/external/spdlog/include
            ${PROJECT_SOURCE_DIR}/external/FP16/include
            ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
            ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
            ${PROJECT_SOURCE_DIR}/external/volk
            ${PROJECT_SOURCE_DIR}/external/glad/include
            ${PROJECT_SOURCE_DIR}/external/glfw/include)
endfunction()
