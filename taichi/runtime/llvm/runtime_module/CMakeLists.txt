# ./taichi/runtime/llvm/runtime_module/CMakeLists.txt

function(COMPILE_LLVM_RUNTIME rtm_arch)
    message(STATUS "Compiling LLVM byte code file for arch ${rtm_arch}")
    set(runtime_target_flags)
    if (WIN32 AND rtm_arch STREQUAL "cuda")
        # CUDA libdevice uses a 32-bit wchar_t module flag. Windows Clang's
        # host default is 16-bit, which LLVM rejects when linking device IR.
        # This is device bitcode only; preserve the native CPU ABI.
        list(APPEND runtime_target_flags -fno-short-wchar)
    endif()
    # Keep the existing output location for runtime loaders and install rules.
    # Compiler depfiles track transitive project and toolchain headers; a target
    # with only COMMAND would rebuild these bitcode files on every invocation.
    set(runtime_bitcode "${CMAKE_CURRENT_SOURCE_DIR}/runtime_${rtm_arch}.bc")
    set(runtime_command ${CLANG_EXECUTABLE} ${CLANG_OSX_FLAGS} ${runtime_target_flags}
        -c "${CMAKE_CURRENT_SOURCE_DIR}/runtime.cpp" -o "${runtime_bitcode}" -fno-exceptions -emit-llvm
        -std=c++17 -D "ARCH_${rtm_arch}" -I "${PROJECT_SOURCE_DIR}")
    if (CMAKE_GENERATOR MATCHES "Ninja|Makefiles" OR CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        set(runtime_depfile "${CMAKE_CURRENT_BINARY_DIR}/runtime_${rtm_arch}.d")
        add_custom_command(
            OUTPUT "${runtime_bitcode}"
            COMMAND ${runtime_command} -MD -MF "${runtime_depfile}" -MQ "${runtime_bitcode}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runtime.cpp" "${CLANG_EXECUTABLE}"
            DEPFILE "${runtime_depfile}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM
        )
        add_custom_target("generate_llvm_runtime_${rtm_arch}" DEPENDS "${runtime_bitcode}")
    else()
        # CMake 3.20 Visual Studio/Xcode generators do not understand DEPFILE.
        # Preserve their always-build behavior rather than miss header changes.
        add_custom_target("generate_llvm_runtime_${rtm_arch}"
            COMMAND ${runtime_command}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM)
    endif()
    add_dependencies(${CORE_LIBRARY_NAME} "generate_llvm_runtime_${rtm_arch}")
    install(FILES "${runtime_bitcode}"
            DESTINATION ${INSTALL_LIB_DIR}/runtime
            COMPONENT runtime)

endfunction()

function(COMPILE_CUSTOM_CUDA_LIBRARY cuda_path)
    add_custom_target(
        "generate_cuda_library"
        COMMAND ${CLANG_EXECUTABLE} -c cuda_runtime.cu --cuda-gpu-arch=sm_60 -emit-llvm -std=c++17 --no-cuda-version-check -nocudalib --cuda-path=${cuda_path}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_dependencies(${CORE_LIBRARY_NAME} "generate_cuda_library")
    install(FILES "${CMAKE_SOURCE_DIR}/taichi/runtime/llvm/runtime_module/cuda_runtime-cuda-nvptx64-nvidia-cuda-sm_60.bc"
            DESTINATION ${INSTALL_LIB_DIR}/runtime
            COMPONENT runtime)

endfunction()

# Build llvm-runtime for host arch and cuda (if available)
foreach(arch IN LISTS HOST_ARCH CUDA_ARCH DX12_ARCH AMDGPU_ARCH)
  compile_llvm_runtime(${arch})
endforeach()

# You can implement hand-written CUDA Kernels in cuda_runtime.cu and then
# uncomment the following code to compile it into .bc files

#if(CUDA_ARCH)
#    find_package(CUDAToolkit)
#    if(${CUDAToolkit_FOUND})
#      if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 8)
#          compile_custom_cuda_library(${CUDAToolkit_TARGET_DIR})
#      endif()
#    endif()
#endif()
