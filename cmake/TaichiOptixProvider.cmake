option(TI_BUILD_OPTIX_PROVIDER
       "Build the user-SDK OptiX provider outside the official wheel" OFF)
set(TI_OPTIX_ROOT "" CACHE PATH
    "User-provided NVIDIA OptiX SDK root (must contain include/optix.h)")

if(NOT TI_BUILD_OPTIX_PROVIDER)
    return()
endif()

if(NOT TI_WITH_CUDA)
    message(FATAL_ERROR "TI_BUILD_OPTIX_PROVIDER requires TI_WITH_CUDA=ON")
endif()
if(NOT TI_OPTIX_ROOT AND DEFINED ENV{OPTIX_ROOT})
    set(TI_OPTIX_ROOT "$ENV{OPTIX_ROOT}")
endif()
if(NOT EXISTS "${TI_OPTIX_ROOT}/include/optix.h" OR
   NOT EXISTS "${TI_OPTIX_ROOT}/include/optix_function_table.h" OR
   NOT EXISTS "${TI_OPTIX_ROOT}/include/optix_stubs.h")
    message(FATAL_ERROR
        "TI_BUILD_OPTIX_PROVIDER requires a user-provided OptiX SDK in "
        "TI_OPTIX_ROOT; Forge does not download or package the SDK")
endif()

file(STRINGS "${TI_OPTIX_ROOT}/include/optix_function_table.h"
     _ti_optix_abi_line REGEX "^#define[ \t]+OPTIX_ABI_VERSION[ \t]+[0-9]+")
string(REGEX MATCH "[0-9]+$" TI_OPTIX_HEADER_ABI "${_ti_optix_abi_line}")
if(NOT TI_OPTIX_HEADER_ABI MATCHES "^(93|105)$")
    message(FATAL_ERROR
        "Unsupported OptiX header ABI ${TI_OPTIX_HEADER_ABI}; "
        "this provider is source-qualified only for ABI 93 (OptiX 8.1) "
        "and ABI 105 (OptiX 9.0)")
endif()

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_NVCC_EXECUTABLE)
    message(FATAL_ERROR "The optional OptiX provider requires nvcc for its PTX artifact")
endif()

set(_ti_optix_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/optix_provider")
set(_ti_optix_ptx "${_ti_optix_generated_dir}/device_program.ptx")
set(_ti_optix_ptx_header "${_ti_optix_generated_dir}/device_program_ptx.h")
add_custom_command(
    OUTPUT "${_ti_optix_ptx}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_ti_optix_generated_dir}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
            --ptx --std=c++17 --use_fast_math
            -I "${TI_OPTIX_ROOT}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/device_program.cu"
            -o "${_ti_optix_ptx}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/device_program.cu"
    VERBATIM)
add_custom_command(
    OUTPUT "${_ti_optix_ptx_header}"
    COMMAND ${CMAKE_COMMAND}
            "-DINPUT_FILE=${_ti_optix_ptx}"
            "-DOUTPUT_FILE=${_ti_optix_ptx_header}"
            "-DSYMBOL_NAME=ti_forge_optix_device_ptx"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake"
    DEPENDS "${_ti_optix_ptx}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake"
    VERBATIM)

add_library(taichi_forge_optix_provider SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/provider.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_provider.h"
    "${_ti_optix_ptx_header}")
target_compile_features(taichi_forge_optix_provider PRIVATE cxx_std_17)
target_compile_definitions(taichi_forge_optix_provider
    PRIVATE TI_FORGE_OPTIX_PROVIDER_BUILD)
target_include_directories(taichi_forge_optix_provider PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${TI_OPTIX_ROOT}/include"
    "${_ti_optix_generated_dir}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(taichi_forge_optix_provider PRIVATE CUDA::cuda_driver)
set_target_properties(taichi_forge_optix_provider PROPERTIES
    OUTPUT_NAME "taichi_forge_optix_provider_abi1")

# Deliberately no install() rule: this target is user-built and never enters
# either official Forge wheel component.
