option(TI_BUILD_BUNDLED_OPTIX_PROVIDERS
       "Build the pinned OptiX provider set for the runtime wheel" OFF)
include(${CMAKE_CURRENT_LIST_DIR}/TaichiProviderSdk.cmake)
option(TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT
       "Allow non-release CUDA Toolkit versions for local adapter builds" OFF)
set(TI_OPTIX_ROOT_105 "" CACHE PATH
    "Optional OptiX ABI 105 header root for a bundled-provider build")
set(TI_OPTIX_ROOT_118 "" CACHE PATH
    "Optional OptiX ABI 118 header root for a bundled-provider build")

# scikit-build-core names this target explicitly on every platform. It remains
# empty where the NVIDIA CUDA/OptiX provider is not supported, so the option
# does not create a new wheel variant or disturb macOS builds.
add_custom_target(taichi_forge_optix_providers)

if(NOT TI_BUILD_BUNDLED_OPTIX_PROVIDERS)
    return()
endif()

if(APPLE OR NOT (WIN32 OR LINUX))
    message(STATUS "Bundled OptiX providers are not built on this platform")
    return()
endif()

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_NVCC_EXECUTABLE)
    message(FATAL_ERROR "The OptiX provider build requires nvcc for its PTX artifact")
endif()
set(_ti_optix_expected_ptx_version "8.7")
if(NOT TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT AND
   (CUDAToolkit_VERSION VERSION_LESS "12.8" OR
    NOT CUDAToolkit_VERSION VERSION_LESS "12.9"))
    message(FATAL_ERROR
        "Bundled OptiX providers require CUDA Toolkit 12.8.x to generate PTX 8.7; found "
        "CUDA ${CUDAToolkit_VERSION}")
endif()
if(TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT)
    set(_ti_optix_expected_ptx_version "")
    message(WARNING
        "Building unqualified OptiX PTX with CUDA ${CUDAToolkit_VERSION}; "
        "this output must not enter a release wheel")
endif()

set(_ti_optix_supported_abis 105 118)

function(_ti_validate_optix_root root expected_abi out_abi)
    if(NOT EXISTS "${root}/include/optix.h" OR
       NOT EXISTS "${root}/include/optix_function_table.h" OR
       NOT EXISTS "${root}/include/optix_stubs.h")
        message(FATAL_ERROR
            "OptiX ABI ${expected_abi} headers were not found under ${root}/include")
    endif()
    file(STRINGS "${root}/include/optix_function_table.h"
         _abi_line REGEX "^#define[ \t]+OPTIX_ABI_VERSION[ \t]+[0-9]+")
    string(REGEX MATCH "[0-9]+$" _header_abi "${_abi_line}")
    list(FIND _ti_optix_supported_abis "${_header_abi}" _supported_index)
    if(_supported_index EQUAL -1)
        message(FATAL_ERROR
            "Unsupported OptiX header ABI ${_header_abi}; Forge supports "
            "ABI 105 (OptiX 9.0) and ABI 118 (OptiX 9.1)")
    endif()
    if(NOT "${expected_abi}" STREQUAL "any" AND
       NOT "${_header_abi}" STREQUAL "${expected_abi}")
        message(FATAL_ERROR
            "Expected OptiX ABI ${expected_abi} headers at ${root}, found ABI ${_header_abi}")
    endif()
    set(${out_abi} "${_header_abi}" PARENT_SCOPE)
endfunction()

# Built-in kernels use only OptiX 9.0 device intrinsics. Generate their PTX once
# with the oldest supported SDK so all host function-table versions can use it.
function(_ti_add_optix_ptx root)
    set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/optix_provider")
    set(_ptx_host_options)
    if(CMAKE_CUDA_HOST_COMPILER)
        list(APPEND _ptx_host_options --compiler-bindir "${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    set(_source "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/device_program.cu")
    foreach(_variant IN ITEMS 0 1 2 3 transform_pack)
        if(_variant STREQUAL "transform_pack")
            set(_name instance_transform_pack)
            set(_define TI_FORGE_OPTIX_TRANSFORM_PACK=1)
            set(_symbol ti_forge_optix_instance_transform_pack_ptx)
        else()
            set(_name device_program_${_variant})
            set(_define TI_FORGE_OPTIX_TYPED=${_variant})
            set(_symbol ti_forge_optix_device_${_variant}_ptx)
        endif()
        set(_ptx "${_generated_dir}/${_name}.ptx")
        set(_header "${_generated_dir}/${_name}_ptx.h")
        add_custom_command(
            OUTPUT "${_ptx}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_generated_dir}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" ${_ptx_host_options}
                    --ptx --std=c++17 --use_fast_math --gpu-architecture=compute_75
                    -D${_define} -I "${root}/include" "${_source}" -o "${_ptx}"
                    -MD -MF "${_ptx}.d"
            DEPENDS "${_source}"
            DEPFILE "${_ptx}.d"
            VERBATIM)
        add_custom_command(
            OUTPUT "${_header}"
            COMMAND ${CMAKE_COMMAND}
                    "-DINPUT_FILE=${_ptx}" "-DOUTPUT_FILE=${_header}"
                    "-DSYMBOL_NAME=${_symbol}"
                    "-DEXPECTED_PTX_VERSION=${_ti_optix_expected_ptx_version}"
                    "-DEXPECTED_PTX_TARGET=sm_75"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake"
            DEPENDS "${_ptx}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake"
            VERBATIM)
        list(APPEND _ptx_headers "${_header}")
    endforeach()
    add_custom_target(taichi_forge_optix_ptx DEPENDS ${_ptx_headers})
endfunction()

function(_ti_add_optix_provider target_name root)
    _ti_validate_optix_root("${root}" any _header_abi)
    add_library(${target_name} SHARED
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/provider.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/program.cpp"
        $<TARGET_OBJECTS:taichi_forge_optix_common>
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_provider.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_runtime_provider.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_query.h")
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_definitions(${target_name}
        PRIVATE TI_FORGE_OPTIX_PROVIDER_BUILD OPTIX_ENABLE_SDK_MIXING)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${root}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/optix_provider"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(${target_name} PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "taichi_forge_optix_provider_abi1_optix${_header_abi}")
    add_dependencies(${target_name} taichi_forge_optix_ptx)
    add_dependencies(taichi_forge_optix_providers ${target_name})

    install(TARGETS ${target_name}
            RUNTIME DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                    COMPONENT runtime
            LIBRARY DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                    COMPONENT runtime)

    set(_ptx_requirement "PTX ${_ti_optix_expected_ptx_version}")
    if(NOT _ti_optix_expected_ptx_version)
        set(_ptx_requirement "generated PTX from unqualified CUDA Toolkit ${CUDAToolkit_VERSION}")
    endif()
    ti_install_provider_sdk(${target_name} optix VARIANT abi${_header_abi}
        PUBLIC_HEADERS
            "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_provider.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_runtime_provider.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/forge_optix_query.h"
        LICENSE_FILES "${root}/LICENSE.txt" "${root}/license_info.txt"
        EXTERNAL_REQUIREMENTS
            "NVIDIA GPU driver with OptiX ABI ${_header_abi} and ${_ptx_requirement} support"
            "driver-provided OptiX runtime and CUDA driver library"
            "platform C/C++ runtime")
endfunction()

include(FetchContent)

function(_ti_resolve_optix_headers abi version commit out_root)
    set(_configured_root "${TI_OPTIX_ROOT_${abi}}")
    if(_configured_root)
        _ti_validate_optix_root("${_configured_root}" "${abi}" _unused_abi)
        set(${out_root} "${_configured_root}" PARENT_SCOPE)
        return()
    endif()

    set(_dependency "ti_optix_${abi}_headers")
    FetchContent_Declare(${_dependency}
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG "${commit}"
        GIT_SHALLOW FALSE
        SOURCE_SUBDIR _forge_headers_only)
    FetchContent_MakeAvailable(${_dependency})
    set(_source_dir "${${_dependency}_SOURCE_DIR}")
    _ti_validate_optix_root("${_source_dir}" "${abi}" _unused_abi)
    message(STATUS
        "Forge bundled OptiX ABI ${abi} uses optix-dev ${version} at ${commit}")
    set(${out_root} "${_source_dir}" PARENT_SCOPE)
endfunction()

# Pin both SDK header sets for reproducible builds of the platform wheel.
_ti_resolve_optix_headers(
    105 9.0.0 fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd _root_105)
_ti_resolve_optix_headers(
    118 9.1.0 f1f6dd803f3159992d248178f6e09421c6eb8b6d _root_118)

_ti_add_optix_ptx("${_root_105}")
# The common Forge session bridge does not include any NVIDIA ABI types.
add_library(taichi_forge_optix_common OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/taichi/optix/provider/runtime_provider.cpp")
target_compile_features(taichi_forge_optix_common PRIVATE cxx_std_17)
target_compile_definitions(taichi_forge_optix_common PRIVATE TI_FORGE_OPTIX_PROVIDER_BUILD)
target_include_directories(taichi_forge_optix_common PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(taichi_forge_optix_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

_ti_add_optix_provider(taichi_forge_optix_provider_105 "${_root_105}")
_ti_add_optix_provider(taichi_forge_optix_provider_118 "${_root_118}")
