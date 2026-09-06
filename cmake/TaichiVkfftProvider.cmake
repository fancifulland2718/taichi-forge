option(TI_BUILD_VKFFT_PROVIDER "Build the optional Vulkan FFT JIT adapter" OFF)
set(TI_VKFFT_ROOT "" CACHE PATH "VkFFT v1.3.4 source root")

add_custom_target(taichi_forge_vkfft_providers)
if(NOT TI_BUILD_VKFFT_PROVIDER)
    return()
endif()

# An explicit build input, not a new runtime compiler downloader or a second
# native runtime profile. Release wheel inclusion is qualified separately.
set(_ti_vkfft_header "${TI_VKFFT_ROOT}/vkFFT/vkFFT.h")
if(NOT EXISTS "${_ti_vkfft_header}")
    message(FATAL_ERROR "Set TI_VKFFT_ROOT to the VkFFT v1.3.4 source tree")
endif()
file(STRINGS "${_ti_vkfft_header}" _ti_vkfft_version REGEX "return 10304;")
if(NOT _ti_vkfft_version)
    message(FATAL_ERROR "The current adapter supports VkFFT v1.3.4")
endif()

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)
find_path(TI_VKFFT_GLSLANG_INCLUDE glslang_c_interface.h
          HINTS "${Vulkan_INCLUDE_DIR}/glslang/Include" REQUIRED)
# Use the compiler distribution's matched static libraries. Do not bind to
# Forge's differently-versioned SPIRV-Tools target or export their symbols.
find_library(TI_VKFFT_GLSLANG_STATIC NAMES glslang
             HINTS "$ENV{VULKAN_SDK}/Lib" REQUIRED)
find_library(TI_VKFFT_SPIRV_OPT_STATIC NAMES SPIRV-Tools-opt
             HINTS "$ENV{VULKAN_SDK}/Lib" REQUIRED)
find_library(TI_VKFFT_SPIRV_STATIC NAMES SPIRV-Tools
             HINTS "$ENV{VULKAN_SDK}/Lib" REQUIRED)
if(UNIX)
    foreach(_ti_vkfft_library TI_VKFFT_GLSLANG_STATIC TI_VKFFT_SPIRV_OPT_STATIC
                             TI_VKFFT_SPIRV_STATIC)
        if(NOT "${${_ti_vkfft_library}}" MATCHES "\\.a$")
            message(FATAL_ERROR "${_ti_vkfft_library} must name a static archive")
        endif()
    endforeach()
endif()
get_filename_component(_ti_vkfft_glslang_include_root
                       "${TI_VKFFT_GLSLANG_INCLUDE}/../.." ABSOLUTE)

add_library(taichi_forge_vkfft_provider SHARED
    "${CMAKE_CURRENT_LIST_DIR}/../taichi/vkfft/provider/provider.cpp")
target_compile_features(taichi_forge_vkfft_provider PRIVATE cxx_std_17)
target_compile_definitions(taichi_forge_vkfft_provider PRIVATE
                          TI_FORGE_VKFFT_PROVIDER_BUILD)
target_include_directories(taichi_forge_vkfft_provider PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/.." "${TI_VKFFT_ROOT}/vkFFT"
    "${TI_VKFFT_GLSLANG_INCLUDE}" "${_ti_vkfft_glslang_include_root}")
target_link_libraries(taichi_forge_vkfft_provider PRIVATE Vulkan::Vulkan
    "${TI_VKFFT_GLSLANG_STATIC}" "${TI_VKFFT_SPIRV_OPT_STATIC}"
    "${TI_VKFFT_SPIRV_STATIC}" Threads::Threads)
set_target_properties(taichi_forge_vkfft_provider PROPERTIES
    OUTPUT_NAME "taichi_forge_vkfft_provider_abi1_vkfft134"
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
    target_link_options(taichi_forge_vkfft_provider PRIVATE "LINKER:--exclude-libs,ALL")
endif()
add_dependencies(taichi_forge_vkfft_providers taichi_forge_vkfft_provider)
