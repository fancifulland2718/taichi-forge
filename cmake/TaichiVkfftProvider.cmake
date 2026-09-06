option(TI_BUILD_VKFFT_PROVIDER "Build the optional Vulkan FFT JIT adapter" OFF)
set(TI_VKFFT_ROOT "" CACHE PATH "VkFFT v1.3.4 source root")

add_custom_target(taichi_forge_vkfft_providers)
if(NOT TI_BUILD_VKFFT_PROVIDER)
    return()
endif()
if(APPLE OR (DEFINED TI_WITH_VULKAN AND NOT TI_WITH_VULKAN))
    message(STATUS "The optional VkFFT adapter requires the Vulkan backend")
    return()
endif()

# Fetch only at build time, like the other bundled adapters. Consumers do not
# download a compiler or build C++ at plan creation. An explicit source root
# remains available for offline builds; it is not a runtime/shim commit lock.
if(NOT TI_VKFFT_ROOT)
    include(FetchContent)
    FetchContent_Declare(ti_vkfft_source
        GIT_REPOSITORY https://github.com/DTolm/VkFFT.git
        GIT_TAG 066a17c17068c0f11c9298d848c2976c71fad1c1 # v1.3.4
        GIT_SHALLOW FALSE
        SOURCE_SUBDIR _forge_headers_only)
    FetchContent_MakeAvailable(ti_vkfft_source)
    set(TI_VKFFT_ROOT "${ti_vkfft_source_SOURCE_DIR}")
endif()
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
             HINTS "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/lib" REQUIRED)
find_library(TI_VKFFT_SPIRV_OPT_STATIC NAMES SPIRV-Tools-opt
             HINTS "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/lib" REQUIRED)
find_library(TI_VKFFT_SPIRV_STATIC NAMES SPIRV-Tools
             HINTS "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/lib" REQUIRED)
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

# Plain standalone adapter builds need not define a Python installation root.
if(INSTALL_LIB_DIR)
    install(TARGETS taichi_forge_vkfft_provider
        RUNTIME DESTINATION ${INSTALL_LIB_DIR}/hardware_providers COMPONENT runtime
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}/hardware_providers COMPONENT runtime)
    install(FILES "${TI_VKFFT_ROOT}/LICENSE"
        DESTINATION ${INSTALL_LIB_DIR}/licenses/vkfft
        RENAME VkFFT-LICENSE.txt COMPONENT runtime)
    install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../packaging/licenses/vkfft/"
        DESTINATION ${INSTALL_LIB_DIR}/licenses/vkfft COMPONENT runtime)
endif()
