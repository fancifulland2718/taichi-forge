option(TI_BUILD_BUNDLED_CUDSS_PROVIDER
       "Build the pinned cuDSS provider for the runtime wheel" OFF)
set(TI_CUDSS_ROOT_080 "" CACHE PATH
    "Optional cuDSS 0.8 package root for a bundled-provider build")

add_custom_target(taichi_forge_cudss_providers)

if(NOT TI_BUILD_BUNDLED_CUDSS_PROVIDER)
    return()
endif()

if(APPLE OR NOT (WIN32 OR LINUX))
    message(STATUS "Bundled cuDSS providers are not built on this platform")
    return()
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT TI_CUDSS_ROOT_080 AND DEFINED ENV{TI_CUDSS_ROOT_080})
    set(TI_CUDSS_ROOT_080 "$ENV{TI_CUDSS_ROOT_080}" CACHE PATH
        "Optional cuDSS 0.8 package root for a bundled-provider build" FORCE)
endif()

if(NOT TI_CUDSS_ROOT_080 AND Python_EXECUTABLE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import importlib.util,pathlib; s=importlib.util.find_spec('nvidia'); roots=[] if s is None or s.submodule_search_locations is None else list(s.submodule_search_locations); print(next((str(pathlib.Path(r)/'cu12') for r in roots if (pathlib.Path(r)/'cu12'/'include'/'cudss.h').is_file()),''))"
        RESULT_VARIABLE _ti_cudss_python_result
        OUTPUT_VARIABLE _ti_cudss_python_root
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_ti_cudss_python_result EQUAL 0 AND _ti_cudss_python_root)
        set(TI_CUDSS_ROOT_080 "${_ti_cudss_python_root}" CACHE PATH
            "Optional cuDSS 0.8 package root for a bundled-provider build" FORCE)
    endif()
endif()

set(_ti_cudss_header "${TI_CUDSS_ROOT_080}/include/cudss.h")
if(NOT EXISTS "${_ti_cudss_header}")
    message(FATAL_ERROR
        "Bundled cuDSS adapter requires official cuDSS 0.8 headers. Set "
        "TI_CUDSS_ROOT_080 to the nvidia-cudss-cu12 package root.")
endif()
file(STRINGS "${_ti_cudss_header}" _ti_cudss_major_line
     REGEX "^#define[ \t]+CUDSS_VERSION_MAJOR[ \t]+[0-9]+")
file(STRINGS "${_ti_cudss_header}" _ti_cudss_minor_line
     REGEX "^#define[ \t]+CUDSS_VERSION_MINOR[ \t]+[0-9]+")
if(NOT _ti_cudss_major_line MATCHES "[ \t]0$" OR
   NOT _ti_cudss_minor_line MATCHES "[ \t]8$")
    message(FATAL_ERROR
        "TI_CUDSS_ROOT_080 must contain cuDSS 0.8.x headers")
endif()

add_library(taichi_forge_cudss_provider_080 SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/taichi/cudss/provider/provider.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/taichi/cudss/forge_cudss_provider.h")
target_compile_features(taichi_forge_cudss_provider_080 PRIVATE cxx_std_17)
target_compile_definitions(taichi_forge_cudss_provider_080
    PRIVATE TI_FORGE_CUDSS_PROVIDER_BUILD)
target_include_directories(taichi_forge_cudss_provider_080 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${TI_CUDSS_ROOT_080}/include"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(taichi_forge_cudss_provider_080 PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(taichi_forge_cudss_provider_080 PROPERTIES
    OUTPUT_NAME "taichi_forge_cudss_provider_abi1_cudss080")
add_dependencies(taichi_forge_cudss_providers
                 taichi_forge_cudss_provider_080)

install(TARGETS taichi_forge_cudss_provider_080
        RUNTIME DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                COMPONENT runtime
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                COMPONENT runtime)
