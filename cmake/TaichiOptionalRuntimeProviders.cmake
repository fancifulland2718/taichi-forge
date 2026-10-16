include(${CMAKE_CURRENT_LIST_DIR}/TaichiProviderSdk.cmake)

option(TI_BUILD_BUNDLED_OPTIONAL_RUNTIME_PROVIDERS
       "Build optional vendor-runtime execution adapters" OFF)

add_custom_target(taichi_forge_optional_runtime_providers)

if(NOT TI_BUILD_BUNDLED_OPTIONAL_RUNTIME_PROVIDERS)
    return()
endif()

if(APPLE OR NOT (WIN32 OR LINUX))
    message(STATUS "Optional runtime providers are not built on this platform")
    return()
endif()

function(ti_add_optional_runtime_provider provider kind output_name)
    set(target "taichi_forge_${provider}_provider")
    add_library(${target} SHARED
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/external_runtime/provider/provider.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/taichi/external_runtime/forge_runtime_provider.h")
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        TI_FORGE_RUNTIME_PROVIDER_BUILD
        TI_FORGE_RUNTIME_PROVIDER_KIND=${kind})
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${output_name}")
    add_dependencies(taichi_forge_optional_runtime_providers ${target})
    install(TARGETS ${target}
            RUNTIME DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                    COMPONENT runtime
            LIBRARY DESTINATION ${INSTALL_LIB_DIR}/hardware_providers
                    COMPONENT runtime)

    if(provider STREQUAL "cusolverdn" OR provider STREQUAL "cublaslt")
        ti_install_provider_sdk(${target} ${provider}
            PUBLIC_HEADERS
                "${CMAKE_CURRENT_SOURCE_DIR}/taichi/external_runtime/forge_${provider}_provider.h")
    else()
        ti_install_provider_sdk(${target} ${provider})
    endif()
endfunction()

ti_add_optional_runtime_provider(
    cusparselt 1 "taichi_forge_cusparselt_provider_abi2_api080_090")
ti_add_optional_runtime_provider(
    cutensor 2 "taichi_forge_cutensor_provider_abi2_api200_207")
ti_add_optional_runtime_provider(
    amgx 3 "taichi_forge_amgx_provider_abi2_stable_c")
ti_add_optional_runtime_provider(
    cusolverdn 4 "taichi_forge_cusolverdn_provider_abi2_generic_v1")
ti_add_optional_runtime_provider(
    cublaslt 5 "taichi_forge_cublaslt_provider_abi2_configured_v1")
