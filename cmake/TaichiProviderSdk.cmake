include_guard(GLOBAL)

# Shared installation contract for independently built provider adapters.
# The component contains only its adapter, common ABI header and license;
# neither vendor libraries nor the Forge runtime core belong to it.
function(ti_install_provider_sdk target provider)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Provider SDK installation requires target ${target}")
    endif()
    set(_ti_provider_component "TaichiSDKProvider_${provider}")
    if(WIN32)
        set(_ti_provider_native_dir bin)
    else()
        set(_ti_provider_native_dir lib)
    endif()
    set(_ti_provider_binary
        "${_ti_provider_native_dir}/hardware_providers/$<TARGET_FILE_NAME:${target}>")
    install(TARGETS ${target}
        RUNTIME DESTINATION bin/hardware_providers
            COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL
        LIBRARY DESTINATION lib/hardware_providers
            COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
    install(FILES "${PROJECT_SOURCE_DIR}/taichi/external_runtime/forge_runtime_provider.h"
        DESTINATION include/taichi
        COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
    install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION licenses
        COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/TaichiProviderManifest.cmake.in"
        "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}.cmake.in" @ONLY)
    file(GENERATE
        OUTPUT "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}_$<CONFIG>.cmake"
        INPUT "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}.cmake.in")
    install(SCRIPT "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}_$<CONFIG>.cmake"
        COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
endfunction()
