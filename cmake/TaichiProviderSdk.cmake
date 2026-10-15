include_guard(GLOBAL)

# Shared installation contract for independently built provider adapters.
# The component contains its adapter, common ABI header and declared licenses;
# independently loaded vendor libraries and the Forge core do not belong to it.
function(ti_install_provider_sdk target provider)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "" ""
        "LICENSE_FILES;EXTERNAL_REQUIREMENTS;PUBLIC_HEADERS")
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown provider SDK arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
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
    set(_ti_provider_payload "${_ti_provider_binary}"
        "include/taichi/forge_runtime_provider.h" "licenses/LICENSE")
    set(_ti_provider_source "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    install(TARGETS ${target}
        RUNTIME DESTINATION bin/hardware_providers
            COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL
        LIBRARY DESTINATION lib/hardware_providers
            COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
    install(FILES "${_ti_provider_source}/taichi/external_runtime/forge_runtime_provider.h"
        DESTINATION include/taichi
        COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
    install(FILES "${_ti_provider_source}/LICENSE" DESTINATION licenses
        COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
    foreach(_header IN LISTS ARG_PUBLIC_HEADERS)
        get_filename_component(_name "${_header}" NAME)
        set(_relative "include/taichi/${_name}")
        if(_relative IN_LIST _ti_provider_payload)
            message(FATAL_ERROR "Duplicate provider header destination: ${_relative}")
        endif()
        install(FILES "${_header}" DESTINATION include/taichi
            COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
        list(APPEND _ti_provider_payload "${_relative}")
    endforeach()
    foreach(_license IN LISTS ARG_LICENSE_FILES)
        get_filename_component(_name "${_license}" NAME)
        set(_relative "licenses/${provider}/${_name}")
        if(_relative IN_LIST _ti_provider_payload)
            message(FATAL_ERROR "Duplicate provider license destination: ${_relative}")
        endif()
        install(FILES "${_license}" DESTINATION "licenses/${provider}"
            COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
        list(APPEND _ti_provider_payload "${_relative}")
    endforeach()
    if(NOT ARG_EXTERNAL_REQUIREMENTS)
        set(ARG_EXTERNAL_REQUIREMENTS "compatible vendor runtime and its dependencies"
            "GPU driver" "platform C/C++ runtime")
    endif()
    set(_ti_provider_requirements_json "")
    set(_separator "")
    foreach(_requirement IN LISTS ARG_EXTERNAL_REQUIREMENTS)
        string(REPLACE "\\" "\\\\" _requirement "${_requirement}")
        string(REPLACE "\"" "\\\"" _requirement "${_requirement}")
        string(APPEND _ti_provider_requirements_json "${_separator}\"${_requirement}\"")
        set(_separator ",")
    endforeach()
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/TaichiProviderManifest.cmake.in"
        "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}.cmake.in" @ONLY)
    file(GENERATE
        OUTPUT "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}_$<CONFIG>.cmake"
        INPUT "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}.cmake.in")
    install(SCRIPT "${PROJECT_BINARY_DIR}/TaichiProvider_${provider}_$<CONFIG>.cmake"
        COMPONENT ${_ti_provider_component} EXCLUDE_FROM_ALL)
endfunction()
