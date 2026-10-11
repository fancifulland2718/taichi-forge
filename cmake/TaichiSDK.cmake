# A separate opt-in install component. Ordinary wheel installs do not duplicate
# their C API/runtime payloads, and standalone consumers need no Python package.
set(_ti_sdk_component TaichiSDK)
install(TARGETS ${TAICHI_C_API_NAME}
    RUNTIME DESTINATION bin COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL
    LIBRARY DESTINATION lib COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL
    ARCHIVE DESTINATION lib COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL
    PUBLIC_HEADER DESTINATION include/taichi COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
install(FILES c_api/include/taichi/cpp/taichi.hpp
    DESTINATION include/taichi/cpp COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)

if(TI_WITH_LLVM)
    install(FILES packaging/licenses/llvm/LICENSE.TXT DESTINATION licenses/llvm
        COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
    foreach(_arch IN LISTS HOST_ARCH CUDA_ARCH)
        install(FILES "${PROJECT_SOURCE_DIR}/taichi/runtime/llvm/runtime_module/runtime_${_arch}.bc"
            DESTINATION runtime COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
    endforeach()
    if(TI_WITH_CUDA)
        install(FILES "${PROJECT_SOURCE_DIR}/external/cuda_libdevice/slim_libdevice.10.bc"
            DESTINATION runtime COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
    endif()
endif()

set(TAICHI_SDK_VERSION "${TI_VERSION_MAJOR}.${TI_VERSION_MINOR}.${TI_VERSION_PATCH}")
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE TAICHI_SDK_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE)
foreach(_feature IN ITEMS LLVM CUDA VULKAN)
    if(TI_WITH_${_feature})
        set(TAICHI_SDK_${_feature} true)
    else()
        set(TAICHI_SDK_${_feature} false)
    endif()
endforeach()
configure_file(cmake/TaichiForgeConfig.cmake.in TaichiForgeConfig.cmake @ONLY)
configure_file(cmake/TaichiSDKManifest.cmake.in TaichiSDKManifest.cmake @ONLY)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/TaichiForgeConfigVersion.cmake"
    VERSION "${TAICHI_SDK_VERSION}" COMPATIBILITY SameMinorVersion)
install(FILES "${PROJECT_BINARY_DIR}/TaichiForgeConfig.cmake"
              "${PROJECT_BINARY_DIR}/TaichiForgeConfigVersion.cmake"
    DESTINATION lib/cmake/TaichiForge COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
install(FILES LICENSE DESTINATION licenses COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
foreach(_license IN ITEMS external/SPIRV-Tools/LICENSE external/SPIRV-Headers/LICENSE
        external/SPIRV-Cross/LICENSE external/SPIRV-Reflect/LICENSE
        external/glfw/LICENSE.md external/spdlog/LICENSE external/volk/LICENSE.md
        external/VulkanMemoryAllocator/LICENSE.txt)
    get_filename_component(_dir "${_license}" DIRECTORY)
    get_filename_component(_name "${_dir}" NAME)
    install(FILES "${_license}" DESTINATION "licenses/${_name}"
        COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
endforeach()
install(DIRECTORY examples/aot_sdk/ DESTINATION examples/aot_sdk
    COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL
    PATTERN "__pycache__" EXCLUDE)
install(SCRIPT "${PROJECT_BINARY_DIR}/TaichiSDKManifest.cmake"
    COMPONENT ${_ti_sdk_component} EXCLUDE_FROM_ALL)
