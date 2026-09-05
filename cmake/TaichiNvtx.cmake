# Header-only annotations: no CUDA Toolkit or nvToolsExt runtime dependency.
# A fixed upstream source is independent of the Forge runtime/shim ABI contract.
include(FetchContent)
FetchContent_Declare(ti_nvtx_headers
    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
    GIT_TAG 3ebbc93ded7285963bff932c678fa367eb393ba6 # v3.3.0
    GIT_SHALLOW FALSE
    SOURCE_SUBDIR _forge_headers_only)
FetchContent_MakeAvailable(ti_nvtx_headers)

# Only the annotation translation unit consumes NVTX's platform headers.
set_property(SOURCE taichi/system/profiler_annotation.cpp APPEND PROPERTY
    INCLUDE_DIRECTORIES "${ti_nvtx_headers_SOURCE_DIR}/c/include")
target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE ${CMAKE_DL_LIBS})
install(FILES "${ti_nvtx_headers_SOURCE_DIR}/LICENSE.txt"
    DESTINATION ${INSTALL_LIB_DIR}/licenses
    RENAME nvtx3-LICENSE.txt COMPONENT runtime)
