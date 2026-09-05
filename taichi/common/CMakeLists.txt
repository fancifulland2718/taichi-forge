# ./taichi/common/CMakeLists.txt

ti_add_native_library(taichi_common)
# These facts must come from the native library, not the independently built
# Python shim's CMake options. No Toolkit headers or linkage are added here.
if (TI_WITH_CUDA AND TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE)
  target_compile_definitions(taichi_common PRIVATE TI_FORGE_BUILD_TOOLKIT_REFERENCE)
endif()
if (TI_WITH_CUDA AND TI_WITH_CUPTI)
  target_compile_definitions(taichi_common PRIVATE TI_FORGE_BUILD_CUPTI)
endif()
target_sources(taichi_common
  PRIVATE
    cleanup.cpp
    core.cpp
    json.cpp
    logging.cpp
    symbol_version.cpp
    virtual_dir.cpp
    zip.cpp
    miniz.c
    dynamic_loader.cpp
  )

target_include_directories(taichi_common
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
