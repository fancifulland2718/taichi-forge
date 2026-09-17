# ./taichi/rhi/amdgpu/CMakeLists.txt

set(AMDGPU_RHI amdgpu_rhi)
# Only headers are needed at build time. HIP remains an optional dynamic runtime.
include(${PROJECT_SOURCE_DIR}/cmake/AMDGPUHeaders.cmake)
ti_add_native_library(${AMDGPU_RHI})
target_compile_definitions(${AMDGPU_RHI} PRIVATE __HIP_PLATFORM_AMD__=1)
target_include_directories(${AMDGPU_RHI} SYSTEM PRIVATE ${TI_HIP_INCLUDE_DIR})
target_sources(${AMDGPU_RHI}
  PRIVATE
    amdgpu_device.cpp
    amdgpu_context.cpp
    amdgpu_driver.cpp
    amdgpu_profiler.cpp
  )

target_include_directories(${AMDGPU_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )
