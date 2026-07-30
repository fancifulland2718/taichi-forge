# ./taichi/rhi/interop/CMakeLists.txt

set(INTEROP_RHI interop_rhi)
ti_add_native_library(${INTEROP_RHI})
target_sources(${INTEROP_RHI}
  PRIVATE
    cuda_external_interop.cpp
    vulkan_cpu_interop.cpp
    vulkan_cuda_interop.cpp
  )

if (TI_WITH_CUDA)
  target_compile_definitions(${INTEROP_RHI} PRIVATE -DTI_WITH_CUDA)
endif()

if (TI_WITH_AMDGPU)
  target_compile_definitions(${INTEROP_RHI} PRIVATE -DTI_WITH_AMDGPU)
endif()

if (TI_WITH_VULKAN)
  target_compile_definitions(${INTEROP_RHI} PRIVATE -DTI_WITH_VULKAN)
endif()

target_include_directories(${INTEROP_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
  )

if (TI_WITH_VULKAN)
  target_link_libraries(${INTEROP_RHI} PRIVATE vulkan_rhi)
endif()
