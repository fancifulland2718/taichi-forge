cmake_minimum_required(VERSION 3.17)

set(TESTS_NAME taichi_cpp_tests)
if (WIN32)
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# TODO(#2195):
# 1. "cpp" -> "cpp_legacy", "cpp_new" -> "cpp"
# 2. Re-implement the legacy CPP tests using googletest
file(GLOB_RECURSE TAICHI_TESTS_SOURCE
        "tests/cpp/analysis/*.cpp"
        "tests/cpp/backends/*.cpp"
        "tests/cpp/codegen/*.cpp"
        "tests/cpp/common/*.cpp"
        "tests/cpp/ir/*.cpp"
        "tests/cpp/program/*.cpp"
        "tests/cpp/rhi/common/*.cpp"
        "tests/cpp/struct/*.cpp"
        "tests/cpp/transforms/*.cpp"
        "tests/cpp/offline_cache/*.cpp")

if (TI_WITH_OPENGL OR TI_WITH_VULKAN)
    file(GLOB TAICHI_TESTS_GFX_UTILS_SOURCE
        "tests/cpp/aot/gfx_utils.cpp")
    list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_GFX_UTILS_SOURCE})
endif()

if(TI_WITH_LLVM)
  file(GLOB TAICHI_TESTS_LLVM_SOURCE "tests/cpp/aot/llvm/*.cpp" "tests/cpp/llvm/*.cpp")
  list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_LLVM_SOURCE})
endif()

if(TI_WITH_VULKAN)
  file(GLOB TAICHI_TESTS_VULKAN_SOURCE "tests/cpp/aot/vulkan/*.cpp")
  list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_VULKAN_SOURCE})
endif()

if(TI_WITH_OPENGL)
  file(GLOB TAICHI_TESTS_OPENGL_SOURCE "tests/cpp/aot/opengl/*.cpp")
  list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_OPENGL_SOURCE})
endif()

if(TI_WITH_DX12)
  file(GLOB TAICHI_TESTS_DX12_SOURCE "tests/cpp/aot/dx12/*.cpp")
  list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_DX12_SOURCE})
endif()

add_executable(${TESTS_NAME} ${TAICHI_TESTS_SOURCE})
if (WIN32)
    # Output the executable to build/ instead of build/Debug/...
    set(TESTS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${TESTS_OUTPUT_DIR})
    if (MSVC AND TI_GENERATE_PDB)
        target_compile_options(${TESTS_NAME} PRIVATE "/Zi")
        target_link_options(${TESTS_NAME} PRIVATE "/DEBUG")
        target_link_options(${TESTS_NAME} PRIVATE "/OPT:REF")
        target_link_options(${TESTS_NAME} PRIVATE "/OPT:ICF")
    endif()
endif()
target_link_libraries(${TESTS_NAME} PRIVATE taichi_core)
target_link_libraries(${TESTS_NAME} PRIVATE gtest_main)
target_link_libraries(${TESTS_NAME} PRIVATE taichi_common)

if (TI_WITH_BACKTRACE)
    target_link_libraries(${TESTS_NAME} PRIVATE ${BACKWARD_ENABLE})
endif()

if (TI_WITH_OPENGL OR TI_WITH_VULKAN)
  target_link_libraries(${TESTS_NAME} PRIVATE gfx_runtime)
endif()

if (TI_WITH_VULKAN)
  target_link_libraries(${TESTS_NAME} PRIVATE vulkan_rhi)
endif()

if (TI_WITH_OPENGL)
  target_link_libraries(${TESTS_NAME} PRIVATE opengl_rhi)
endif()

if (TI_WITH_DX12)
  target_link_libraries(${TESTS_NAME} PRIVATE dx12_runtime)
  target_link_libraries(${TESTS_NAME} PRIVATE dx12_rhi)
endif()

target_include_directories(${TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
  )

target_include_directories(${TESTS_NAME} SYSTEM
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
  )

if (NOT ANDROID)
  target_include_directories(${TESTS_NAME}
  PRIVATE
    external/glfw/include
  )
endif ()

if(LINUX)
    target_link_options(${TESTS_NAME} PUBLIC -Wl,--exclude-libs=ALL)
    target_link_options(${TESTS_NAME} PUBLIC -static-libgcc -static-libstdc++)
endif()
add_test(NAME ${TESTS_NAME} COMMAND ${TESTS_NAME})

# Keep the focused runtime state-machine and CPU provider contract tests
# independently runnable. The aggregate executable also compiles
# backend/compiler mocks, so an unrelated mock API change must not hide
# lifecycle regressions in this layer.
set(TAICHI_RUNTIME_FOUNDATION_TESTS_NAME taichi_runtime_foundation_tests)
add_executable(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
  tests/cpp/program/linear_operator_test.cpp
  tests/cpp/program/primitive_workspace_test.cpp
  tests/cpp/program/runtime_completion_test.cpp
  tests/cpp/program/runtime_fault_test.cpp
  tests/cpp/program/runtime_statistics_test.cpp
  tests/cpp/program/runtime_trace_test.cpp
  tests/cpp/program/sparse_numeric_transaction_test.cpp
  tests/cpp/program/storage_view_test.cpp
  tests/cpp/rhi/common/host_memory_pool_test.cpp)
target_link_libraries(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
  PRIVATE
    taichi_core
    taichi_common
    gtest_main)
if (TI_WITH_OPENGL OR TI_WITH_VULKAN)
  target_link_libraries(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
    PRIVATE gfx_runtime)
endif()
if (TI_WITH_VULKAN)
  target_link_libraries(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
    PRIVATE vulkan_rhi)
endif()
if (TI_WITH_OPENGL)
  target_link_libraries(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
    PRIVATE opengl_rhi)
endif()
if (TI_WITH_DX12)
  target_link_libraries(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
    PRIVATE dx12_runtime dx12_rhi)
endif()
target_include_directories(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include)
target_include_directories(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
  SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include)
if (NOT ANDROID)
  target_include_directories(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
    PRIVATE external/glfw/include)
endif()
if (WIN32)
  set_target_properties(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TESTS_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TESTS_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TESTS_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${TESTS_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${TESTS_OUTPUT_DIR}")
endif()
if (LINUX)
  target_link_options(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME} PUBLIC
    -Wl,--exclude-libs=ALL -static-libgcc -static-libstdc++)
endif()
add_test(NAME ${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME}
  COMMAND ${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME})
if (TI_WITH_LLVM)
  set_tests_properties(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME} PROPERTIES
    ENVIRONMENT
      "TI_LIB_DIR=${PROJECT_SOURCE_DIR}/taichi/runtime/llvm/runtime_module")
endif()

# F5 measurement-only host allocator benchmark. Keep it independent from the
# aggregate C++ tests so unrelated compiler/backend mocks cannot block the
# allocator performance gate.
add_executable(taichi_host_allocator_bench
  benchmarks/host_allocator_bench.cpp)
target_link_libraries(taichi_host_allocator_bench
  PRIVATE
    common_rhi
    ti_device_api
    taichi_core
    taichi_common)
target_include_directories(taichi_host_allocator_bench
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include)
if (WIN32)
  set_target_properties(taichi_host_allocator_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TESTS_OUTPUT_DIR}")
endif()

# Keep the concurrency-sensitive backend regression cases independently
# runnable.  The main C++ test executable also includes LLVM-only tests, which
# may be unavailable in a lightweight backend runtime build.
if (TI_WITH_VULKAN AND TI_WITH_CUDA)
  set(TAICHI_BACKEND_SAFETY_TESTS_NAME taichi_backend_safety_tests)
  add_executable(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
    tests/cpp/common/threading_test.cpp
    tests/cpp/rhi/common/allocation_registry_test.cpp
    tests/cpp/rhi/common/cpu_device_test.cpp
    tests/cpp/rhi/common/cuda_context_test.cpp
    tests/cpp/rhi/common/cuda_profiler_test.cpp
    tests/cpp/aot/graph_replay_identity_test.cpp
    tests/cpp/aot/gfx_utils.cpp
    tests/cpp/aot/vulkan/device_test.cpp)
  target_link_libraries(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
    PRIVATE
      taichi_core
      taichi_common
      gtest_main
      gfx_runtime
      cuda_rhi
      vulkan_rhi)
  target_include_directories(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
    PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/external/spdlog/include
      ${PROJECT_SOURCE_DIR}/external/include
      ${PROJECT_SOURCE_DIR}/external/eigen
      ${PROJECT_SOURCE_DIR}/external/volk
      ${PROJECT_SOURCE_DIR}/external/glad/include
      ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
      ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers/include
      ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include)
  target_include_directories(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
    SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include)
  if (NOT ANDROID)
    target_include_directories(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
      PRIVATE external/glfw/include)
  endif()
  if (WIN32)
    set_target_properties(${TAICHI_BACKEND_SAFETY_TESTS_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${TESTS_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TESTS_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TESTS_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${TESTS_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${TESTS_OUTPUT_DIR}")
  endif()
  add_test(NAME ${TAICHI_BACKEND_SAFETY_TESTS_NAME}
           COMMAND ${TAICHI_BACKEND_SAFETY_TESTS_NAME})
endif()
