cmake_minimum_required(VERSION 3.17)

set(TESTS_NAME taichi_cpp_tests)
if(TARGET taichi_runtime AND TI_WITH_SPLIT_PYTHON_RUNTIME)
  add_executable(taichi_completion_tests
    tests/cpp/program/runtime_completion_test.cpp
    tests/cpp/program/runtime_fault_test.cpp)
  target_link_libraries(taichi_completion_tests PRIVATE taichi_runtime gtest_main)
  target_compile_definitions(taichi_completion_tests PRIVATE
    TI_WITH_SPLIT_PYTHON_RUNTIME)
  target_include_directories(taichi_completion_tests PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS})
  if(TI_WITH_GGUI AND TI_WITH_VULKAN)
    target_sources(taichi_completion_tests PRIVATE
      tests/cpp/rhi/common/vulkan_present_policy_test.cpp)
    target_compile_definitions(taichi_completion_tests PRIVATE TI_TEST_DISPLAY_COMPLETION)
    target_include_directories(taichi_completion_tests PRIVATE
      ${PROJECT_SOURCE_DIR}/external/eigen
      ${PROJECT_SOURCE_DIR}/external/FP16/include
      ${PROJECT_SOURCE_DIR}/external/glm
      ${PROJECT_SOURCE_DIR}/external/volk
      ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
      ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include)
  endif()
  add_test(NAME taichi_completion_tests COMMAND taichi_completion_tests)
endif()
if(TI_WITH_GGUI AND NOT ANDROID)
  add_executable(taichi_glfw_input_tests
    tests/cpp/ui/glfw_input_test.cpp
    taichi/ui/common/glfw_input.cpp)
  target_include_directories(taichi_glfw_input_tests PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(taichi_glfw_input_tests PRIVATE gtest_main glfw imgui)
  add_test(NAME taichi_glfw_input_tests COMMAND taichi_glfw_input_tests)
endif()
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
# These Windows/C API fixtures belong to the explicitly configured CiG targets
# below; the standalone probe has its own main(). The generic test target must
# not acquire their headers, exports, global environment or entry point.
list(REMOVE_ITEM TAICHI_TESTS_SOURCE
    "${PROJECT_SOURCE_DIR}/tests/cpp/rhi/common/cuda_cig_test_environment.cpp"
    "${PROJECT_SOURCE_DIR}/tests/cpp/rhi/common/vulkan_cig_fixture.cpp"
    "${PROJECT_SOURCE_DIR}/tests/cpp/rhi/common/native_vulkan_cig_probe.cpp"
    "${PROJECT_SOURCE_DIR}/tests/cpp/rhi/common/vulkan_cig_transfer_probe.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/struct_layout_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/value_parameters_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/argument_origins_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/argpack_storage_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/kernel_result_storage_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/execution_plan_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/native_operation_registry_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/native_array_contract_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/fixed_branch_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/host_branch_contract_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/ordered_storage_ring_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/ordered_storage_plan_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/reduce_core_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/transform_core_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/indexed_copy_core_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/scatter_add_core_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/histogram_core_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/bucket_builder_core_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/diagnostic_reduce_core_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/diagnostic_cuda_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/native_diagnostic_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/native_grouped_reduce_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/native_sort_workspace_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/histogram_cuda_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/grouped_reduce_cuda_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/bucket_builder_cuda_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/control_encoding_cost_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/cuda_predicate_graph_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/cudss_plan_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/cusolverdn_plan_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/cublaslt_plan_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/native_cudss_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/cuda_ordered_submission_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/cuda_binding_frame_replay_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/runtime_client_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/submission_completion_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/completion_wait_set_test.cpp")
list(APPEND TAICHI_TESTS_SOURCE "tests/cpp/aot/optix_provider_test.cpp")

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
  # Includes native primitive preparation and fixed-replay ownership checks.
  file(GLOB TAICHI_TESTS_VULKAN_SOURCE "tests/cpp/aot/vulkan/*.cpp")
  # This dedicated executable replaces operator new for fault injection.
  list(REMOVE_ITEM TAICHI_TESTS_VULKAN_SOURCE
    "${PROJECT_SOURCE_DIR}/tests/cpp/aot/vulkan/submit_exception_safety_test.cpp")
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
if(WIN32 OR LINUX)
    # Exercise the real dynamic-library boundary without a GPU or vendor SDK.
    # Keep the stub out of production objects and install components.
    add_library(taichi_runtime_provider_test_stub SHARED
        tests/native/runtime_provider_stub.cpp)
    target_compile_features(taichi_runtime_provider_test_stub PRIVATE cxx_std_17)
    target_compile_definitions(taichi_runtime_provider_test_stub PRIVATE
        TI_FORGE_RUNTIME_PROVIDER_BUILD)
    target_include_directories(taichi_runtime_provider_test_stub PRIVATE
        ${PROJECT_SOURCE_DIR})
    target_sources(${TESTS_NAME} PRIVATE tests/cpp/aot/native_provider_test.cpp)
    set_property(SOURCE tests/cpp/aot/native_provider_test.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS
        TI_TEST_RUNTIME_PROVIDER_PATH="$<TARGET_FILE:taichi_runtime_provider_test_stub>")
    add_dependencies(${TESTS_NAME} taichi_runtime_provider_test_stub)
    if(TARGET taichi_forge_cutensor_provider)
        foreach(_provider IN ITEMS cutensor cusparselt amgx cublaslt cub_source cutlass_source)
            string(TOUPPER "${_provider}" _provider_upper)
            set_property(SOURCE tests/cpp/aot/native_provider_test.cpp APPEND PROPERTY
                COMPILE_DEFINITIONS
                TI_TEST_${_provider_upper}_PROVIDER_PATH="$<TARGET_FILE:taichi_forge_${_provider}_provider>")
            add_dependencies(${TESTS_NAME} taichi_forge_${_provider}_provider)
        endforeach()
    endif()
endif()
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

# Object-library usage requirements propagate transitively, but their object
# files do not. Split-runtime builds need the native object closure here too;
# linking only taichi_core leaves backend/compiler symbols unresolved.
include(cmake/TaichiObjectDependencies.cmake)
ti_link_object_dependencies(${TESTS_NAME} taichi_core)

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
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers/include
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
  tests/cpp/common/window_layout_test.cpp
  tests/cpp/program/external_dense_storage_test.cpp
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
# Like the aggregate tests, this target uses private native implementation
# symbols. Transitive OBJECT libraries propagate usage requirements, not their
# object files; consume the existing dependency closure in split builds too.
ti_link_object_dependencies(${TAICHI_RUNTIME_FOUNDATION_TESTS_NAME} taichi_core)
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
  add_library(taichi_backend_test_dependencies INTERFACE)
  target_link_libraries(taichi_backend_test_dependencies
    INTERFACE
      taichi_common
      taichi_util
      compilation_manager
      ti_device_api
      cpu_codegen
      cpu_runtime
      cuda_codegen
      cuda_runtime
      llvm_program_impl
      llvm_codegen
      llvm_runtime
      gfx_program_impl
      opengl_program_impl
      vulkan_program_impl
      spirv_codegen
      gfx_runtime
      common_rhi
      interop_rhi
      cpu_rhi
      cuda_rhi
      llvm_rhi
      opengl_rhi
      vulkan_rhi)

  set(TAICHI_BACKEND_SAFETY_TESTS_NAME taichi_backend_safety_tests)
  add_executable(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
    tests/cpp/common/threading_test.cpp
    tests/cpp/rhi/common/allocation_registry_test.cpp
    tests/cpp/rhi/common/cpu_device_test.cpp
    tests/cpp/rhi/common/cuda_context_test.cpp
    tests/cpp/rhi/common/cuda_profiler_test.cpp
    tests/cpp/transforms/make_block_local_test.cpp
    tests/cpp/aot/graph_replay_identity_test.cpp
    tests/cpp/aot/gfx_utils.cpp
    tests/cpp/aot/vulkan/device_test.cpp)
  target_sources(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
    PRIVATE tests/cpp/aot/vulkan/ray_pipeline_test.cpp
            tests/cpp/aot/vulkan/execution_queue_test.cpp
            tests/cpp/aot/vulkan/display_buffer_owner_test.cpp)
  # OBJECT libraries carry usage requirements, but not their object files,
  # through an INTERFACE library. Link the existing native targets directly so
  # split-runtime builds do not require exporting private RHI symbols.
  get_target_property(_ti_backend_native_targets
    taichi_backend_test_dependencies INTERFACE_LINK_LIBRARIES)
  target_link_libraries(${TAICHI_BACKEND_SAFETY_TESTS_NAME}
    PRIVATE
      taichi_core
      gtest_main
      ${_ti_backend_native_targets})
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

  # Real CUDA/Vulkan gates qualify independent device progress without Python
  # or a test spin kernel occupying the GPU. Link private RHI implementation
  # objects rather than expanding the package ABI for test helpers.
  add_executable(taichi_cuda_execution_domain_tests
    tests/cpp/aot/completion_wait_set_test.cpp
    tests/cpp/rhi/common/cuda_execution_domain_test.cpp
    tests/cpp/rhi/common/cuda_retained_memory_test.cpp
    tests/cpp/aot/runtime_client_test.cpp
    tests/cpp/program/external_dense_storage_test.cpp
    tests/cpp/aot/prepared_external_access_test.cpp
    tests/cpp/aot/cuda_ordered_submission_test.cpp
    tests/cpp/aot/cuda_execution_provider_test.cpp
    tests/cpp/aot/llvm/cuda_execution_context_aot_test.cpp
    tests/cpp/aot/llvm/cuda_execution_instance_aot_test.cpp
    tests/cpp/aot/llvm/cuda_gated_aot_test.cpp)
  target_link_libraries(taichi_cuda_execution_domain_tests PRIVATE
    taichi_core gtest_main ${_ti_backend_native_targets})
  if(TARGET taichi_runtime_c_api)
    target_sources(taichi_cuda_execution_domain_tests PRIVATE
      tests/cpp/aot/completion_publication_test.cpp
      tests/cpp/aot/llvm/cuda_execution_owner_test.cpp
      tests/cpp/aot/vulkan/execution_domain_owner_test.cpp
      tests/cpp/aot/vulkan/shared_image_pending_lifetime_test.cpp
      tests/cpp/aot/vulkan/image_import_layout_test.cpp
      tests/cpp/aot/vulkan/shared_image_alias_import_test.cpp
      tests/cpp/aot/argpack_aot_binding_test.cpp)
    target_compile_definitions(taichi_cuda_execution_domain_tests PRIVATE TI_TEST_RUNTIME_C_API)
    target_link_libraries(taichi_cuda_execution_domain_tests PRIVATE taichi_runtime_c_api)
  endif()
  get_target_property(_ti_backend_test_includes
    ${TAICHI_BACKEND_SAFETY_TESTS_NAME} INCLUDE_DIRECTORIES)
  if(WIN32)
    # Allocation failure injection must stay in its own executable: it replaces
    # operator new only on the calling test thread, after real Vulkan enqueue.
    add_executable(taichi_vulkan_submit_safety_tests
      tests/cpp/aot/vulkan/submit_exception_safety_test.cpp)
    target_link_libraries(taichi_vulkan_submit_safety_tests PRIVATE
      taichi_core gtest_main ${_ti_backend_native_targets})
    target_include_directories(taichi_vulkan_submit_safety_tests PRIVATE
      ${_ti_backend_test_includes})
    set_target_properties(taichi_vulkan_submit_safety_tests PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${TESTS_OUTPUT_DIR}")
    add_test(NAME taichi_vulkan_submit_safety_tests
      COMMAND taichi_vulkan_submit_safety_tests)
    set_tests_properties(taichi_vulkan_submit_safety_tests PROPERTIES TIMEOUT 30)
  endif()
  target_include_directories(taichi_cuda_execution_domain_tests PRIVATE
    ${_ti_backend_test_includes})
  set_target_properties(taichi_cuda_execution_domain_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TESTS_OUTPUT_DIR}")
  add_test(NAME taichi_cuda_execution_domain_tests
    COMMAND taichi_cuda_execution_domain_tests)
  set_tests_properties(taichi_cuda_execution_domain_tests PROPERTIES TIMEOUT 30)
  if(WIN32)
    if(TARGET taichi_runtime_c_api)
      target_sources(taichi_cuda_execution_domain_tests PRIVATE
        tests/cpp/rhi/common/cuda_cig_test_environment.cpp)
    endif()
    add_library(taichi_vulkan_cig_fixture SHARED
      tests/cpp/rhi/common/vulkan_cig_fixture.cpp)
    target_include_directories(taichi_vulkan_cig_fixture PRIVATE
      ${PROJECT_SOURCE_DIR}/c_api/include
      ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include)
    set_target_properties(taichi_vulkan_cig_fixture PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${TESTS_OUTPUT_DIR}")
    add_executable(taichi_vulkan_cig_transfer_probe
      tests/cpp/rhi/common/vulkan_cig_transfer_probe.cpp)
    target_include_directories(taichi_vulkan_cig_transfer_probe PRIVATE
      ${PROJECT_SOURCE_DIR}/c_api/include
      ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include)
    set_target_properties(taichi_vulkan_cig_transfer_probe PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${TESTS_OUTPUT_DIR}")
    add_dependencies(taichi_vulkan_cig_transfer_probe taichi_vulkan_cig_fixture)
  endif()
endif()
