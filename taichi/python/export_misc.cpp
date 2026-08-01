/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/rhi/metal/metal_api.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/rhi/dx/dx_api.h"
#include "taichi/common/core.h"
#include "taichi/common/interface.h"
#include "taichi/common/task.h"
#include "taichi/math/math.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/program/py_print_buffer.h"
#include "taichi/python/exception.h"
#include "taichi/python/export.h"
#include "taichi/python/memory_usage_monitor.h"
#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/rhi/llvm/device_memory_pool.h"
#include "taichi/system/benchmark.h"
#include "taichi/system/hacked_signal_handler.h"
#include "taichi/system/profiler.h"
#include "taichi/util/offline_cache.h"
#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"
#include "taichi/rhi/cuda/primitives/solver_ptx.h"
#endif

#include "taichi/platform/amdgpu/detect_amdgpu.h"
#if defined(TI_WITH_AMDGPU)
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#endif

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

#ifdef TI_WITH_OPENGL
#include "taichi/rhi/opengl/opengl_api.h"
#endif

#ifdef TI_WITH_DX12
#include "taichi/rhi/dx12/dx12_api.h"
#endif

namespace taichi {

void test_raise_error() {
  raise_assertion_failure_in_python("Just a test.");
}

void print_all_units() {
  std::vector<std::string> names;
  auto interfaces = InterfaceHolder::get_instance()->interfaces;
  for (auto &kv : interfaces) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  int all_units = 0;
  for (auto &interface_name : names) {
    auto impls = interfaces[interface_name]->get_implementation_names();
    std::cout << " * " << interface_name << " [" << int(impls.size()) << "]"
              << std::endl;
    all_units += int(impls.size());
    std::sort(impls.begin(), impls.end());
    for (auto &impl : impls) {
      std::cout << "   + " << impl << std::endl;
    }
  }
  std::cout << all_units << " units in all." << std::endl;
}

void export_misc(py::module &m) {
  py::class_<Config>(m, "Config");  // NOLINT(bugprone-unused-raii)
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const ExceptionForPython &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  py::class_<Task, std::shared_ptr<Task>>(m, "Task")
      .def("initialize", &Task::initialize)
      .def("run",
           static_cast<std::string (Task::*)(const std::vector<std::string> &)>(
               &Task::run));

  py::class_<Benchmark, std::shared_ptr<Benchmark>>(m, "Benchmark")
      .def("run", &Benchmark::run)
      .def("test", &Benchmark::test)
      .def("initialize", &Benchmark::initialize);

#define TI_EXPORT_LOGGING(X)               \
  m.def(#X, [](const std::string &msg) {   \
    taichi::Logger::get_instance().X(msg); \
  });

  m.def("flush_log", []() { taichi::Logger::get_instance().flush(); });

  TI_EXPORT_LOGGING(trace);
  TI_EXPORT_LOGGING(debug);
  TI_EXPORT_LOGGING(info);
  TI_EXPORT_LOGGING(warn);
  TI_EXPORT_LOGGING(error);
  TI_EXPORT_LOGGING(critical);

  m.def("print_all_units", print_all_units);
  m.def("set_core_state_python_imported", CoreState::set_python_imported);
  m.def("set_logging_level", [](const std::string &level) {
    Logger::get_instance().set_level(level);
  });
  m.def("logging_effective", [](const std::string &level) {
    return Logger::get_instance().is_level_effective(level);
  });
  m.def("set_logging_level_default",
        []() { Logger::get_instance().set_level_default(); });
  m.def("set_core_trigger_gdb_when_crash",
        CoreState::set_trigger_gdb_when_crash);
  m.def("test_raise_error", test_raise_error);
  m.def("get_default_float_size", []() { return sizeof(real); });
  m.def("trigger_sig_fpe", []() {
    int a = 2;
    a -= 2;
    return 1 / a;
  });
  m.def("print_profile_info",
        [&]() { Profiling::get_instance().print_profile_info(); });
  m.def("clear_profile_info",
        [&]() { Profiling::get_instance().clear_profile_info(); });
  m.def("export_compile_profile_csv", [](const std::string &path) {
    return Profiling::get_instance().export_csv(path);
  });
  m.def("export_compile_profile_trace", [](const std::string &path) {
    return Profiling::get_instance().export_chrome_trace(path);
  });
  // P-Compile-7: runtime toggle for ti.compile_profile() context manager.
  m.def("set_compile_profile_runtime_enabled",
        &Profiling::set_tracing_runtime_override);
  m.def("clear_compile_profile_runtime_override",
        &Profiling::clear_tracing_runtime_override);
  m.def("is_compile_profile_enabled", &Profiling::is_tracing_enabled);

  // R1.c: read-only memory pool diagnostic snapshots. Adds one extra mutex
  // acquire per call; not on any hot path. See compile_doc/运行时优化规划.md
  // §3.1 for the exposed-fields contract.
  m.def("get_host_memory_pool_stats", []() {
    auto s = taichi::lang::HostMemoryPool::get_instance().get_stats();
    py::dict d;
    d["allocate_count"] = s.allocate_count;
    d["release_count"] = s.release_count;
    d["bytes_allocated_total"] = s.bytes_allocated_total;
    d["bytes_released_total"] = s.bytes_released_total;
    d["raw_chunks"] = s.raw_chunks;
    d["raw_bytes"] = s.raw_bytes;
    d["unified_chunks"] = s.unified_chunks;
    d["requested_live_bytes"] = s.requested_live_bytes;
    d["reserved_bytes"] = s.reserved_bytes;
    d["committed_bytes"] =
        s.committed_bytes_available ? py::cast(s.committed_bytes) : py::none();
    d["capacity_bytes"] = s.capacity_bytes;
    d["used_bytes"] = s.used_bytes;
    d["available_bytes"] = s.available_bytes;
    d["alignment_waste_bytes"] = s.alignment_waste_bytes;
    d["unreclaimed_released_bytes"] = s.unreclaimed_released_bytes;
    d["wasted_bytes"] = s.wasted_bytes;
    d["slab_chunks"] = s.slab_chunks;
    d["large_chunks"] = s.large_chunks;
    d["exclusive_chunks"] = s.exclusive_chunks;
    d["peak_requested_live_bytes"] = s.peak_requested_live_bytes;
    d["peak_reserved_bytes"] = s.peak_reserved_bytes;
    d["peak_used_bytes"] = s.peak_used_bytes;
    d["peak_wasted_bytes"] = s.peak_wasted_bytes;
    d["peak_chunks"] = s.peak_chunks;
    return d;
  });
  m.def("get_device_memory_pool_stats", []() {
    auto s = taichi::lang::DeviceMemoryPool::get_instance().get_stats();
    py::dict d;
    d["allocate_count"] = s.allocate_count;
    d["release_count"] = s.release_count;
    d["bytes_allocated_total"] = s.bytes_allocated_total;
    d["bytes_released_total"] = s.bytes_released_total;
    d["cache_hit_count"] = s.cache_hit_count;
    d["cache_miss_count"] = s.cache_miss_count;
    d["raw_chunks"] = s.raw_chunks;
    d["raw_bytes"] = s.raw_bytes;
    d["cached_blocks"] = s.cached_blocks;
    d["cached_bytes"] = s.cached_bytes;
    return d;
  });
  m.def("start_memory_monitoring", start_memory_monitoring);
  m.def("get_repo_dir", get_repo_dir);
  m.def("get_python_package_dir", get_python_package_dir);
  m.def("set_python_package_dir", set_python_package_dir);
  m.def("cuda_version", get_cuda_version_string);
  m.def("cuda_driver_api_version", []() -> py::object {
#if defined(TI_WITH_CUDA)
    auto &driver = taichi::lang::CUDADriver::get_instance_without_context();
    if (!driver.detected()) {
      return py::none();
    }
    return py::int_(driver.get_version_major() * 1000 +
                    driver.get_version_minor() * 10);
#else
    return py::none();
#endif
  });
  m.def("cuda_conditional_graph_capabilities", []() {
    py::dict result;
#if defined(TI_WITH_CUDA)
    auto &driver = taichi::lang::CUDADriver::get_instance_without_context();
    const bool driver_loaded = driver.detected();
    const int driver_api_version = driver_loaded
                                       ? driver.get_version_major() * 1000 +
                                             driver.get_version_minor() * 10
                                       : 0;
    result["driver_loaded"] = driver_loaded;
    result["driver_api_version"] =
        driver_loaded ? py::cast(driver_api_version) : py::none();
    result["minimum_driver_api_version"] = 12080;
    result["driver_version_eligible"] =
        driver_loaded && driver_api_version >= 12080;
    const bool symbols_loaded =
        driver_loaded && driver.stream_begin_capture_to_graph.available() &&
        driver.stream_end_capture.available() &&
        driver.graph_create.available() &&
        driver.graph_conditional_handle_create.available() &&
        driver.graph_add_node.available() &&
        driver.graph_get_nodes.available() &&
        driver.graph_instantiate_with_flags.available() &&
        driver.graph_launch.available() && driver.graph_destroy.available() &&
        driver.graph_exec_destroy.available();
    const bool setter_compiled =
        taichi::lang::cuda::driver_cg_conditional_setter_compiled();
    const bool graph_setter_compiled =
        taichi::lang::cuda::driver_graph_conditional_setter_compiled();
    const bool ordinary_graph_symbols_loaded =
        driver_loaded && driver.stream_begin_capture.available() &&
        driver.stream_end_capture.available() &&
        driver.graph_instantiate_with_flags.available() &&
        driver.graph_launch.available() && driver.graph_destroy.available() &&
        driver.graph_exec_destroy.available();
    const bool masked_latch_compiled =
        taichi::lang::cuda::driver_graph_mask_latch_compiled();
    auto &cublas = taichi::lang::CUBLASDriver::get_instance();
    const bool cublas_loaded =
        cublas.is_loaded() ? true : cublas.load_cublas();
    const bool cublas_workspace_symbol_loaded =
        cublas_loaded && cublas.cubSetWorkspace.available();
    result["conditional_graph_symbols_loaded"] = symbols_loaded;
    result["device_setter_lowering_compiled"] = setter_compiled;
    result["general_device_setter_lowering_compiled"] =
        graph_setter_compiled;
    result["ordinary_graph_symbols_loaded"] = ordinary_graph_symbols_loaded;
    result["internal_masked_latch_compiled"] = masked_latch_compiled;
    result["runtime_path_compiled"] = true;
    result["cublas_workspace_symbol_loaded"] =
        cublas_workspace_symbol_loaded;
    const bool base_available = driver_loaded &&
                                driver_api_version >= 12080 && symbols_loaded;
    result["stored_solver_device_control_available"] =
        base_available && setter_compiled && cublas_workspace_symbol_loaded;
    const bool exact_graph_control_available =
        base_available && graph_setter_compiled;
    const bool internal_masked_graph_available =
        ordinary_graph_symbols_loaded && masked_latch_compiled;
    result["general_graph_exact_control_available"] =
        exact_graph_control_available;
    result["internal_masked_graph_available"] = internal_masked_graph_available;
    result["general_graph_device_control_available"] =
        exact_graph_control_available || internal_masked_graph_available;
    // Kept as the stored-solver compatibility aggregate. Generic Graph
    // consumers must use general_graph_device_control_available instead.
    result["fully_available"] =
        result["stored_solver_device_control_available"];
#else
    result["driver_loaded"] = false;
    result["driver_api_version"] = py::none();
    result["minimum_driver_api_version"] = 12080;
    result["driver_version_eligible"] = false;
    result["conditional_graph_symbols_loaded"] = false;
    result["device_setter_lowering_compiled"] = false;
    result["general_device_setter_lowering_compiled"] = false;
    result["ordinary_graph_symbols_loaded"] = false;
    result["internal_masked_latch_compiled"] = false;
    result["runtime_path_compiled"] = false;
    result["cublas_workspace_symbol_loaded"] = false;
    result["stored_solver_device_control_available"] = false;
    result["general_graph_device_control_available"] = false;
    result["general_graph_exact_control_available"] = false;
    result["internal_masked_graph_available"] = false;
    result["fully_available"] = false;
#endif
    return result;
  });
  m.def("test_cpp_exception", [] {
    try {
      throw std::exception();
    } catch (const std::exception &e) {
      printf("caught.\n");
    }
    printf("test was successful.\n");
  });
  m.def("pop_python_print_buffer", []() { return py_cout.pop_content(); });
  m.def("toggle_python_print_buffer", [](bool opt) { py_cout.enabled = opt; });
  m.def("with_cuda", is_cuda_api_available);
  m.def("with_amdgpu", is_rocm_api_available);
#ifdef TI_WITH_METAL
  m.def("with_metal", taichi::lang::metal::is_metal_api_available);
#else
  m.def("with_metal", []() { return false; });
#endif
#ifdef TI_WITH_OPENGL
  m.def("with_opengl", taichi::lang::opengl::is_opengl_api_available,
        py::arg("use_gles") = false);
#else
  m.def("with_opengl", [](bool use_gles) { return false; });
#endif
#ifdef TI_WITH_VULKAN
  m.def("with_vulkan", taichi::lang::vulkan::is_vulkan_api_available);
  m.def("set_vulkan_visible_device",
        taichi::lang::vulkan::set_vulkan_visible_device);
#else
  m.def("with_vulkan", []() { return false; });
#endif
#ifdef TI_WITH_DX11
  m.def("with_dx11", taichi::lang::directx11::is_dx_api_available);
#else
  m.def("with_dx11", []() { return false; });
#endif
#ifdef TI_WITH_DX12
  m.def("with_dx12", taichi::lang::directx12::is_dx12_api_available);
#else
  m.def("with_dx12", []() { return false; });
#endif

  m.def("clean_offline_cache_files",
        lang::offline_cache::clean_offline_cache_files);

  py::class_<HackedSignalRegister>(m, "HackedSignalRegister").def(py::init<>());
}

}  // namespace taichi
