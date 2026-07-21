// Bindings for the python frontend

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include "taichi/ir/snode.h"

#if TI_WITH_LLVM
#include "llvm/Config/llvm-config.h"
#endif

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "fp16.h"

#include "taichi/ir/expression_ops.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/graph_builder.h"
#include "taichi/program/extension.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/matrix.h"
#include "taichi/python/export.h"
#include "taichi/math/svd.h"
#include "taichi/system/timeline.h"
#include "taichi/codegen/spirv/spv_stats.h"
#include "taichi/python/snode_registry.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/program/sparse_preconditioner.h"
#include "taichi/program/sparse_solver.h"
#include "taichi/program/conjugate_gradient.h"
#include "taichi/program/sparse_bicgstab.h"
#include "taichi/program/sparse_fixed_bicgstab.h"
#include "taichi/program/sparse_minres.h"
#include "taichi/aot/graph_data.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/ir/mesh.h"

#include "taichi/program/kernel_profiler.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

namespace taichi {
bool test_threading();

}  // namespace taichi

namespace taichi::lang {

std::string libdevice_path();

namespace irpass {
void get_fs_inner_stats(uint64_t *entries,
                        uint64_t *noop_returns,
                        uint64_t *total_iterations);
void reset_fs_inner_stats();
}  // namespace irpass

}  // namespace taichi::lang

namespace taichi {
namespace {

// Record native primitive telemetry inside the existing pybind call. Keeping
// this at the binding boundary covers cold calls, cached plan descriptors and
// direct advanced usage without adding a second Python-to-C++ round trip.
template <typename Result, typename... Args>
auto tracked_native_program_method(Result (lang::Program::*method)(Args...)) {
  return [method](lang::Program *program, Args... args) -> Result {
    try {
      if constexpr (std::is_void_v<Result>) {
        (program->*method)(std::forward<Args>(args)...);
        program->record_runtime_submission_stat(
            lang::RuntimeSubmissionKind::kNative);
      } else {
        Result result = (program->*method)(std::forward<Args>(args)...);
        program->record_runtime_submission_stat(
            lang::RuntimeSubmissionKind::kNative);
        return result;
      }
    } catch (...) {
      program->record_runtime_submission_failure();
      throw;
    }
  };
}

std::size_t checked_buffer_nbytes(const py::buffer_info &info,
                                  const char *invalid_message,
                                  const char *oversized_message) {
  TI_ERROR_IF(info.ndim < 0 || info.itemsize < 0 || info.size < 0, "{}",
              invalid_message);
  const auto size = static_cast<std::size_t>(info.size);
  const auto itemsize = static_cast<std::size_t>(info.itemsize);
  TI_ERROR_IF(itemsize != 0 &&
                  size > std::numeric_limits<std::size_t>::max() / itemsize,
              "{}", oversized_message);
  return size * itemsize;
}

py::dict runtime_trace_snapshot_to_dict(
    const lang::RuntimeTraceSnapshot &snapshot) {
  py::dict result;
  result["program_domain"] = snapshot.program_domain;
  result["session"] = snapshot.session;
  result["enabled"] = snapshot.enabled;
  result["max_threads"] = snapshot.max_threads;
  result["events_per_thread"] = snapshot.events_per_thread;
  result["event_capacity"] = snapshot.event_capacity;
  result["allocated_bytes"] = snapshot.allocated_bytes;
  result["recorded_events"] = snapshot.recorded_events;
  result["dropped_events"] = snapshot.dropped_events;
  return result;
}

py::dict primitive_workspace_snapshot_to_dict(
    const lang::PrimitiveWorkspaceSnapshot &snapshot) {
  py::dict result;
  result["budget_bytes"] = snapshot.budget_bytes;
  result["reserved_bytes"] = snapshot.reserved_bytes;
  result["in_use_bytes"] = snapshot.in_use_bytes;
  result["persistent_bytes"] = snapshot.persistent_bytes;
  result["reclaimable_bytes"] = snapshot.reclaimable_bytes;
  result["over_budget_bytes"] = snapshot.over_budget_bytes;
  result["peak_reserved_bytes"] = snapshot.peak_reserved_bytes;
  result["peak_in_use_bytes"] = snapshot.peak_in_use_bytes;
  result["entries"] = snapshot.entries;
  result["active_leases"] = snapshot.active_leases;
  result["acquisitions"] = snapshot.acquisitions;
  result["cache_hits"] = snapshot.cache_hits;
  result["cache_misses"] = snapshot.cache_misses;
  result["growth_events"] = snapshot.growth_events;
  result["clear_calls"] = snapshot.clear_calls;
  result["cleared_entries"] = snapshot.cleared_entries;
  result["trim_calls"] = snapshot.trim_calls;
  result["evictions"] = snapshot.evictions;
  result["lock_samples"] = snapshot.lock_samples;
  result["lock_contentions"] = snapshot.lock_contentions;
  result["lock_wait_ns"] = snapshot.lock_wait_ns;
  return result;
}

}  // namespace

void export_lang(py::module &m) {
  using namespace taichi::lang;
  using namespace std::placeholders;

  py::register_exception<TaichiTypeError>(m, "TaichiTypeError",
                                          PyExc_TypeError);
  py::register_exception<TaichiSyntaxError>(m, "TaichiSyntaxError",
                                            PyExc_SyntaxError);
  py::register_exception<TaichiIndexError>(m, "TaichiIndexError",
                                           PyExc_IndexError);
  py::register_exception<TaichiRuntimeError>(m, "TaichiRuntimeError",
                                             PyExc_RuntimeError);
  py::register_exception<TaichiAssertionError>(m, "TaichiAssertionError",
                                               PyExc_AssertionError);
  py::enum_<Arch>(m, "Arch", py::arithmetic())
#define PER_ARCH(x) .value(#x, Arch::x)
#include "taichi/inc/archs.inc.h"
#undef PER_ARCH
      .export_values();

  m.def("arch_name", arch_name);
  m.def("arch_from_name", arch_from_name);

  py::enum_<SNodeType>(m, "SNodeType", py::arithmetic())
#define PER_SNODE(x) .value(#x, SNodeType::x)
#include "taichi/inc/snodes.inc.h"
#undef PER_SNODE
      .export_values();

  py::enum_<Extension>(m, "Extension", py::arithmetic())
#define PER_EXTENSION(x) .value(#x, Extension::x)
#include "taichi/inc/extensions.inc.h"
#undef PER_EXTENSION
      .export_values();

  py::enum_<ExternalArrayLayout>(m, "Layout", py::arithmetic())
      .value("AOS", ExternalArrayLayout::kAOS)
      .value("SOA", ExternalArrayLayout::kSOA)
      .value("NULL", ExternalArrayLayout::kNull)
      .export_values();

  py::enum_<AutodiffMode>(m, "AutodiffMode", py::arithmetic())
      .value("NONE", AutodiffMode::kNone)
      .value("VALIDATION", AutodiffMode::kCheckAutodiffValid)
      .value("FORWARD", AutodiffMode::kForward)
      .value("REVERSE", AutodiffMode::kReverse)
      .export_values();

  py::enum_<SNodeGradType>(m, "SNodeGradType", py::arithmetic())
      .value("PRIMAL", SNodeGradType::kPrimal)
      .value("ADJOINT", SNodeGradType::kAdjoint)
      .value("DUAL", SNodeGradType::kDual)
      .value("ADJOINT_CHECKBIT", SNodeGradType::kAdjointCheckbit)
      .export_values();

  py::enum_<BoundaryMode>(m, "BoundaryMode", py::arithmetic())
      .value("UNSAFE", BoundaryMode::kUnsafe)
      .value("CLAMP", BoundaryMode::kClamp)
      .export_values();

  // TODO(type): This should be removed
  py::class_<DataType>(m, "DataType")
      .def(py::init<Type *>())
      .def(py::self == py::self)
      .def("__hash__", &DataType::hash)
      .def("to_string", &DataType::to_string)
      .def("__str__", &DataType::to_string)
      .def("shape", &DataType::get_shape)
      .def("element_type", &DataType::get_element_type)
      .def("ptr_removed", &DataType::ptr_removed)
      .def(
          "get_ptr", [](DataType *dtype) -> Type * { return *dtype; },
          py::return_value_policy::reference)
      .def("__call__",
           [](DataType *dtype, py::args args, const py::kwargs &kwargs) {
             // Defining __call__ here to make DataType callable in Python,
             // which enables us to write `typing.Tuple[ti.i32, ti.i32]`.
             throw TaichiSyntaxError(
                 "Taichi data types cannot be called outside Taichi kernels.");
           })
      .def(py::pickle(
          [](const DataType &dt) {
            // Note: this only works for primitive types, which is fine for now.
            auto primitive =
                dynamic_cast<const PrimitiveType *>((const Type *)dt);
            TI_ASSERT(primitive);
            return py::make_tuple((std::size_t)primitive->type);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid state!");

            DataType dt =
                PrimitiveType::get((PrimitiveTypeID)(t[0].cast<std::size_t>()));

            return dt;
          }));

  py::class_<DebugInfo>(m, "DebugInfo")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def(py::init<>())
      .def_readwrite("tb", &DebugInfo::tb)
      .def_readwrite("src_loc", &DebugInfo::src_loc);

  py::class_<CompileConfig>(m, "CompileConfig")
      .def(py::init<>())
      .def_readwrite("arch", &CompileConfig::arch)
      .def_readwrite("opt_level", &CompileConfig::opt_level)
      .def_readwrite("llvm_opt_level", &CompileConfig::llvm_opt_level)
      .def_readwrite("compile_tier", &CompileConfig::compile_tier)
      .def_readwrite("spirv_parallel_codegen",
                     &CompileConfig::spirv_parallel_codegen)
      .def_readwrite("spirv_skip_loop_unroll",
                     &CompileConfig::spirv_skip_loop_unroll)
      .def_readwrite("vulkan_sparse_experimental",
                     &CompileConfig::vulkan_sparse_experimental)
      .def_readwrite("vulkan_quant_experimental",
                     &CompileConfig::vulkan_quant_experimental)
      .def_readwrite("hash_snode_experimental",
                     &CompileConfig::hash_snode_experimental)
      .def_readwrite("hash_snode_default_load_factor",
                     &CompileConfig::hash_snode_default_load_factor)
      .def_readwrite("hash_snode_active_list",
                     &CompileConfig::hash_snode_active_list)
      .def_readwrite("hash_snode_diagnostics",
                     &CompileConfig::hash_snode_diagnostics)
      .def_readwrite("hash_snode_compact_child_pool",
                     &CompileConfig::hash_snode_compact_child_pool)
      .def_readwrite("spirv_disabled_passes",
                     &CompileConfig::spirv_disabled_passes)
      .def_readwrite("spirv_adaptive_opt",
            &CompileConfig::spirv_adaptive_opt)
      .def_readwrite("spirv_adaptive_opt_threshold",
            &CompileConfig::spirv_adaptive_opt_threshold)
      .def_readwrite("compile_dag_scheduler",
                     &CompileConfig::compile_dag_scheduler)
      .def_readwrite("vulkan_pointer_freelist",
                     &CompileConfig::vulkan_pointer_freelist)
      .def_readwrite("vulkan_pointer_ambient_zone",
                     &CompileConfig::vulkan_pointer_ambient_zone)
      .def_readwrite("vulkan_pointer_cas_marker",
                     &CompileConfig::vulkan_pointer_cas_marker)
      .def_readwrite("vulkan_pointer_pool_fraction",
                     &CompileConfig::vulkan_pointer_pool_fraction)
      .def_readwrite("vulkan_pointer_independent_pool",
                     &CompileConfig::vulkan_pointer_independent_pool)
      .def_readwrite("vulkan_pointer_allocator_kind",
                     &CompileConfig::vulkan_pointer_allocator_kind)
      .def_readwrite("vulkan_pointer_max_chunks",
                     &CompileConfig::vulkan_pointer_max_chunks)
      .def_readwrite("vulkan_pointer_deterministic_slot",
                     &CompileConfig::vulkan_pointer_deterministic_slot)
      .def_readwrite("cuda_pointer_deterministic_slot",
                     &CompileConfig::cuda_pointer_deterministic_slot)
      .def_readwrite("cuda_pointer_fast_reset",
                     &CompileConfig::cuda_pointer_fast_reset)
      .def_readwrite("cuda_listgen_reuse",
            &CompileConfig::cuda_listgen_reuse)
      .def_readwrite("cuda_listgen_reuse_adaptive",
             &CompileConfig::cuda_listgen_reuse_adaptive)
      .def_readwrite("bitmasked_clear_data_on_deactivate",
                     &CompileConfig::bitmasked_clear_data_on_deactivate)
      .def_readwrite("print_ir", &CompileConfig::print_ir)
      .def_readwrite("print_preprocessed_ir",
                     &CompileConfig::print_preprocessed_ir)
      .def_readwrite("print_ir_dbg_info", &CompileConfig::print_ir_dbg_info)
      .def_readwrite("debug", &CompileConfig::debug)
      .def_readwrite("cfg_optimization", &CompileConfig::cfg_optimization)
      .def_readwrite("check_out_of_bound", &CompileConfig::check_out_of_bound)
      .def_readwrite("print_accessor_ir", &CompileConfig::print_accessor_ir)
      .def_readwrite("use_llvm", &CompileConfig::use_llvm)
      .def_readwrite("print_struct_llvm_ir",
                     &CompileConfig::print_struct_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir",
                     &CompileConfig::print_kernel_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir_optimized",
                     &CompileConfig::print_kernel_llvm_ir_optimized)
      .def_readwrite("print_kernel_asm", &CompileConfig::print_kernel_asm)
      .def_readwrite("print_kernel_amdgcn", &CompileConfig::print_kernel_amdgcn)
      .def_readwrite("simplify_before_lower_access",
                     &CompileConfig::simplify_before_lower_access)
      .def_readwrite("simplify_after_lower_access",
                     &CompileConfig::simplify_after_lower_access)
      .def_readwrite("lower_access", &CompileConfig::lower_access)
      .def_readwrite("move_loop_invariant_outside_if",
                     &CompileConfig::move_loop_invariant_outside_if)
      .def_readwrite("cache_loop_invariant_global_vars",
                     &CompileConfig::cache_loop_invariant_global_vars)
      .def_readwrite("tiered_full_simplify",
                     &CompileConfig::tiered_full_simplify)
      .def_readwrite("full_simplify_global_iter_cap",
                     &CompileConfig::full_simplify_global_iter_cap)
      .def_readwrite("auto_real_function",
                     &CompileConfig::auto_real_function)
      .def_readwrite("auto_real_function_threshold_us",
                     &CompileConfig::auto_real_function_threshold_us)
      .def_readwrite("auto_real_function_inline_budget",
                     &CompileConfig::auto_real_function_inline_budget)
      .def_readwrite("default_cpu_block_dim",
                     &CompileConfig::default_cpu_block_dim)
      .def_readwrite("cpu_block_dim_adaptive",
                     &CompileConfig::cpu_block_dim_adaptive)
      .def_readwrite("default_gpu_block_dim",
                     &CompileConfig::default_gpu_block_dim)
      .def_readwrite("gpu_max_reg", &CompileConfig::gpu_max_reg)
      .def_readwrite("saturating_grid_dim", &CompileConfig::saturating_grid_dim)
      .def_readwrite("max_block_dim", &CompileConfig::max_block_dim)
      .def_readwrite("cpu_max_num_threads", &CompileConfig::cpu_max_num_threads)
      .def_readwrite("random_seed", &CompileConfig::random_seed)
      .def_readwrite("verbose_kernel_launches",
                     &CompileConfig::verbose_kernel_launches)
      .def_readwrite("verbose", &CompileConfig::verbose)
      .def_readwrite("demote_dense_struct_fors",
                     &CompileConfig::demote_dense_struct_fors)
      .def_readwrite("spirv_skip_intermediate_listgen",
                     &CompileConfig::spirv_skip_intermediate_listgen)
      .def_readwrite("spirv_listgen_subgroup_ballot",
                     &CompileConfig::spirv_listgen_subgroup_ballot)
      .def_readwrite("listgen_static_grid_dim",
                     &CompileConfig::listgen_static_grid_dim)
      .def_readwrite("vulkan_listgen_dynamic_size",
                     &CompileConfig::vulkan_listgen_dynamic_size)
      .def_readwrite("vulkan_listgen_buffer_MB",
                     &CompileConfig::vulkan_listgen_buffer_MB)
      .def_readwrite("vulkan_dispatch_cache",
                     &CompileConfig::vulkan_dispatch_cache)
      .def_readwrite("vulkan_dispatch_cache_size",
             &CompileConfig::vulkan_dispatch_cache_size)
      .def_readwrite("vulkan_descriptor_cache_lru",
             &CompileConfig::vulkan_descriptor_cache_lru)
      .def_readwrite("vulkan_listgen_lite_barrier",
             &CompileConfig::vulkan_listgen_lite_barrier)
      .def_readwrite("vulkan_listgen_reuse",
            &CompileConfig::vulkan_listgen_reuse)
      .def_readwrite("vulkan_listgen_reuse_adaptive",
             &CompileConfig::vulkan_listgen_reuse_adaptive)
      .def_readwrite("vulkan_spv_stats", &CompileConfig::vulkan_spv_stats)
      .def_readwrite("vulkan_spv_stats_filter",
            &CompileConfig::vulkan_spv_stats_filter)
      .def_readwrite("vulkan_spv_stats_capacity",
            &CompileConfig::vulkan_spv_stats_capacity)
      .def_readwrite("vulkan_spv_stats_to_stderr",
            &CompileConfig::vulkan_spv_stats_to_stderr)
      .def_readwrite("kernel_profiler", &CompileConfig::kernel_profiler)
      .def_readwrite("timeline", &CompileConfig::timeline)
      .def_readwrite("default_fp", &CompileConfig::default_fp)
      .def_readwrite("default_ip", &CompileConfig::default_ip)
      .def_readwrite("default_up", &CompileConfig::default_up)
      .def_readwrite("device_memory_GB", &CompileConfig::device_memory_GB)
      .def_readwrite("cuda_sparse_pool_size_GB",
                     &CompileConfig::cuda_sparse_pool_size_GB)
      .def_readwrite("cuda_sparse_pool_size_floor_MiB",
                     &CompileConfig::cuda_sparse_pool_size_floor_MiB)
      .def_readwrite("cuda_sparse_pool_auto_size",
                     &CompileConfig::cuda_sparse_pool_auto_size)
      .def_readwrite("cuda_sparse_per_snode_pool",
                     &CompileConfig::cuda_sparse_per_snode_pool)
      .def_readwrite("device_memory_fraction",
                     &CompileConfig::device_memory_fraction)
      .def_readwrite("fast_math", &CompileConfig::fast_math)
      .def_readwrite("advanced_optimization",
                     &CompileConfig::advanced_optimization)
      .def_readwrite("ad_stack_size", &CompileConfig::ad_stack_size)
      .def_readwrite("flatten_if", &CompileConfig::flatten_if)
      .def_readwrite("make_thread_local", &CompileConfig::make_thread_local)
      .def_readwrite("make_block_local", &CompileConfig::make_block_local)
      .def_readwrite("detect_read_only", &CompileConfig::detect_read_only)
      .def_readwrite("real_matrix_scalarize",
                     &CompileConfig::real_matrix_scalarize)
      .def_readwrite("force_scalarize_matrix",
                     &CompileConfig::force_scalarize_matrix)
      .def_readwrite("half2_vectorization", &CompileConfig::half2_vectorization)
      .def_readwrite("make_cpu_multithreading_loop",
                     &CompileConfig::make_cpu_multithreading_loop)
      .def_readwrite("quant_opt_store_fusion",
                     &CompileConfig::quant_opt_store_fusion)
      .def_readwrite("quant_opt_atomic_demotion",
                     &CompileConfig::quant_opt_atomic_demotion)
      .def_readwrite("allow_nv_shader_extension",
                     &CompileConfig::allow_nv_shader_extension)
      .def_readwrite("make_mesh_block_local",
                     &CompileConfig::make_mesh_block_local)
      .def_readwrite("mesh_localize_to_end_mapping",
                     &CompileConfig::mesh_localize_to_end_mapping)
      .def_readwrite("mesh_localize_from_end_mapping",
                     &CompileConfig::mesh_localize_from_end_mapping)
      .def_readwrite("optimize_mesh_reordered_mapping",
                     &CompileConfig::optimize_mesh_reordered_mapping)
      .def_readwrite("mesh_localize_all_attr_mappings",
                     &CompileConfig::mesh_localize_all_attr_mappings)
      .def_readwrite("demote_no_access_mesh_fors",
                     &CompileConfig::demote_no_access_mesh_fors)
      .def_readwrite("experimental_auto_mesh_local",
                     &CompileConfig::experimental_auto_mesh_local)
      .def_readwrite("auto_mesh_local_default_occupacy",
                     &CompileConfig::auto_mesh_local_default_occupacy)
      .def_readwrite("offline_cache", &CompileConfig::offline_cache)
      .def_readwrite("offline_cache_file_path",
                     &CompileConfig::offline_cache_file_path)
      .def_readwrite("offline_cache_cleaning_policy",
                     &CompileConfig::offline_cache_cleaning_policy)
      .def_readwrite("offline_cache_max_size_of_files",
                     &CompileConfig::offline_cache_max_size_of_files)
      .def_readwrite("offline_cache_cleaning_factor",
                     &CompileConfig::offline_cache_cleaning_factor)
      .def_readwrite("num_compile_threads", &CompileConfig::num_compile_threads)
      .def_readwrite("vk_api_version", &CompileConfig::vk_api_version)
      .def_readwrite("vulkan_launch_buffer_pool",
                     &CompileConfig::vulkan_launch_buffer_pool)
      .def_readwrite("vulkan_launch_buffer_pool_capacity",
                     &CompileConfig::vulkan_launch_buffer_pool_capacity)
      .def_readwrite("gfx_ctx_buffer_ring",
            &CompileConfig::gfx_ctx_buffer_ring)
      .def_readwrite("gfx_ctx_buffer_ring_size",
            &CompileConfig::gfx_ctx_buffer_ring_size)
      .def_readwrite("gfx_cmdlist_lazy_submit",
                     &CompileConfig::gfx_cmdlist_lazy_submit)
      .def_readwrite("gfx_cmdlist_max_dispatches",
                     &CompileConfig::gfx_cmdlist_max_dispatches)
      .def_readwrite("cuda_stack_limit", &CompileConfig::cuda_stack_limit)
      .def_readwrite("external_optimization_level",
                     &CompileConfig::external_optimization_level);

  m.def("reset_default_compile_config",
        [&]() { default_compile_config = CompileConfig(); });
  m.def("_set_check_out_of_bound_explicit", [](bool explicit_value) {
    default_compile_config.check_out_of_bound_explicit = explicit_value;
  });

  m.def(
      "default_compile_config",
      [&]() -> CompileConfig & { return default_compile_config; },
      py::return_value_policy::reference);

  m.def("get_last_vulkan_spv_stats", []() {
    py::list ret;
    for (const auto &s : spirv::get_last_spv_stats()) {
      py::dict item;
      item["kernel"] = s.kernel_name;
      item["task_id"] = s.task_id;
      item["task_name"] = s.task_name;
      item["type"] = s.task_type;
      item["snode_id"] = s.snode_id;
      item["word_before"] = py::int_(s.before_words);
      item["word_after"] = py::int_(s.after_words);
      item["opt_run"] = s.opt_run;
      item["opt_ok"] = s.opt_ok;
      item["duration_us"] = s.opt_us;
      item["is_listgen"] = s.listgen_related;
      item["is_pointer"] = s.pointer_related;
      item["skipped_passes"] = s.skipped_passes;
      ret.append(item);
    }
    return ret;
  });

  m.def("get_fs_inner_stats", []() {
    uint64_t e = 0, n = 0, it = 0;
    taichi::lang::irpass::get_fs_inner_stats(&e, &n, &it);
    return py::make_tuple(e, n, it);
  });
  m.def("reset_fs_inner_stats",
        &taichi::lang::irpass::reset_fs_inner_stats);

  py::class_<Program::KernelProfilerQueryResult>(m, "KernelProfilerQueryResult")
      .def_readwrite("counter", &Program::KernelProfilerQueryResult::counter)
      .def_readwrite("min", &Program::KernelProfilerQueryResult::min)
      .def_readwrite("max", &Program::KernelProfilerQueryResult::max)
      .def_readwrite("avg", &Program::KernelProfilerQueryResult::avg);

  py::class_<KernelProfileTracedRecord>(m, "KernelProfileTracedRecord")
      .def_readwrite("register_per_thread",
                     &KernelProfileTracedRecord::register_per_thread)
      .def_readwrite("shared_mem_per_block",
                     &KernelProfileTracedRecord::shared_mem_per_block)
      .def_readwrite("grid_size", &KernelProfileTracedRecord::grid_size)
      .def_readwrite("block_size", &KernelProfileTracedRecord::block_size)
      .def_readwrite(
          "active_blocks_per_multiprocessor",
          &KernelProfileTracedRecord::active_blocks_per_multiprocessor)
      .def_readwrite("kernel_time",
                     &KernelProfileTracedRecord::kernel_elapsed_time_in_ms)
      .def_readwrite("base_time", &KernelProfileTracedRecord::time_since_base)
      .def_readwrite("name", &KernelProfileTracedRecord::name)
      .def_readwrite("metric_values",
                     &KernelProfileTracedRecord::metric_values);

  py::enum_<SNodeAccessFlag>(m, "SNodeAccessFlag", py::arithmetic())
      .value("block_local", SNodeAccessFlag::block_local)
      .value("read_only", SNodeAccessFlag::read_only)
      .value("mesh_local", SNodeAccessFlag::mesh_local)
      .export_values();

  // Export ASTBuilder
  py::class_<ASTBuilder>(m, "ASTBuilder")
      .def("make_id_expr", &ASTBuilder::make_id_expr)
      .def("create_kernel_exprgroup_return",
           &ASTBuilder::create_kernel_exprgroup_return)
      .def("create_print", &ASTBuilder::create_print)
      .def("begin_func", &ASTBuilder::begin_func)
      .def("end_func", &ASTBuilder::end_func)
      .def("stop_grad", &ASTBuilder::stop_gradient)
      .def("begin_frontend_if", &ASTBuilder::begin_frontend_if)
      .def("begin_frontend_if_true", &ASTBuilder::begin_frontend_if_true)
      .def("pop_scope", &ASTBuilder::pop_scope)
      .def("begin_frontend_if_false", &ASTBuilder::begin_frontend_if_false)
      .def("insert_deactivate", &ASTBuilder::insert_snode_deactivate)
      .def("insert_activate", &ASTBuilder::insert_snode_activate)
      .def("expr_snode_get_addr", &ASTBuilder::snode_get_addr)
      .def("expr_snode_append", &ASTBuilder::snode_append)
      .def("expr_snode_is_active", &ASTBuilder::snode_is_active)
      .def("expr_snode_length", &ASTBuilder::snode_length)
      .def("insert_external_func_call", &ASTBuilder::insert_external_func_call)
      .def("make_matrix_expr", &ASTBuilder::make_matrix_expr)
      .def("expr_alloca", &ASTBuilder::expr_alloca)
      .def("expr_alloca_shared_array", &ASTBuilder::expr_alloca_shared_array)
      .def("create_assert_stmt", &ASTBuilder::create_assert_stmt)
      .def("expr_assign", &ASTBuilder::expr_assign)
      .def("begin_frontend_range_for", &ASTBuilder::begin_frontend_range_for)
      .def("end_frontend_range_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_struct_for_on_snode",
           &ASTBuilder::begin_frontend_struct_for_on_snode)
      .def("begin_frontend_struct_for_on_external_tensor",
           &ASTBuilder::begin_frontend_struct_for_on_external_tensor)
      .def("end_frontend_struct_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_mesh_for", &ASTBuilder::begin_frontend_mesh_for)
      .def("end_frontend_mesh_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_while", &ASTBuilder::begin_frontend_while)
      .def("insert_break_stmt", &ASTBuilder::insert_break_stmt)
      .def("insert_continue_stmt", &ASTBuilder::insert_continue_stmt)
      .def("insert_expr_stmt", &ASTBuilder::insert_expr_stmt)
      .def("insert_thread_idx_expr", &ASTBuilder::insert_thread_idx_expr)
      .def("insert_patch_idx_expr", &ASTBuilder::insert_patch_idx_expr)
      .def("make_texture_op_expr", &ASTBuilder::make_texture_op_expr)
      .def("expand_exprs", &ASTBuilder::expand_exprs)
      .def("mesh_index_conversion", &ASTBuilder::mesh_index_conversion)
      .def("expr_subscript", &ASTBuilder::expr_subscript)
      .def("insert_func_call", &ASTBuilder::insert_func_call)
      .def("sifakis_svd_f32", sifakis_svd_export<float32, int32>)
      .def("sifakis_svd_f64", sifakis_svd_export<float64, int64>)
      .def("expr_var", &ASTBuilder::make_var)
      .def("bit_vectorize", &ASTBuilder::bit_vectorize)
      .def("parallelize", &ASTBuilder::parallelize)
      .def("strictly_serialize", &ASTBuilder::strictly_serialize)
      .def("block_dim", &ASTBuilder::block_dim)
      .def("insert_snode_access_flag", &ASTBuilder::insert_snode_access_flag)
      .def("reset_snode_access_flag", &ASTBuilder::reset_snode_access_flag);

  py::class_<DeviceCapabilityConfig>(
      m, "DeviceCapabilityConfig");  // NOLINT(bugprone-unused-raii)

  py::class_<CompiledKernelData>(
      m, "CompiledKernelData");  // NOLINT(bugprone-unused-raii)

  // Internal F2 validation surface. The public API remains Graph.run() -> None
  // until CPU/CUDA/Vulkan Ticket semantics and reset/fault lifetime complete
  // the F2.3 acceptance gate.
  py::class_<RuntimeCompletion>(m, "_RuntimeCompletion")
      .def("done", &RuntimeCompletion::done,
           py::call_guard<py::gil_scoped_release>())
      .def("wait", &RuntimeCompletion::wait,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("backend", [](const RuntimeCompletion &value) {
        return arch_name(value.backend());
      })
      .def_property_readonly("program_domain",
                             &RuntimeCompletion::program_domain)
      .def_property_readonly("sequence", &RuntimeCompletion::sequence)
      .def_property_readonly("has_backend_work",
                             &RuntimeCompletion::has_backend_work)
      .def_property_readonly("first_error",
                             &RuntimeCompletion::first_error_message);

  py::class_<Program::RuntimeSubmissionTransaction>(
      m, "_RuntimeSubmissionTransaction")
      .def("_mark_submission",
           &Program::RuntimeSubmissionTransaction::mark_submission)
      .def("_finish", &Program::RuntimeSubmissionTransaction::finish,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Program>(m, "Program")
      .def(py::init<>())
      .def("config", &Program::compile_config,
           py::return_value_policy::reference)
      .def("sync_kernel_profiler",
           [](Program *program) { program->profiler->sync(); })
      .def("update_kernel_profiler",
           [](Program *program) { program->profiler->update(); })
      .def("clear_kernel_profiler",
           [](Program *program) { program->profiler->clear(); })
      .def("query_kernel_profile_info",
           [](Program *program, const std::string &name) {
             return program->query_kernel_profile_info(name);
           })
      .def("get_kernel_profiler_records",
           [](Program *program) {
             return program->profiler->get_traced_records();
           })
      .def(
          "get_kernel_profiler_device_name",
          [](Program *program) { return program->profiler->get_device_name(); })
      .def("reinit_kernel_profiler_with_metrics",
           [](Program *program, const std::vector<std::string> metrics) {
             return program->profiler->reinit_with_metrics(metrics);
           })
      .def("kernel_profiler_total_time",
           [](Program *program) { return program->profiler->get_total_time(); })
      .def("set_kernel_profiler_toolkit",
           [](Program *program, const std::string toolkit_name) {
             return program->profiler->set_profiler_toolkit(toolkit_name);
           })
      .def("timeline_clear",
           [](Program *) { Timelines::get_instance().clear(); })
      .def("timeline_save",
           [](Program *, const std::string &fn) {
             Timelines::get_instance().save(fn);
           })
      .def("print_memory_profiler_info", &Program::print_memory_profiler_info)
      .def("finalize", &Program::finalize)
      .def("get_total_compilation_time", &Program::get_total_compilation_time)
      .def("get_snode_num_dynamically_allocated",
           &Program::get_snode_num_dynamically_allocated)
      .def("reset_hash_snode_probe_stats",
           &Program::reset_hash_snode_probe_stats)
      .def("get_hash_snode_probe_stats",
           [](Program *program) {
             auto raw = program->get_hash_snode_probe_stats();
             py::dict ret;
             ret["insert_count"] = raw.size() > 0 ? raw[0] : 0;
             ret["insert_total"] = raw.size() > 1 ? raw[1] : 0;
             ret["insert_max"] = raw.size() > 2 ? raw[2] : 0;
             ret["lookup_count"] = raw.size() > 3 ? raw[3] : 0;
             ret["lookup_total"] = raw.size() > 4 ? raw[4] : 0;
             ret["lookup_max"] = raw.size() > 5 ? raw[5] : 0;
             ret["insert_mean"] =
                 raw.size() > 1 && raw[0] > 0 ? double(raw[1]) / raw[0] : 0.0;
             ret["lookup_mean"] =
                 raw.size() > 4 && raw[3] > 0 ? double(raw[4]) / raw[3] : 0.0;
             return ret;
           })
      .def("synchronize", &Program::synchronize)
      .def("_runtime_has_fatal_fault", &Program::runtime_has_fatal_fault)
      .def("_debug_inject_runtime_fault",
           &Program::debug_inject_runtime_fault)
      .def("_debug_runtime_fault_state", [](const Program &program) {
        const RuntimeFaultSnapshot snapshot = program.runtime_fault_snapshot();
        py::dict result;
        result["state"] = runtime_lifecycle_state_name(snapshot.state);
        result["domain"] = snapshot.program_domain;
        result["rejected_submissions"] = snapshot.rejected_submissions;
        if (snapshot.first_fault) {
          const RuntimeFaultRecord &fault = *snapshot.first_fault;
          result["backend"] = arch_name(fault.backend);
          result["backend_code"] = fault.backend_code;
          result["sequence"] = fault.submission_sequence;
          result["operation"] = fault.operation;
          result["message"] = fault.message;
        } else {
          result["backend"] = py::none();
          result["backend_code"] = py::none();
          result["sequence"] = py::none();
          result["operation"] = py::none();
          result["message"] = py::none();
        }
        return result;
      })
      .def("_runtime_statistics_snapshot", [](Program &program) {
        const RuntimeStatisticsSnapshot snapshot =
            program.runtime_statistics_snapshot();
        auto optional_counter =
            [](const RuntimeOptionalCounter &counter) -> py::object {
          if (!counter.available) {
            return py::none();
          }
          return py::int_(counter.value);
        };

        py::dict submission;
        submission["kernel_submissions"] =
            snapshot.submission.kernel_submissions;
        submission["graph_submissions"] =
            snapshot.submission.graph_submissions;
        submission["graph_backend_submissions"] =
            snapshot.submission.graph_backend_submissions;
        submission["native_submissions"] =
            snapshot.submission.native_submissions;
        submission["failed_submissions"] =
            snapshot.submission.failed_submissions;

        py::dict synchronization;
        synchronization["program_syncs"] =
            snapshot.synchronization.program_syncs;
        synchronization["program_sync_wait_ns"] =
            snapshot.synchronization.program_sync_wait_ns;
        synchronization["completion_polls"] =
            snapshot.synchronization.completion_polls;
        synchronization["completion_waits"] =
            snapshot.synchronization.completion_waits;
        synchronization["completion_wait_ns"] =
            snapshot.synchronization.completion_wait_ns;
        synchronization["backend_waits"] =
            optional_counter(snapshot.synchronization.backend_waits);
        synchronization["backend_wait_ns"] =
            optional_counter(snapshot.synchronization.backend_wait_ns);
        synchronization["backend_lock_samples"] =
            optional_counter(snapshot.synchronization.backend_lock_samples);
        synchronization["backend_lock_contentions"] = optional_counter(
            snapshot.synchronization.backend_lock_contentions);
        synchronization["backend_lock_sampled_wait_ns"] = optional_counter(
            snapshot.synchronization.backend_lock_sampled_wait_ns);

        py::dict memory;
        memory["live_resources"] = snapshot.memory.live_resources;
        memory["retiring_resources"] = snapshot.memory.retiring_resources;
        memory["inflight_resources"] = snapshot.memory.inflight_resources;
        const auto &host_allocator_snapshot =
            snapshot.memory.host_allocator;
        py::dict host_allocator;
        host_allocator["requested_live_bytes"] = optional_counter(
            host_allocator_snapshot.requested_live_bytes);
        host_allocator["peak_requested_live_bytes"] = optional_counter(
            host_allocator_snapshot.peak_requested_live_bytes);
        host_allocator["reserved_bytes"] = optional_counter(
            host_allocator_snapshot.reserved_bytes);
        host_allocator["committed_bytes"] = optional_counter(
            host_allocator_snapshot.committed_bytes);
        host_allocator["capacity_bytes"] = optional_counter(
            host_allocator_snapshot.capacity_bytes);
        host_allocator["used_bytes"] =
            optional_counter(host_allocator_snapshot.used_bytes);
        host_allocator["available_bytes"] = optional_counter(
            host_allocator_snapshot.available_bytes);
        host_allocator["alignment_waste_bytes"] = optional_counter(
            host_allocator_snapshot.alignment_waste_bytes);
        host_allocator["unreclaimed_released_bytes"] = optional_counter(
            host_allocator_snapshot.unreclaimed_released_bytes);
        host_allocator["wasted_bytes"] =
            optional_counter(host_allocator_snapshot.wasted_bytes);
        host_allocator["chunk_count"] =
            optional_counter(host_allocator_snapshot.chunk_count);
        host_allocator["slab_chunk_count"] = optional_counter(
            host_allocator_snapshot.slab_chunk_count);
        host_allocator["large_chunk_count"] = optional_counter(
            host_allocator_snapshot.large_chunk_count);
        host_allocator["exclusive_chunk_count"] = optional_counter(
            host_allocator_snapshot.exclusive_chunk_count);
        host_allocator["peak_reserved_bytes"] = optional_counter(
            host_allocator_snapshot.peak_reserved_bytes);
        host_allocator["peak_used_bytes"] = optional_counter(
            host_allocator_snapshot.peak_used_bytes);
        host_allocator["peak_wasted_bytes"] = optional_counter(
            host_allocator_snapshot.peak_wasted_bytes);
        host_allocator["peak_chunk_count"] = optional_counter(
            host_allocator_snapshot.peak_chunk_count);
        memory["host_allocator"] = host_allocator;
        memory["host_requested_live_bytes"] =
            optional_counter(snapshot.memory.host_requested_live_bytes);
        memory["host_raw_bytes"] =
            optional_counter(snapshot.memory.host_raw_bytes);
        memory["host_capacity_bytes"] =
            optional_counter(snapshot.memory.host_capacity_bytes);
        memory["device_requested_live_bytes"] =
            optional_counter(snapshot.memory.device_requested_live_bytes);
        memory["device_raw_bytes"] =
            optional_counter(snapshot.memory.device_raw_bytes);
        memory["device_cached_bytes"] =
            optional_counter(snapshot.memory.device_cached_bytes);
        memory["cuda_mempool_reserved_bytes"] =
            optional_counter(snapshot.memory.cuda_mempool_reserved_bytes);
        memory["cuda_mempool_used_bytes"] =
            optional_counter(snapshot.memory.cuda_mempool_used_bytes);

        py::dict transfer;
        transfer["host_to_device_bytes"] =
            snapshot.transfer.host_to_device_bytes;
        transfer["device_to_host_bytes"] =
            snapshot.transfer.device_to_host_bytes;
        transfer["device_to_device_bytes"] =
            snapshot.transfer.device_to_device_bytes;
        transfer["cuda_vulkan_direct_bytes"] =
            snapshot.transfer.cuda_vulkan_direct_bytes;
        transfer["cuda_vulkan_fallback_bytes"] =
            snapshot.transfer.cuda_vulkan_fallback_bytes;

        py::dict graph;
        graph["captures"] = snapshot.graph.captures;
        graph["recaptures"] = snapshot.graph.recaptures;
        graph["replays"] = snapshot.graph.replays;
        graph["ordinary_fallbacks"] =
            snapshot.graph.ordinary_fallbacks;
        graph["replay_slot_saturation_fallbacks"] =
            snapshot.graph.replay_slot_saturation_fallbacks;

        py::dict display;
        display["accepted_frames"] = snapshot.display.accepted_frames;
        display["submitted_frames"] = snapshot.display.submitted_frames;
        display["dropped_frames"] = snapshot.display.dropped_frames;
        display["accepted_frame_bytes"] =
            snapshot.display.accepted_frame_bytes;

        const RuntimeFaultSnapshot fault_snapshot =
            program.runtime_fault_snapshot();
        py::dict fault;
        fault["state"] = runtime_lifecycle_state_name(fault_snapshot.state);
        fault["first_fatal_faults"] =
            snapshot.fault.first_fatal_faults;
        fault["rejected_submissions"] =
            snapshot.fault.rejected_submissions;
        if (fault_snapshot.first_fault) {
          const RuntimeFaultRecord &first = *fault_snapshot.first_fault;
          py::dict first_fault;
          first_fault["backend"] = arch_name(first.backend);
          first_fault["backend_code"] = first.backend_code;
          first_fault["sequence"] = first.submission_sequence;
          first_fault["operation"] = first.operation;
          first_fault["message"] = first.message;
          fault["first_fault"] = std::move(first_fault);
        } else {
          fault["first_fault"] = py::none();
        }

        py::dict trace;
        trace["recorded_events"] = snapshot.trace.recorded_events;
        trace["dropped_events"] = snapshot.trace.dropped_events;

        py::dict result;
        result["schema_version"] = snapshot.schema_version;
        result["backend"] = arch_name(snapshot.backend);
        result["program_domain"] = snapshot.program_domain;
        result["submission"] = std::move(submission);
        result["synchronization"] = std::move(synchronization);
        result["memory"] = std::move(memory);
        result["transfer"] = std::move(transfer);
        result["graph"] = std::move(graph);
        result["display"] = std::move(display);
        result["fault"] = std::move(fault);
        result["trace"] = std::move(trace);
        return result;
      })
      .def("_primitive_workspace_stats", [](const Program &program) {
        return primitive_workspace_snapshot_to_dict(
            program.primitive_workspace_snapshot());
      })
      .def("_primitive_workspace_set_budget_bytes",
           &Program::set_primitive_workspace_budget)
      .def("_primitive_workspace_clear", &Program::clear_primitive_workspaces,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "_runtime_trace_start",
          [](Program &program, std::size_t max_threads,
             std::size_t events_per_thread) {
            RuntimeTraceSnapshot snapshot;
            {
              py::gil_scoped_release release;
              snapshot =
                  program.runtime_trace().start(max_threads, events_per_thread);
            }
            return runtime_trace_snapshot_to_dict(snapshot);
          },
          py::arg("max_threads") = RuntimeTraceRecorder::kDefaultMaxThreads,
          py::arg("events_per_thread") =
              RuntimeTraceRecorder::kDefaultEventsPerThread)
      .def("_runtime_trace_stop", [](Program &program) {
        RuntimeTraceSnapshot snapshot;
        {
          py::gil_scoped_release release;
          snapshot = program.runtime_trace().stop();
        }
        return runtime_trace_snapshot_to_dict(snapshot);
      })
      .def("_runtime_trace_snapshot", [](const Program &program) {
        return runtime_trace_snapshot_to_dict(program.runtime_trace().snapshot());
      })
      .def("_runtime_trace_export", [](const Program &program,
                                        const std::string &path) {
        py::gil_scoped_release release;
        return program.runtime_trace().export_chrome_trace(path);
      })
      .def("_record_runtime_completion",
           &Program::record_runtime_completion,
           py::call_guard<py::gil_scoped_release>())
      .def("_record_runtime_graph_submission", [](Program &program) {
        program.record_runtime_submission_stat(RuntimeSubmissionKind::kGraph);
      })
      .def("_begin_runtime_submission_transaction",
           &Program::begin_runtime_submission_transaction,
           py::call_guard<py::gil_scoped_release>())
      .def("_debug_runtime_completion_stats",
           &Program::debug_runtime_completion_stats)
      .def("_debug_kernel_definition_count", [](const Program &program) {
        return program.kernels.size();
      })
      .def("_debug_kernel_lifecycle_stats", [](Program &program) {
        std::size_t live_definitions = 0;
        std::size_t retired_shells = 0;
        for (const auto &kernel : program.kernels) {
          if (kernel->ir == nullptr) {
            ++retired_shells;
          } else {
            ++live_definitions;
          }
        }
        py::dict result;
        result["total_slots"] = program.kernels.size();
        result["live_definitions"] = live_definitions;
        result["retired_shells"] = retired_shells;
        result["retired_shell_inline_bytes_lower_bound"] =
            retired_shells * sizeof(Kernel);
        result["retired_shell_total_owned_bytes_reported"] = false;
        result["registered_executables"] =
            program.get_kernel_launcher().debug_registered_kernel_count();
        return result;
      })
      .def("_debug_kernel_registration_count", [](Program &program) {
        return program.get_kernel_launcher().debug_registered_kernel_count();
      })
      .def("_debug_snode_field_mapping_count", [](Program &program) {
        return program.get_snode_to_fields()->size();
      })
      .def("_debug_sparse_snode_tree_stats",
           [](Program &program, int tree_id) {
             SparseSNodeTreeStatistics snapshot;
             {
               py::gil_scoped_release release;
               snapshot =
                   program.debug_sparse_snode_tree_statistics(tree_id);
             }
             const auto optional_counter =
                 [](const RuntimeOptionalCounter &counter) -> py::object {
               if (!counter.available) {
                 return py::none();
               }
               return py::int_(counter.value);
             };
             const auto &source = snapshot.memory;
             py::dict memory;
             memory["tree_owned_reserved_bytes"] =
                 optional_counter(source.tree_owned_reserved_bytes);
             memory["root_reserved_bytes"] =
                 optional_counter(source.root_reserved_bytes);
             memory["sparse_pool_reserved_bytes"] =
                 optional_counter(source.sparse_pool_reserved_bytes);
             memory["runtime_metadata_requested_bytes"] =
                 optional_counter(source.runtime_metadata_requested_bytes);
             memory["direct_ambient_requested_bytes"] =
                 optional_counter(source.direct_ambient_requested_bytes);
             memory["allocator_payload_reserved_bytes"] =
                 optional_counter(source.allocator_payload_reserved_bytes);
             memory["allocator_payload_used_bytes"] =
                 optional_counter(source.allocator_payload_used_bytes);
             memory["allocator_bookkeeping_reserved_bytes"] =
                 optional_counter(
                     source.allocator_bookkeeping_reserved_bytes);
             memory["active_list_reserved_bytes"] =
                 optional_counter(source.active_list_reserved_bytes);
             memory["active_list_used_bytes"] =
                 optional_counter(source.active_list_used_bytes);
             memory["allocator_in_use_elements"] =
                 optional_counter(source.allocator_in_use_elements);
             memory["allocator_free_elements"] =
                 optional_counter(source.allocator_free_elements);
             memory["allocator_recycled_elements"] =
                 optional_counter(source.allocator_recycled_elements);
             memory["shared_listgen_workspace_reserved_bytes"] =
                 optional_counter(
                     source.shared_listgen_workspace_reserved_bytes);
             memory["tree_owned_scope"] = source.tree_owned_scope;
             memory["runtime_resource_scope"] =
                 source.runtime_resource_scope;
             memory["shared_listgen_workspace_scope"] =
                 source.shared_listgen_workspace_scope;

             py::list listgen_nodes;
             std::uint64_t requests = 0;
             std::uint64_t rebuilds = 0;
             std::uint64_t reuse_hits = 0;
             std::uint64_t invalidations = 0;
             std::uint64_t resident_evictions = 0;
             std::uint64_t candidate_slots = 0;
             std::uint64_t scanned_elements = 0;
             std::uint64_t emitted_elements = 0;
             std::uint64_t serial_rebuilds = 0;
             std::uint64_t parallel_rebuilds = 0;
             bool candidate_slots_available = true;
             bool resident_evictions_available = true;
             bool scanned_elements_available = true;
             bool emitted_elements_available = true;
             bool serial_rebuilds_available = true;
             bool parallel_rebuilds_available = true;
             for (const auto &node : snapshot.listgen.nodes) {
               py::dict item;
               item["snode_id"] = node.snode_id;
               item["parent_snode_id"] = node.parent_snode_id;
               item["requests"] = node.requests;
               item["rebuilds"] = node.rebuilds;
               item["reuse_hits"] = node.reuse_hits;
               item["invalidations"] = node.invalidations;
               item["resident_evictions"] =
                   optional_counter(node.resident_evictions);
               item["candidate_slots_dispatched"] =
                   optional_counter(node.candidate_slots_dispatched);
               item["scanned_elements"] =
                   optional_counter(node.scanned_elements);
               item["emitted_elements"] =
                   optional_counter(node.emitted_elements);
               item["serial_rebuilds"] =
                   optional_counter(node.serial_rebuilds);
               item["parallel_rebuilds"] =
                   optional_counter(node.parallel_rebuilds);
               item["last_rebuild_reason"] = node.last_rebuild_reason;
               listgen_nodes.append(std::move(item));
               requests += node.requests;
               rebuilds += node.rebuilds;
               reuse_hits += node.reuse_hits;
               invalidations += node.invalidations;
               resident_evictions_available &=
                   node.resident_evictions.available;
               resident_evictions += node.resident_evictions.value;
               candidate_slots_available &=
                   node.candidate_slots_dispatched.available;
               candidate_slots += node.candidate_slots_dispatched.value;
               scanned_elements_available &= node.scanned_elements.available;
               scanned_elements += node.scanned_elements.value;
               emitted_elements_available &= node.emitted_elements.available;
               emitted_elements += node.emitted_elements.value;
               serial_rebuilds_available &= node.serial_rebuilds.available;
               serial_rebuilds += node.serial_rebuilds.value;
               parallel_rebuilds_available &= node.parallel_rebuilds.available;
               parallel_rebuilds += node.parallel_rebuilds.value;
             }
             py::dict listgen_totals;
             listgen_totals["requests"] = requests;
             listgen_totals["rebuilds"] = rebuilds;
             listgen_totals["reuse_hits"] = reuse_hits;
             listgen_totals["invalidations"] = invalidations;
             listgen_totals["resident_evictions"] =
                 resident_evictions_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(resident_evictions)
                     : py::none();
             listgen_totals["candidate_slots_dispatched"] =
                 candidate_slots_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(candidate_slots)
                     : py::none();
             listgen_totals["scanned_elements"] =
                 scanned_elements_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(scanned_elements)
                     : py::none();
             listgen_totals["emitted_elements"] =
                 emitted_elements_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(emitted_elements)
                     : py::none();
             listgen_totals["serial_rebuilds"] =
                 serial_rebuilds_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(serial_rebuilds)
                     : py::none();
             listgen_totals["parallel_rebuilds"] =
                 parallel_rebuilds_available && !snapshot.listgen.nodes.empty()
                     ? py::cast(parallel_rebuilds)
                     : py::none();
             py::dict listgen;
             listgen["available"] = snapshot.listgen.available;
             listgen["totals"] = std::move(listgen_totals);
             listgen["nodes"] = std::move(listgen_nodes);

             py::dict result;
             result["schema_version"] = 1;
             result["tree_id"] = snapshot.tree_id;
             result["generation"] = snapshot.generation;
             result["layout_fingerprint"] = snapshot.layout_fingerprint;
             result["backend"] = arch_name(snapshot.backend);
             result["memory"] = std::move(memory);
             result["listgen"] = std::move(listgen);
             return result;
           })
      .def("_debug_reset_sparse_listgen_stats",
           &Program::debug_reset_sparse_listgen_statistics,
           py::call_guard<py::gil_scoped_release>())
      .def("materialize_runtime", &Program::materialize_runtime)
      .def("make_aot_module_builder", &Program::make_aot_module_builder)
      .def("get_snode_tree_size", &Program::get_snode_tree_size)
      .def("get_active_snode_tree_ids", &Program::get_active_snode_tree_ids)
      .def("get_snode_root", &Program::get_snode_root,
           py::return_value_policy::reference)
      .def(
          "create_kernel",
          [](Program *program, const std::function<void(Kernel *)> &body,
             const std::string &name, AutodiffMode autodiff_mode) -> Kernel * {
            py::gil_scoped_release release;
            return &program->kernel(body, name, autodiff_mode);
          },
          py::return_value_policy::reference)
      .def("create_function", &Program::create_function,
           py::return_value_policy::reference)
       .def("create_sparse_matrix",
           [](Program *program, int n, int m, DataType dtype,
              std::string storage_format) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                             !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             if (arch_is_cpu(program->compile_config().arch))
               return make_sparse_matrix(n, m, dtype, storage_format);
              else
                return make_cu_sparse_matrix(n, m, dtype);
            })
      .def("_create_bsr_pattern",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices) {
             return std::make_shared<SparseBsrPattern>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices);
           },
           py::keep_alive<0, 1>())
      .def("_create_csr_pattern",
           [](Program *program, int rows, int cols,
              const Ndarray &row_offsets, const Ndarray &column_indices) {
             return std::make_shared<SparseCsrPattern>(
                 program, rows, cols, row_offsets, column_indices);
           },
           py::keep_alive<0, 1>())
      .def("_create_compiled_kernel_linear_operator",
           [](Program *program, Kernel &kernel, int size,
              std::uint64_t topology_version,
              std::uint64_t numeric_version,
              const Ndarray &operator_data) -> std::unique_ptr<SparseMatrix> {
             return std::make_unique<CompiledKernelLinearOperator>(
                 program, kernel, size, topology_version, numeric_version,
                 operator_data);
           },
           py::keep_alive<0, 1>())
      .def("_create_compiled_kernel_linear_operator_with_numeric_data",
           [](Program *program, Kernel &kernel, int size,
              std::uint64_t topology_version,
              std::uint64_t numeric_version,
              const Ndarray &topology_data,
              const Ndarray &numeric_data) -> std::unique_ptr<SparseMatrix> {
             return std::make_unique<CompiledKernelLinearOperator>(
                 program, kernel, size, topology_version, numeric_version,
                 topology_data, numeric_data);
            },
            py::keep_alive<0, 1>())
      .def("_create_compiled_graph_linear_operator",
           [](Program *program, const aot::CompiledGraph &graph, int size,
              std::uint64_t topology_version,
              std::uint64_t numeric_version, const py::dict &fixed_i32_args,
              const py::dict &topology_args, const py::dict &numeric_args,
              const py::dict &workspace_args) -> std::unique_ptr<SparseMatrix> {
             CompiledGraphLinearOperator::FixedI32Arguments fixed_i32;
             for (const auto &item : fixed_i32_args) {
               fixed_i32.emplace(py::cast<std::string>(item.first),
                                 py::cast<std::int32_t>(item.second));
             }
             auto parse_ndarrays = [](const py::dict &arguments) {
               CompiledGraphLinearOperator::NdarrayArguments result;
               for (const auto &item : arguments) {
                 result.emplace(py::cast<std::string>(item.first),
                                &py::cast<const Ndarray &>(item.second));
               }
               return result;
             };
             return std::make_unique<CompiledGraphLinearOperator>(
                 program, graph, size, topology_version, numeric_version,
                 std::move(fixed_i32), parse_ndarrays(topology_args),
                 parse_ndarrays(numeric_args),
                 parse_ndarrays(workspace_args));
           },
           py::keep_alive<0, 1>())
      .def("_create_csr_matrix_from_pattern",
           [](Program *program, std::shared_ptr<SparseCsrPattern> pattern,
              const Ndarray &values) -> std::unique_ptr<SparseMatrix> {
             TI_ERROR_IF(!pattern || pattern->program() != program,
                         "Internal CSR operator construction requires a "
                         "pattern owned by the same Program.");
             if (arch_is_cpu(program->compile_config().arch)) {
               return std::make_unique<CpuSparseCsrMatrix>(
                   std::move(pattern), values);
             }
             if (arch_is_cuda(program->compile_config().arch)) {
               return std::make_unique<CuSparseMatrix>(
                   std::move(pattern), values);
             }
             if (program->compile_config().arch == Arch::vulkan) {
               return std::make_unique<VulkanSparseMatrix>(
                   std::move(pattern), values);
             }
             TI_ERROR(
                 "Internal CSR operators support CPU, CUDA, and Vulkan "
                 "backends, got {}.",
                 arch_name(program->compile_config().arch));
           },
           py::keep_alive<0, 1>())
      .def("_create_bsr_matrix_from_pattern",
           [](Program *program, std::shared_ptr<SparseBsrPattern> pattern,
              const Ndarray &values) -> std::unique_ptr<SparseMatrix> {
             TI_ERROR_IF(!pattern || pattern->program() != program,
                         "Internal BSR operator construction requires a "
                         "pattern owned by the same Program.");
             if (arch_is_cpu(program->compile_config().arch)) {
               return std::make_unique<CpuSparseBsrMatrix>(
                   std::move(pattern), values);
             }
             if (arch_is_cuda(program->compile_config().arch)) {
               return std::make_unique<CuSparseBsrMatrix>(
                   std::move(pattern), values);
             }
             if (program->compile_config().arch == Arch::vulkan) {
               return std::make_unique<VulkanSparseBsrMatrix>(
                   std::move(pattern), values);
             }
             TI_ERROR(
                 "Internal BSR operators support CPU, CUDA, and Vulkan "
                 "backends, got {}.",
                 arch_name(program->compile_config().arch));
           },
           py::keep_alive<0, 1>())
      .def("_create_cpu_bsr_matrix",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices,
              const Ndarray &values) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch),
                         "Internal CPU BSR matrices require a CPU backend.");
             return std::make_unique<CpuSparseBsrMatrix>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_create_cuda_bsr_matrix",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices,
              const Ndarray &values) {
             TI_ERROR_IF(!arch_is_cuda(program->compile_config().arch),
                         "Internal BSR matrices require the CUDA backend.");
             return std::make_unique<CuSparseBsrMatrix>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_create_vulkan_csr_matrix",
           [](Program *program, int rows, int cols, const Ndarray &row_offsets,
              const Ndarray &column_indices, const Ndarray &values) {
             TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
                         "Internal Vulkan CSR matrices require the Vulkan "
                         "backend.");
             return std::make_unique<VulkanSparseMatrix>(
                 program, rows, cols, row_offsets, column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_create_vulkan_bsr_matrix",
           [](Program *program, int block_rows, int block_cols, int block_size,
              const Ndarray &row_offsets, const Ndarray &column_indices,
              const Ndarray &values) {
             TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
                         "Internal Vulkan BSR matrices require the Vulkan "
                         "backend.");
             return std::make_unique<VulkanSparseBsrMatrix>(
                 program, block_rows, block_cols, block_size, row_offsets,
                 column_indices, values);
           },
           py::keep_alive<0, 1>())
      .def("_vulkan_sparse_assembly_available",
           &Program::vulkan_sparse_assembly_available)
      .def("_cuda_sparse_assembly_available",
           &Program::cuda_sparse_assembly_available)
      .def("_vulkan_sparse_axpy",
           tracked_native_program_method(&Program::vulkan_sparse_axpy),
           py::arg("x"), py::arg("y"), py::arg("n"),
           py::arg("alpha"))
      .def("_vulkan_sparse_dot",
           tracked_native_program_method(&Program::vulkan_sparse_dot),
           py::arg("x"), py::arg("y"), py::arg("output"),
           py::arg("n"))
      .def("_vulkan_sparse_norm",
           tracked_native_program_method(&Program::vulkan_sparse_norm),
           py::arg("x"), py::arg("output"), py::arg("n"))
      .def("_vulkan_sparse_scalar_divide",
           tracked_native_program_method(
               &Program::vulkan_sparse_scalar_divide),
           py::arg("numerator"), py::arg("denominator"),
           py::arg("quotient"), py::arg("status"))
      .def("_vulkan_sparse_cg_update",
           tracked_native_program_method(&Program::vulkan_sparse_cg_update),
           py::arg("direction"), py::arg("applied_direction"),
           py::arg("alpha"), py::arg("solution"),
           py::arg("residual"), py::arg("n"))
      .def("_vulkan_sparse_cg_direction",
           tracked_native_program_method(
               &Program::vulkan_sparse_cg_direction),
           py::arg("residual"), py::arg("beta"),
           py::arg("direction"), py::arg("n"))
      .def("_vulkan_sparse_convergence",
           tracked_native_program_method(
               &Program::vulkan_sparse_convergence),
           py::arg("residual_squared"), py::arg("status"),
           py::arg("completed_iterations"),
           py::arg("tolerance_squared"), py::arg("iteration"))
      .def("_vulkan_sparse_algebra_clear_workspace",
           &Program::vulkan_sparse_algebra_clear_workspace)
      .def("_vulkan_sparse_algebra_workspace_bytes",
           &Program::vulkan_sparse_algebra_workspace_bytes)
      .def("make_sparse_matrix_from_ndarray",
           [](Program *program, SparseMatrix &sm, const Ndarray &ndarray) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                             !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             return make_sparse_matrix_from_ndarray(program, sm, ndarray);
           })
      .def("make_id_expr",
           [](Program *program, const std::string &name) {
             return Expr::make<IdExpression>(program->get_next_global_id(name));
           })
      .def(
          "create_ndarray",
          [&](Program *program, const DataType &dt,
              const std::vector<int> &shape, ExternalArrayLayout layout,
              bool zero_fill, DebugInfo dbg_info) -> Ndarray * {
            return program->create_ndarray(dt, shape, layout, zero_fill,
                                           dbg_info);
          },
          py::arg("dt"), py::arg("shape"),
          py::arg("layout") = ExternalArrayLayout::kNull,
          py::arg("zero_fill") = false, py::arg("dbg_info") = DebugInfo(),
          py::return_value_policy::reference)
      .def("delete_ndarray", &Program::delete_ndarray)
      .def(
          "create_argpack",
          [&](Program *program, const DataType &dt) -> ArgPack * {
            return program->create_argpack(dt);
          },
          py::arg("dt"), py::return_value_policy::reference)
      .def("delete_argpack", &Program::delete_argpack)
      .def("delete_texture", &Program::delete_texture)
      .def("_debug_argpack_resource_stats",
           &Program::debug_argpack_resource_stats)
      .def("_debug_argpack_resource_identity",
           &Program::debug_argpack_resource_identity)
      .def("_debug_ndarray_resource_stats",
           &Program::debug_ndarray_resource_stats)
      .def("_debug_ndarray_resource_identity",
           &Program::debug_ndarray_resource_identity)
      .def("_debug_texture_resource_stats",
           &Program::debug_texture_resource_stats)
      .def("_debug_texture_resource_identity",
           &Program::debug_texture_resource_identity)
      .def("_debug_dense_field_staging_stats",
           &Program::debug_dense_field_staging_stats)
      .def(
          "create_texture",
          [&](Program *program, BufferFormat fmt, const std::vector<int> &shape)
              -> Texture * { return program->create_texture(fmt, shape); },
          py::arg("fmt"), py::arg("shape") = py::tuple(),
          py::return_value_policy::reference)
      .def("get_ndarray_data_ptr_as_int",
           [](Program *program, Ndarray *ndarray) {
             return program->get_ndarray_data_ptr_as_int(ndarray);
           })
      .def("fill_float",
           [](Program *program, Ndarray *ndarray, float val) {
             program->fill_ndarray_fast_u32(ndarray,
                                            reinterpret_cast<uint32_t &>(val));
           })
      .def("fill_int",
           [](Program *program, Ndarray *ndarray, int32_t val) {
             program->fill_ndarray_fast_u32(ndarray,
                                            reinterpret_cast<int32_t &>(val));
           })
      .def("fill_uint",
           [](Program *program, Ndarray *ndarray, uint32_t val) {
             program->fill_ndarray_fast_u32(ndarray, val);
           })
      .def("copy_ndarray",
           [](Program *program, Ndarray *dst, Ndarray *src) {
             program->copy_ndarray_fast(dst, src);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("copy_ndarray_from_host",
           [](Program *program, Ndarray *dst, py::buffer src) {
             py::buffer_info info = src.request();
             const auto bytes = static_cast<std::size_t>(info.size) *
                                static_cast<std::size_t>(info.itemsize);
             py::gil_scoped_release release;
             program->copy_ndarray_from_host(dst, info.ptr, bytes);
           })
      .def("copy_ndarray_to_host",
           [](Program *program, Ndarray *src, py::buffer dst) {
             py::buffer_info info = dst.request();
             TI_ERROR_IF(info.readonly,
                         "copy_ndarray_to_host received a read-only buffer.");
             const auto bytes = static_cast<std::size_t>(info.size) *
                                static_cast<std::size_t>(info.itemsize);
             py::gil_scoped_release release;
             program->copy_ndarray_to_host(src, info.ptr, bytes);
           })
      .def("cuda_device_transform_available",
           &Program::cuda_device_transform_available)
      .def("cuda_toolkit_transform_available",
           &Program::cuda_toolkit_transform_available)
      .def("cuda_device_transform_affine_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_member_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("offset"), py::arg("stride"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_packed_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_transform_affine_packed_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("lane_count"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_transform_affine_dense_field",
           tracked_native_program_method(&Program::cuda_device_transform_affine_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_zero_dense_field",
           tracked_native_program_method(&Program::cuda_device_zero_dense_field), py::arg("dst"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_merge_available",
           &Program::cuda_device_add_merge_available)
      .def("cuda_device_add_merge_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_merge_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scaled_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_scaled_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scalar_ndarray_to_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_scalar_ndarray_to_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_merge_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_add_merge_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_merge_dense_field",
           tracked_native_program_method(&Program::cuda_device_add_merge_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scaled_dense_field",
           tracked_native_program_method(&Program::cuda_device_add_scaled_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_add_scalar_field_to_dense_field",
           tracked_native_program_method(&Program::cuda_device_add_scalar_field_to_dense_field),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("n"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_indexed_copy_available",
           &Program::cuda_device_indexed_copy_available)
      .def("cuda_device_indexed_copy_payload_available",
           &Program::cuda_device_indexed_copy_payload_available,
           py::arg("item_bytes"))
      .def("cuda_device_gather_ndarray", tracked_native_program_method(&Program::cuda_device_gather_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_gather_strided_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field_packed), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field_packed_indices_field",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field_packed_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_dense_field_indices_field",
           tracked_native_program_method(&Program::cuda_device_gather_dense_field_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_add_ndarray",
           tracked_native_program_method(&Program::cuda_device_gather_add_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_add_dense_field",
           tracked_native_program_method(&Program::cuda_device_gather_add_dense_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_gather_add_dense_field_indices_field",
           tracked_native_program_method(&Program::cuda_device_gather_add_dense_field_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_strided_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field_packed",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field_packed), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field_packed_indices_field",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field_packed_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_dense_field_indices_field",
            tracked_native_program_method(&Program::cuda_device_scatter_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_available",
            &Program::cuda_device_scatter_add_available)
       .def("cuda_device_scatter_add_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_add_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_member_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_add_member_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_strided_ndarray",
            tracked_native_program_method(&Program::cuda_device_scatter_add_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_dense_field",
            tracked_native_program_method(&Program::cuda_device_scatter_add_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_scatter_add_dense_field_indices_field",
            tracked_native_program_method(&Program::cuda_device_scatter_add_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_bucket_builder_available",
            &Program::cuda_device_bucket_builder_available)
       .def("cuda_device_bucket_builder_i32_ndarray",
            tracked_native_program_method(&Program::cuda_device_bucket_builder_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_bucket_builder_ndarray",
            tracked_native_program_method(&Program::cuda_device_bucket_builder_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_bucket_builder_dense_field",
            tracked_native_program_method(&Program::cuda_device_bucket_builder_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"), py::arg("n"),
            py::arg("num_bins"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_available",
            &Program::cuda_device_grouped_reduce_available)
       .def("cuda_device_grouped_reduce_atomic_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_dense_field",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_dense_field),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("n"), py::arg("num_groups"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_member_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_member_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("offset"), py::arg("stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_strided_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_strided_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_atomic_strided_keys_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_atomic_strided_keys_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("keys_offset"),
            py::arg("keys_stride"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_i32_atomic_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_i32_atomic_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_i32_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("value_type"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cuda_device_grouped_reduce_segmented_strided_keys_ndarray",
            tracked_native_program_method(&Program::cuda_device_grouped_reduce_segmented_strided_keys_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("offsets"), py::arg("scratch"), py::arg("cursor"),
            py::arg("value_type"), py::arg("keys_offset"),
            py::arg("keys_stride"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_available",
           &Program::cuda_device_radix_sort_available)
      .def("cuda_device_radix_sort_clear_workspace",
           &Program::cuda_device_radix_sort_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_workspace_bytes",
           &Program::cuda_device_radix_sort_workspace_bytes)
      .def("cuda_device_radix_sort_ndarray",
           tracked_native_program_method(&Program::cuda_device_radix_sort_ndarray),
           py::arg("keys"), py::arg("values"), py::arg("key_type"),
           py::arg("value_type"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type, int nan_policy) {
             return program->cuda_device_radix_sort_ndarray(
                 keys, nullptr, key_type, 0, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_dense_field",
           tracked_native_program_method(&Program::cuda_device_radix_sort_dense_field),
           py::arg("keys"), py::arg("values"), py::arg("key_type"),
           py::arg("value_type"), py::arg("n"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_radix_sort_keys_dense_field",
           [](Program *program, SNode *keys, int key_type, std::size_t n,
              int nan_policy) {
             return program->cuda_device_radix_sort_dense_field(
                 keys, nullptr, key_type, 0, n, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("n"),
           py::arg("nan_policy"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_available",
           &Program::cuda_cub_radix_sort_available)
      .def("cuda_cub_radix_sort_clear_workspace",
           &Program::cuda_cub_radix_sort_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_workspace_bytes",
           &Program::cuda_cub_radix_sort_workspace_bytes)
      .def("cuda_cub_radix_sort_ndarray",
           tracked_native_program_method(&Program::cuda_cub_radix_sort_ndarray), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("mode"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type, int mode,
              int nan_policy) {
             return program->cuda_cub_radix_sort_ndarray(keys, nullptr,
                                                         key_type, 0, mode,
                                                         nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("mode"),
           py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_dense_field",
           tracked_native_program_method(&Program::cuda_cub_radix_sort_dense_field), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("n"), py::arg("mode"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_radix_sort_keys_dense_field",
           [](Program *program, SNode *keys, int key_type, std::size_t n,
              int mode, int nan_policy) {
             return program->cuda_cub_radix_sort_dense_field(
                 keys, nullptr, key_type, 0, n, mode, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("n"),
           py::arg("mode"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_available", &Program::cpu_stable_sort_available)
      .def("cpu_stable_sort_ndarray", tracked_native_program_method(&Program::cpu_stable_sort_ndarray),
           py::arg("keys"), py::arg("values"), py::arg("key_type"),
           py::arg("value_type"), py::arg("descending"),
           py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type, bool descending,
              int nan_policy) {
             return program->cpu_stable_sort_ndarray(
                 keys, nullptr, key_type, 0, descending, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("descending"),
           py::arg("nan_policy"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_dense_field",
           tracked_native_program_method(&Program::cpu_stable_sort_dense_field), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("n"), py::arg("descending"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_stable_sort_keys_dense_field",
           [](Program *program, SNode *keys, int key_type, std::size_t n,
              bool descending, int nan_policy) {
             return program->cpu_stable_sort_dense_field(
                 keys, nullptr, key_type, 0, n, descending, nan_policy);
           },
           py::arg("keys"), py::arg("key_type"), py::arg("n"),
           py::arg("descending"), py::arg("nan_policy"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_scan_available", &Program::cuda_device_scan_available)
      .def("cuda_device_scan_clear_workspace",
           &Program::cuda_device_scan_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_scan_workspace_bytes",
           &Program::cuda_device_scan_workspace_bytes)
      .def("cuda_device_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_ndarray),
           py::arg("data"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_ndarray),
           py::arg("data"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_inclusive_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_scan_available", &Program::cuda_cub_scan_available)
      .def("cuda_cub_scan_clear_workspace",
           &Program::cuda_cub_scan_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_scan_workspace_bytes",
           &Program::cuda_cub_scan_workspace_bytes)
      .def("cuda_cub_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_member_ndarray), py::arg("data"),
           py::arg("value_type"), py::arg("offset"), py::arg("stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_cub_inclusive_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::cuda_cub_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_available",
           &Program::cuda_device_compact_available)
      .def("cuda_device_compact_ndarray",
           tracked_native_program_method(&Program::cuda_device_compact_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_dense_field",
           tracked_native_program_method(
               &Program::cuda_device_compact_dense_field),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_clear_workspace",
           &Program::cuda_device_compact_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_compact_workspace_bytes",
           &Program::cuda_device_compact_workspace_bytes)
      .def("cuda_cub_select_available", &Program::cuda_cub_select_available)
      .def("cuda_cub_select_clear_workspace",
           &Program::cuda_cub_select_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_select_workspace_bytes",
           &Program::cuda_cub_select_workspace_bytes)
      .def("cuda_cub_select_ndarray", tracked_native_program_method(&Program::cuda_cub_select_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_select_dense_field",
           tracked_native_program_method(&Program::cuda_cub_select_dense_field), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_select_i32_ndarray",
           tracked_native_program_method(&Program::cuda_cub_select_i32_ndarray), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_available",
           &Program::cuda_device_histogram_available)
      .def("cuda_device_histogram_ndarray",
           tracked_native_program_method(&Program::cuda_device_histogram_ndarray),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_dense_field",
           tracked_native_program_method(&Program::cuda_device_histogram_dense_field),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type"), py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_clear_workspace",
           &Program::cuda_device_histogram_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_histogram_workspace_bytes",
           &Program::cuda_device_histogram_workspace_bytes)
      .def("cuda_cub_histogram_available",
           &Program::cuda_cub_histogram_available)
      .def("cuda_cub_histogram_clear_workspace",
           &Program::cuda_cub_histogram_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_histogram_workspace_bytes",
           &Program::cuda_cub_histogram_workspace_bytes)
      .def("cuda_cub_histogram_ndarray",
           tracked_native_program_method(&Program::cuda_cub_histogram_ndarray), py::arg("values"),
           py::arg("bins"), py::arg("value_type"), py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_histogram_i32_ndarray",
           tracked_native_program_method(&Program::cuda_cub_histogram_i32_ndarray), py::arg("values"),
           py::arg("bins"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_histogram_dense_field",
           tracked_native_program_method(&Program::cuda_cub_histogram_dense_field), py::arg("values"),
           py::arg("bins"), py::arg("value_type"), py::arg("bin_type"),
           py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_available",
           &Program::cuda_device_reduce_available)
      .def("cuda_device_reduce_clear_workspace",
           &Program::cuda_device_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_workspace_bytes",
           &Program::cuda_device_reduce_workspace_bytes)
      .def("cuda_device_reduce_ndarray",
           tracked_native_program_method(&Program::cuda_device_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_member_ndarray",
           tracked_native_program_method(&Program::cuda_device_reduce_member_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("offset"), py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_reduce_strided_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_dense_field",
           tracked_native_program_method(&Program::cuda_device_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_reduce_dense_field_packed",
           tracked_native_program_method(&Program::cuda_device_reduce_dense_field_packed),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_available", &Program::cuda_cub_reduce_available)
      .def("cuda_cub_reduce_clear_workspace",
           &Program::cuda_cub_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_workspace_bytes",
           &Program::cuda_cub_reduce_workspace_bytes)
      .def("cuda_device_check_count_available",
           &Program::cuda_device_check_count_available)
      .def("cuda_device_check_count_clear_workspace",
           &Program::cuda_device_check_count_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_check_count_workspace_bytes",
           &Program::cuda_device_check_count_workspace_bytes)
      .def("cuda_device_check_count_ndarray",
           tracked_native_program_method(&Program::cuda_device_check_count_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("check_op"),
           py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_check_count_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_check_count_dense_field",
           tracked_native_program_method(&Program::cuda_device_check_count_dense_field), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_available",
           &Program::cuda_device_metric_reduce_available)
      .def("cuda_device_metric_reduce_value_type_available",
           &Program::cuda_device_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("cuda_device_metric_reduce_clear_workspace",
           &Program::cuda_device_metric_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_workspace_bytes",
           &Program::cuda_device_metric_reduce_workspace_bytes)
      .def("cuda_device_metric_reduce_ndarray",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_dense_field",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_device_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::cuda_device_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_available",
           &Program::cuda_cub_check_count_available)
      .def("cuda_cub_check_count_clear_workspace",
           &Program::cuda_cub_check_count_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_workspace_bytes",
           &Program::cuda_cub_check_count_workspace_bytes)
      .def("cuda_cub_check_count_ndarray",
           tracked_native_program_method(&Program::cuda_cub_check_count_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("check_op"),
           py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_check_count_dense_field",
           tracked_native_program_method(&Program::cuda_cub_check_count_dense_field), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_available",
           &Program::cuda_cub_metric_reduce_available)
      .def("cuda_cub_metric_reduce_value_type_available",
           &Program::cuda_cub_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("cuda_cub_metric_reduce_clear_workspace",
           &Program::cuda_cub_metric_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_workspace_bytes",
           &Program::cuda_cub_metric_reduce_workspace_bytes)
      .def("cuda_cub_metric_reduce_ndarray",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_dense_field",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_ndarray", tracked_native_program_method(&Program::cuda_cub_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_member_ndarray",
           tracked_native_program_method(&Program::cuda_cub_reduce_member_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cuda_cub_reduce_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_dense_field", tracked_native_program_method(&Program::cuda_cub_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_cub_reduce_dense_field_packed",
           tracked_native_program_method(&Program::cuda_cub_reduce_dense_field_packed), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_scan_available", &Program::cpu_scan_available)
      .def("cpu_scan_workspace_bytes", &Program::cpu_scan_workspace_bytes)
      .def("cpu_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_scan_ndarray), py::arg("data"),
           py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_ndarray), py::arg("data"),
           py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_scan_member_ndarray), py::arg("data"),
           py::arg("value_type"), py::arg("offset"), py::arg("stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::cpu_inclusive_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::cpu_inclusive_scan_dense_field_packed), py::arg("data"),
           py::arg("value_type"), py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::cpu_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_available", &Program::cpu_compact_available)
      .def("cpu_compact_workspace_bytes", &Program::cpu_compact_workspace_bytes)
      .def("fill_dense_field", &Program::fill_dense_field, py::arg("dst"),
           py::arg("value_type"), py::arg("value_bits"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("fill_dense_field_packed", &Program::fill_dense_field_packed,
           py::arg("dst"), py::arg("value_type"), py::arg("value_bits"),
           py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("transform_affine_dense_field_packed",
           &Program::transform_affine_dense_field_packed, py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("copy_dense_field", &Program::copy_dense_field, py::arg("dst"),
           py::arg("src"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("copy_dense_field_packed", &Program::copy_dense_field_packed,
           py::arg("dst"), py::arg("src"), py::arg("value_type"),
           py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "copy_dense_field_from_host",
          [](Program &program, SNode *dst, py::buffer src, int value_type,
             std::size_t n) {
            py::buffer_info info = src.request();
            const auto src_ptr =
                reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t src_bytes = checked_buffer_nbytes(
                info,
                "Native dense field host copy received an invalid source "
                "buffer.",
                "Native dense field host copy received an oversized source "
                "buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_from_host(dst, src_ptr, src_bytes,
                                               value_type, n);
          },
          py::arg("dst"), py::arg("src"), py::arg("value_type"), py::arg("n"))
      .def(
          "copy_dense_field_packed_from_host",
          [](Program &program, SNode *dst, py::buffer src, int value_type,
             std::size_t n, int lane_count) {
            py::buffer_info info = src.request();
            const auto src_ptr =
                reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t src_bytes = checked_buffer_nbytes(
                info,
                "Native packed dense field host copy received an invalid "
                "source buffer.",
                "Native packed dense field host copy received an oversized "
                "source buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_packed_from_host(
                dst, src_ptr, src_bytes, value_type, n, lane_count);
          },
          py::arg("dst"), py::arg("src"), py::arg("value_type"), py::arg("n"),
          py::arg("lane_count"))
      .def(
          "copy_dense_field_to_host",
          [](Program &program, SNode *src, py::buffer dst, int value_type,
             std::size_t n) {
            py::buffer_info info = dst.request();
            const auto dst_ptr = reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t dst_bytes = checked_buffer_nbytes(
                info,
                "Native dense field host readback received an invalid "
                "destination buffer.",
                "Native dense field host readback received an oversized "
                "destination buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_to_host(src, dst_ptr, dst_bytes,
                                             value_type, n);
          },
          py::arg("src"), py::arg("dst"), py::arg("value_type"), py::arg("n"))
      .def(
          "copy_dense_field_packed_to_host",
          [](Program &program, SNode *src, py::buffer dst, int value_type,
             std::size_t n, int lane_count) {
            py::buffer_info info = dst.request();
            const auto dst_ptr = reinterpret_cast<std::uintptr_t>(info.ptr);
            const std::size_t dst_bytes = checked_buffer_nbytes(
                info,
                "Native packed dense field host readback received an invalid "
                "destination buffer.",
                "Native packed dense field host readback received an "
                "oversized destination buffer.");
            py::gil_scoped_release release;
            program.copy_dense_field_packed_to_host(
                src, dst_ptr, dst_bytes, value_type, n, lane_count);
          },
          py::arg("src"), py::arg("dst"), py::arg("value_type"),
          py::arg("n"), py::arg("lane_count"))
      .def("add_merge_dense_field_packed",
           &Program::add_merge_dense_field_packed, py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("scatter_add_dense_field_packed",
           &Program::scatter_add_dense_field_packed, py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("scatter_add_dense_field_packed_indices_field",
           &Program::scatter_add_dense_field_packed_indices_field,
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_ndarray", tracked_native_program_method(&Program::cpu_compact_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_dense_field", tracked_native_program_method(&Program::cpu_compact_dense_field),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_compact_i32_ndarray",
           tracked_native_program_method(&Program::cpu_compact_i32_ndarray), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_histogram_available", &Program::cpu_histogram_available)
      .def("cpu_histogram_workspace_bytes",
           &Program::cpu_histogram_workspace_bytes)
      .def("cpu_histogram_ndarray", tracked_native_program_method(&Program::cpu_histogram_ndarray),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_histogram_i32_ndarray",
           tracked_native_program_method(&Program::cpu_histogram_i32_ndarray), py::arg("values"),
           py::arg("bins"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_histogram_dense_field", tracked_native_program_method(&Program::cpu_histogram_dense_field),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type"), py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_available", &Program::cpu_reduce_available)
      .def("cpu_reduce_workspace_bytes", &Program::cpu_reduce_workspace_bytes)
      .def("cpu_check_count_available", &Program::cpu_check_count_available)
      .def("cpu_check_count_workspace_bytes",
           &Program::cpu_check_count_workspace_bytes)
      .def("cpu_check_count_ndarray", tracked_native_program_method(&Program::cpu_check_count_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_check_count_strided_ndarray",
           tracked_native_program_method(&Program::cpu_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_check_count_dense_field", tracked_native_program_method(&Program::cpu_check_count_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_available", &Program::cpu_metric_reduce_available)
      .def("cpu_metric_reduce_value_type_available",
           &Program::cpu_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("cpu_metric_reduce_workspace_bytes",
           &Program::cpu_metric_reduce_workspace_bytes)
      .def("cpu_metric_reduce_ndarray", tracked_native_program_method(&Program::cpu_metric_reduce_ndarray),
           py::arg("values"), py::arg("other"), py::arg("output"),
           py::arg("value_type"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::cpu_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_dense_field",
           tracked_native_program_method(&Program::cpu_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::cpu_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_ndarray", tracked_native_program_method(&Program::cpu_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_member_ndarray",
           tracked_native_program_method(&Program::cpu_reduce_member_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_strided_ndarray", tracked_native_program_method(&Program::cpu_reduce_strided_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_dense_field", tracked_native_program_method(&Program::cpu_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_reduce_dense_field_packed",
           tracked_native_program_method(&Program::cpu_reduce_dense_field_packed), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_available", &Program::cpu_transform_available)
      .def("cpu_transform_workspace_bytes",
           &Program::cpu_transform_workspace_bytes)
      .def("cpu_transform_affine_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_member_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_member_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_strided_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_packed_strided_ndarray",
           tracked_native_program_method(&Program::cpu_transform_affine_packed_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("lane_count"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_transform_affine_dense_field",
           tracked_native_program_method(&Program::cpu_transform_affine_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_merge_available", &Program::cpu_add_merge_available)
      .def("cpu_add_merge_ndarray", tracked_native_program_method(&Program::cpu_add_merge_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scaled_ndarray", tracked_native_program_method(&Program::cpu_add_scaled_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scalar_ndarray_to_ndarray",
           tracked_native_program_method(&Program::cpu_add_scalar_ndarray_to_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_merge_strided_ndarray",
           tracked_native_program_method(&Program::cpu_add_merge_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_merge_dense_field",
           tracked_native_program_method(&Program::cpu_add_merge_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scaled_dense_field",
           tracked_native_program_method(&Program::cpu_add_scaled_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_add_scalar_field_to_dense_field",
           tracked_native_program_method(&Program::cpu_add_scalar_field_to_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_indexed_copy_available", &Program::cpu_indexed_copy_available)
      .def("cpu_indexed_copy_workspace_bytes",
           &Program::cpu_indexed_copy_workspace_bytes)
      .def("cpu_gather_ndarray", tracked_native_program_method(&Program::cpu_gather_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_strided_ndarray",
           tracked_native_program_method(&Program::cpu_gather_strided_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field", tracked_native_program_method(&Program::cpu_gather_dense_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field_packed",
           tracked_native_program_method(&Program::cpu_gather_dense_field_packed), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field_packed_indices_field",
           tracked_native_program_method(&Program::cpu_gather_dense_field_packed_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_dense_field_indices_field",
           tracked_native_program_method(&Program::cpu_gather_dense_field_indices_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_add_ndarray", tracked_native_program_method(&Program::cpu_gather_add_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_add_dense_field",
           tracked_native_program_method(&Program::cpu_gather_add_dense_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_gather_add_dense_field_indices_field",
           tracked_native_program_method(&Program::cpu_gather_add_dense_field_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_ndarray", tracked_native_program_method(&Program::cpu_scatter_ndarray),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_strided_ndarray",
            tracked_native_program_method(&Program::cpu_scatter_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field", tracked_native_program_method(&Program::cpu_scatter_dense_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field_packed",
            tracked_native_program_method(&Program::cpu_scatter_dense_field_packed), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field_packed_indices_field",
            tracked_native_program_method(&Program::cpu_scatter_dense_field_packed_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_dense_field_indices_field",
            tracked_native_program_method(&Program::cpu_scatter_dense_field_indices_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_available", &Program::cpu_scatter_add_available)
       .def("cpu_scatter_add_workspace_bytes",
            &Program::cpu_scatter_add_workspace_bytes)
       .def("cpu_scatter_add_clear_workspace",
            &Program::cpu_scatter_add_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_ndarray", tracked_native_program_method(&Program::cpu_scatter_add_ndarray),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_member_ndarray",
            tracked_native_program_method(&Program::cpu_scatter_add_member_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_strided_ndarray",
            tracked_native_program_method(&Program::cpu_scatter_add_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_dense_field",
            tracked_native_program_method(&Program::cpu_scatter_add_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_scatter_add_dense_field_indices_field",
            tracked_native_program_method(&Program::cpu_scatter_add_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_bucket_builder_available",
            &Program::cpu_bucket_builder_available)
       .def("cpu_bucket_builder_workspace_bytes",
            &Program::cpu_bucket_builder_workspace_bytes)
       .def("cpu_bucket_builder_i32_ndarray",
            tracked_native_program_method(&Program::cpu_bucket_builder_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_bucket_builder_ndarray",
            tracked_native_program_method(&Program::cpu_bucket_builder_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("value_type"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_bucket_builder_dense_field",
            tracked_native_program_method(&Program::cpu_bucket_builder_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("value_type"), py::arg("n"), py::arg("num_bins"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_available",
            &Program::cpu_grouped_reduce_available)
       .def("cpu_grouped_reduce_ndarray", tracked_native_program_method(&Program::cpu_grouped_reduce_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_dense_field",
            tracked_native_program_method(&Program::cpu_grouped_reduce_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("n"), py::arg("num_groups"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_member_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_member_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_strided_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_strided_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("values_offset"), py::arg("values_stride"),
            py::arg("output_offset"), py::arg("output_stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_strided_keys_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_strided_keys_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("keys_offset"), py::arg("keys_stride"),
            py::arg("values_offset"), py::arg("values_stride"),
            py::arg("output_offset"), py::arg("output_stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_workspace_bytes",
            &Program::cpu_grouped_reduce_workspace_bytes)
       .def("cpu_grouped_reduce_clear_workspace",
            &Program::cpu_grouped_reduce_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("cpu_grouped_reduce_i32_ndarray",
            tracked_native_program_method(&Program::cpu_grouped_reduce_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
      .def("vulkan_radix_sort_available",
           &Program::vulkan_radix_sort_available)
      .def("vulkan_radix_sort_clear_workspace",
           &Program::vulkan_radix_sort_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_radix_sort_workspace_bytes",
           &Program::vulkan_radix_sort_workspace_bytes)
      .def("vulkan_radix_sort_cpu_profile_clear",
           &Program::vulkan_radix_sort_cpu_profile_clear)
      .def("vulkan_radix_sort_cpu_profile_report",
           &Program::vulkan_radix_sort_cpu_profile_report)
      .def("vulkan_scan_available", &Program::vulkan_scan_available)
      .def("vulkan_scan_value_type_available",
           &Program::vulkan_scan_value_type_available, py::arg("value_type"))
      .def("vulkan_scan_clear_workspace",
           &Program::vulkan_scan_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_scan_workspace_bytes",
           &Program::vulkan_scan_workspace_bytes)
      .def("vulkan_inclusive_scan_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_ndarray), py::arg("data"),
           py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_scan_member_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_member_ndarray), py::arg("data"),
           py::arg("value_type"), py::arg("offset"), py::arg("stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_member_ndarray",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_member_ndarray),
           py::arg("data"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_scan_dense_field",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_dense_field), py::arg("data"),
           py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_dense_field",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_dense_field),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_scan_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_inclusive_scan_dense_field_packed), py::arg("data"),
           py::arg("value_type"), py::arg("n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_inclusive_reverse_scan_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_inclusive_reverse_scan_dense_field_packed),
           py::arg("data"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_available", &Program::vulkan_compact_available)
      .def("vulkan_compact_clear_workspace",
           &Program::vulkan_compact_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_workspace_bytes",
           &Program::vulkan_compact_workspace_bytes)
      .def("vulkan_compact_ndarray", tracked_native_program_method(&Program::vulkan_compact_ndarray),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_dense_field", tracked_native_program_method(&Program::vulkan_compact_dense_field),
           py::arg("values"), py::arg("flags"), py::arg("output"),
           py::arg("count"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_compact_i32_ndarray",
           tracked_native_program_method(&Program::vulkan_compact_i32_ndarray), py::arg("values"),
           py::arg("flags"), py::arg("output"), py::arg("count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_available",
           &Program::vulkan_histogram_available)
      .def("vulkan_histogram_clear_workspace",
           &Program::vulkan_histogram_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_workspace_bytes",
           &Program::vulkan_histogram_workspace_bytes)
      .def("vulkan_histogram_value_type_available",
           &Program::vulkan_histogram_value_type_available,
           py::arg("value_type"), py::arg("bin_type") = 0)
      .def("vulkan_histogram_ndarray", tracked_native_program_method(&Program::vulkan_histogram_ndarray),
           py::arg("values"), py::arg("bins"), py::arg("value_type"),
           py::arg("bin_type") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_i32_ndarray",
           tracked_native_program_method(&Program::vulkan_histogram_i32_ndarray), py::arg("values"),
           py::arg("bins"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_histogram_dense_field",
           tracked_native_program_method(&Program::vulkan_histogram_dense_field), py::arg("values"),
           py::arg("bins"), py::arg("value_type"), py::arg("bin_type"),
           py::arg("n"), py::arg("num_bins"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_available", &Program::vulkan_reduce_available)
      .def("vulkan_reduce_clear_workspace",
           &Program::vulkan_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_workspace_bytes",
           &Program::vulkan_reduce_workspace_bytes)
      .def("vulkan_check_count_available",
           &Program::vulkan_check_count_available)
      .def("vulkan_check_count_clear_workspace",
           &Program::vulkan_check_count_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_check_count_workspace_bytes",
           &Program::vulkan_check_count_workspace_bytes)
      .def("vulkan_check_count_value_type_available",
           &Program::vulkan_check_count_value_type_available,
           py::arg("value_type"))
      .def("vulkan_check_count_ndarray", tracked_native_program_method(&Program::vulkan_check_count_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_check_count_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_check_count_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("check_op"), py::arg("lower"),
           py::arg("upper"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_check_count_dense_field",
           tracked_native_program_method(&Program::vulkan_check_count_dense_field), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("check_op"), py::arg("lower"), py::arg("upper"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_available",
           &Program::vulkan_metric_reduce_available)
      .def("vulkan_metric_reduce_clear_workspace",
           &Program::vulkan_metric_reduce_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_workspace_bytes",
           &Program::vulkan_metric_reduce_workspace_bytes)
      .def("vulkan_metric_reduce_value_type_available",
           &Program::vulkan_metric_reduce_value_type_available,
           py::arg("value_type"))
      .def("vulkan_metric_reduce_ndarray",
           tracked_native_program_method(&Program::vulkan_metric_reduce_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_metric_reduce_strided_ndarray), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("other_offset"), py::arg("other_stride"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_dense_field",
           tracked_native_program_method(&Program::vulkan_metric_reduce_dense_field), py::arg("values"),
           py::arg("other"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("metric_op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_metric_reduce_dense_field_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_metric_reduce_dense_field_strided_ndarray),
           py::arg("field"), py::arg("array"), py::arg("output"),
           py::arg("value_type"), py::arg("n"), py::arg("array_offset"),
           py::arg("array_stride"), py::arg("field_is_values"),
           py::arg("metric_op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_value_type_available",
           &Program::vulkan_reduce_value_type_available,
           py::arg("value_type"))
      .def("vulkan_reduce_ndarray", tracked_native_program_method(&Program::vulkan_reduce_ndarray),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("op"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_member_ndarray",
           tracked_native_program_method(&Program::vulkan_reduce_member_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_reduce_strided_ndarray), py::arg("values"),
           py::arg("output"), py::arg("value_type"),
           py::arg("values_offset"), py::arg("values_stride"),
           py::arg("output_offset"), py::arg("output_stride"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_dense_field", tracked_native_program_method(&Program::vulkan_reduce_dense_field),
           py::arg("values"), py::arg("output"), py::arg("value_type"),
           py::arg("n"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_reduce_dense_field_packed), py::arg("values"),
           py::arg("output"), py::arg("value_type"), py::arg("n"),
           py::arg("lane_count"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_reduce_i32_ndarray",
           tracked_native_program_method(&Program::vulkan_reduce_i32_ndarray), py::arg("values"),
           py::arg("output"), py::arg("op"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_available", &Program::vulkan_transform_available)
      .def("vulkan_transform_value_type_available",
           &Program::vulkan_transform_value_type_available,
           py::arg("value_type"))
      .def("vulkan_transform_clear_workspace",
           &Program::vulkan_transform_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_workspace_bytes",
           &Program::vulkan_transform_workspace_bytes)
      .def("vulkan_transform_affine_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_ndarray_trusted",
           tracked_native_program_method(&Program::vulkan_transform_affine_ndarray_trusted), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_indexed_affine_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_indexed_affine_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_member_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_member_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("offset"),
           py::arg("stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_strided_ndarray_trusted",
           tracked_native_program_method(&Program::vulkan_transform_affine_strided_ndarray_trusted),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"), py::arg("scale"),
           py::arg("bias"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_packed_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_transform_affine_packed_strided_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("lane_count"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_dense_field",
           tracked_native_program_method(&Program::vulkan_transform_affine_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_transform_affine_dense_field_trusted",
           tracked_native_program_method(&Program::vulkan_transform_affine_dense_field_trusted),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::arg("n"), py::arg("scale"), py::arg("bias"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_zero_dense_field", tracked_native_program_method(&Program::vulkan_zero_dense_field),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_zero_dense_fields", tracked_native_program_method(&Program::vulkan_zero_dense_fields),
           py::arg("dsts"), py::arg("value_types"), py::arg("ns"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_available", &Program::vulkan_add_merge_available)
      .def("vulkan_add_merge_value_type_available",
           &Program::vulkan_add_merge_value_type_available,
           py::arg("value_type"))
      .def("vulkan_add_merge_clear_workspace",
           &Program::vulkan_add_merge_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_workspace_bytes",
           &Program::vulkan_add_merge_workspace_bytes)
      .def("vulkan_add_merge_ndarray", tracked_native_program_method(&Program::vulkan_add_merge_ndarray),
           py::arg("src"), py::arg("dst"), py::arg("value_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_add_merge_strided_ndarray), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("src_offset"),
           py::arg("src_stride"), py::arg("dst_offset"),
           py::arg("dst_stride"), py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_merge_dense_field",
           tracked_native_program_method(&Program::vulkan_add_merge_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_add_scalar_field_to_dense_field",
           tracked_native_program_method(&Program::vulkan_add_scalar_field_to_dense_field), py::arg("src"),
           py::arg("dst"), py::arg("value_type"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_indexed_copy_available",
           &Program::vulkan_indexed_copy_available)
      .def("vulkan_indexed_copy_clear_workspace",
           &Program::vulkan_indexed_copy_clear_workspace,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_indexed_copy_workspace_bytes",
           &Program::vulkan_indexed_copy_workspace_bytes)
      .def("vulkan_gather_ndarray", tracked_native_program_method(&Program::vulkan_gather_ndarray),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_strided_ndarray",
           tracked_native_program_method(&Program::vulkan_gather_strided_ndarray), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
           py::arg("src_offset"), py::arg("src_stride"),
           py::arg("dst_offset"), py::arg("dst_stride"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field", tracked_native_program_method(&Program::vulkan_gather_dense_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field_packed",
           tracked_native_program_method(&Program::vulkan_gather_dense_field_packed), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field_packed_indices_field",
           tracked_native_program_method(&Program::vulkan_gather_dense_field_packed_indices_field),
           py::arg("src"), py::arg("indices"), py::arg("dst"),
           py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
           py::arg("dst_n"), py::arg("lane_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_gather_dense_field_indices_field",
           tracked_native_program_method(&Program::vulkan_gather_dense_field_indices_field), py::arg("src"),
           py::arg("indices"), py::arg("dst"), py::arg("value_type"),
           py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
           py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_ndarray", tracked_native_program_method(&Program::vulkan_scatter_ndarray),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_strided_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("item_bytes"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field_packed",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field_packed), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field_packed_indices_field",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field_packed_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::arg("lane_count"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_dense_field_indices_field",
            tracked_native_program_method(&Program::vulkan_scatter_dense_field_indices_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("indices_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_available",
            &Program::vulkan_scatter_add_available)
       .def("vulkan_scatter_add_value_type_available",
            &Program::vulkan_scatter_add_value_type_available,
            py::arg("value_type"))
       .def("vulkan_scatter_add_clear_workspace",
            &Program::vulkan_scatter_add_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_workspace_bytes",
            &Program::vulkan_scatter_add_workspace_bytes)
       .def("vulkan_scatter_add_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_add_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_member_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_add_member_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("offset"), py::arg("stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_strided_ndarray",
            tracked_native_program_method(&Program::vulkan_scatter_add_strided_ndarray), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_offset"), py::arg("src_stride"),
            py::arg("dst_offset"), py::arg("dst_stride"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_dense_field",
            tracked_native_program_method(&Program::vulkan_scatter_add_dense_field), py::arg("src"),
            py::arg("indices"), py::arg("dst"), py::arg("value_type"),
            py::arg("src_n"), py::arg("dst_n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_scatter_add_dense_field_indices_field",
            tracked_native_program_method(&Program::vulkan_scatter_add_dense_field_indices_field),
            py::arg("src"), py::arg("indices"), py::arg("dst"),
            py::arg("value_type"), py::arg("src_n"), py::arg("indices_n"),
            py::arg("dst_n"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_available",
            &Program::vulkan_bucket_builder_available)
       .def("vulkan_bucket_builder_clear_workspace",
            &Program::vulkan_bucket_builder_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_workspace_bytes",
            &Program::vulkan_bucket_builder_workspace_bytes)
       .def("vulkan_bucket_builder_value_type_available",
            &Program::vulkan_bucket_builder_value_type_available,
            py::arg("value_type"))
       .def("vulkan_bucket_builder_i32_ndarray",
            tracked_native_program_method(&Program::vulkan_bucket_builder_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_ndarray",
            tracked_native_program_method(&Program::vulkan_bucket_builder_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_bucket_builder_dense_field",
            tracked_native_program_method(&Program::vulkan_bucket_builder_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("offsets"), py::arg("output"),
            py::arg("cursor"), py::arg("value_type"), py::arg("n"),
            py::arg("num_bins"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_available",
            &Program::vulkan_grouped_reduce_available)
       .def("vulkan_grouped_reduce_value_type_available",
            &Program::vulkan_grouped_reduce_value_type_available,
            py::arg("value_type"))
       .def("vulkan_grouped_reduce_atomic_value_type_available",
            &Program::vulkan_grouped_reduce_atomic_value_type_available,
            py::arg("value_type"))
       .def("vulkan_grouped_reduce_atomic_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("value_type"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_dense_field",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_dense_field),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("n"), py::arg("num_groups"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_member_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_member_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("offset"), py::arg("stride"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_strided_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_strided_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_atomic_strided_keys_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_atomic_strided_keys_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("value_type"), py::arg("keys_offset"),
            py::arg("keys_stride"), py::arg("values_offset"),
            py::arg("values_stride"), py::arg("output_offset"),
            py::arg("output_stride"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_clear_workspace",
            &Program::vulkan_grouped_reduce_clear_workspace,
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_workspace_bytes",
            &Program::vulkan_grouped_reduce_workspace_bytes)
       .def("vulkan_grouped_reduce_i32_atomic_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_i32_atomic_ndarray),
            py::arg("keys"), py::arg("values"), py::arg("output"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_i32_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_i32_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("op"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_grouped_reduce_ndarray",
            tracked_native_program_method(&Program::vulkan_grouped_reduce_ndarray), py::arg("keys"),
            py::arg("values"), py::arg("output"), py::arg("offsets"),
            py::arg("scratch"), py::arg("cursor"), py::arg("value_type"),
            py::arg("op"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_radix_sort_u32_dense_field",
            tracked_native_program_method(&Program::vulkan_radix_sort_u32_dense_field), py::arg("keys"),
            py::arg("values"), py::arg("key_type"), py::arg("value_type"),
            py::arg("n"), py::call_guard<py::gil_scoped_release>())
       .def("vulkan_radix_sort_u32_keys_dense_field",
            [](Program *program, SNode *keys, int key_type, std::size_t n) {
              return program->vulkan_radix_sort_u32_dense_field(
                  keys, nullptr, key_type, 0, n);
            },
            py::arg("keys"), py::arg("key_type"), py::arg("n"),
            py::call_guard<py::gil_scoped_release>())
       .def("vulkan_radix_sort_u32_ndarray",
           tracked_native_program_method(&Program::vulkan_radix_sort_u32_ndarray), py::arg("keys"),
           py::arg("values"), py::arg("key_type"), py::arg("value_type"),
           py::arg("key_offset") = 0, py::arg("value_offset") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("vulkan_radix_sort_u32_keys_ndarray",
           [](Program *program, Ndarray *keys, int key_type) {
             return program->vulkan_radix_sort_u32_ndarray(keys, nullptr,
                                                           key_type, 0);
           },
           py::arg("keys"), py::arg("key_type"),
           py::call_guard<py::gil_scoped_release>())
      .def("get_graphics_device",
           [](Program *program) { return program->get_graphics_device(); })
      .def("compile_kernel", &Program::compile_kernel,
           py::return_value_policy::reference)
      .def(
          "compile_kernels",
          [](Program *program, const CompileConfig &cfg,
             const std::vector<Kernel *> &kernels) {
            std::vector<const Kernel *> ks(kernels.begin(), kernels.end());
            program->compile_kernels(cfg, ks);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("launch_kernel", &Program::launch_kernel)
      .def("compile_and_launch_kernel", &Program::compile_and_launch_kernel,
           py::call_guard<py::gil_scoped_release>())
      .def("get_device_caps", &Program::get_device_caps);

  py::class_<AotModuleBuilder>(m, "AotModuleBuilder")
      .def("add_field", &AotModuleBuilder::add_field)
      .def("add", &AotModuleBuilder::add)
      .def("add_kernel_template", &AotModuleBuilder::add_kernel_template)
      .def("add_graph", &AotModuleBuilder::add_graph)
      .def("dump", &AotModuleBuilder::dump);

  py::class_<Axis>(m, "Axis").def(py::init<int>());
  py::class_<SNode>(m, "SNode")
      .def(py::init<>())
      .def_readwrite("parent", &SNode::parent)
      .def_readonly("type", &SNode::type)
      .def_readonly("id", &SNode::id)
      .def("get_snode_tree_id", &SNode::get_snode_tree_id)
      .def_readonly("offset", &SNode::index_offsets)
      .def("dense",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::dense),
           py::return_value_policy::reference)
      .def("pointer",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::pointer),
           py::return_value_policy::reference)
      // C-1 (2026-05): pointer + per-SNode pool capacity hint。后端在
      // SNodeTree finalize 时读取 vk_max_active_hint 决议池容量，绕过 worst-
      // case 与全局 fraction（详见 SNode_Vulkan_规划.md §11.1）。
      .def(
          "pointer_with_hint",
          [](SNode &self, const std::vector<Axis> &axes,
             const std::vector<int> &sizes, int64_t vk_max_active,
             const DebugInfo &dbg_info) -> SNode & {
            SNode &child = self.pointer(axes, sizes, dbg_info);
            child.vk_max_active_hint = vk_max_active;
            return child;
          },
          py::return_value_policy::reference)
      .def("hash",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::hash),
           py::return_value_policy::reference)
      .def(
          "hash_with_capacity",
          [](SNode &self, const std::vector<Axis> &axes,
             const std::vector<int> &sizes, int64_t table_capacity,
             const DebugInfo &dbg_info) -> SNode & {
            SNode &child = self.hash(axes, sizes, dbg_info);
            child.vk_max_active_hint = table_capacity;
            return child;
          },
          py::return_value_policy::reference)
      .def(
          "hash_with_capacity_and_active_hint",
          [](SNode &self, const std::vector<Axis> &axes,
             const std::vector<int> &sizes, int64_t table_capacity,
             int64_t expected_active, const DebugInfo &dbg_info) -> SNode & {
            SNode &child = self.hash(axes, sizes, dbg_info);
            child.vk_max_active_hint = table_capacity;
            child.hash_expected_active_hint = expected_active;
            return child;
          },
          py::return_value_policy::reference)
      .def("dynamic", &SNode::dynamic, py::return_value_policy::reference)
      .def("bitmasked",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const DebugInfo &))(&SNode::bitmasked),
           py::return_value_policy::reference)
      .def("bit_struct", &SNode::bit_struct, py::return_value_policy::reference)
      .def("quant_array", &SNode::quant_array,
           py::return_value_policy::reference)
      .def("place", &SNode::place)
      .def("data_type", [](SNode *snode) { return snode->dt; })
      .def("name", [](SNode *snode) { return snode->name; })
      .def("get_num_ch",
           [](SNode *snode) -> int { return (int)snode->ch.size(); })
      .def(
          "get_ch",
          [](SNode *snode, int i) -> SNode * { return snode->ch[i].get(); },
          py::return_value_policy::reference)
      .def("lazy_grad", &SNode::lazy_grad)
      .def("lazy_dual", &SNode::lazy_dual)
      .def("allocate_adjoint_checkbit", &SNode::allocate_adjoint_checkbit)
      .def("read_int", &SNode::read_int)
      .def("read_uint", &SNode::read_uint)
      .def("read_float", &SNode::read_float)
      .def("has_adjoint", &SNode::has_adjoint)
      .def("has_adjoint_checkbit", &SNode::has_adjoint_checkbit)
      .def("get_snode_grad_type", &SNode::get_snode_grad_type)
      .def("has_dual", &SNode::has_dual)
      .def("is_primal", &SNode::is_primal)
      .def("is_place", &SNode::is_place)
      .def("get_expr", &SNode::get_expr)
      .def("write_int", &SNode::write_int)
      .def("write_uint", &SNode::write_uint)
      .def("write_float", &SNode::write_float)
      .def("get_shape_along_axis", &SNode::shape_along_axis)
      .def("get_physical_index_position",
           [](SNode *snode) {
             return std::vector<int>(
                 snode->physical_index_position,
                 snode->physical_index_position + taichi_max_num_indices);
           })
      .def("num_active_indices",
           [](SNode *snode) { return snode->num_active_indices; })
      .def_readonly("cell_size_bytes", &SNode::cell_size_bytes)
      .def_readonly("offset_bytes_in_parent_cell",
                    &SNode::offset_bytes_in_parent_cell);

  py::class_<SNodeTree>(m, "SNodeTree")
      .def("id", &SNodeTree::id)
      .def("generation", &SNodeTree::generation)
      .def("destroy_snode_tree", [](SNodeTree *snode_tree, Program *program) {
        program->destroy_snode_tree(snode_tree);
      });

  py::class_<DeviceAllocation>(m, "DeviceAllocation")
      .def(py::init([](uint64_t device, uint64_t alloc_id) -> DeviceAllocation {
             DeviceAllocation alloc;
             alloc.device = (Device *)device;
             alloc.alloc_id = (DeviceAllocationId)alloc_id;
             return alloc;
           }),
           py::arg("device"), py::arg("alloc_id"))
      .def_readonly("device", &DeviceAllocation::device)
      .def_readonly("alloc_id", &DeviceAllocation::alloc_id);

  py::class_<Ndarray>(m, "Ndarray")
      .def("device_allocation_ptr", &Ndarray::get_device_allocation_ptr_as_int)
      .def("device_allocation", &Ndarray::get_device_allocation)
      .def("element_size", &Ndarray::get_element_size)
      .def("nelement", &Ndarray::get_nelement)
      .def("read_int", &Ndarray::read_int)
      .def("read_uint", &Ndarray::read_uint)
      .def("read_float", &Ndarray::read_float)
      .def("write_int", &Ndarray::write_int)
      .def("write_float", &Ndarray::write_float)
      .def("total_shape", &Ndarray::total_shape)
      .def("element_shape", &Ndarray::get_element_shape)
      .def("element_data_type", &Ndarray::get_element_data_type)
      .def_readonly("dtype", &Ndarray::dtype)
      .def_readonly("shape", &Ndarray::shape);

  py::class_<ArgPack>(m, "ArgPack")
      .def("device_allocation_ptr", &ArgPack::get_device_allocation_ptr_as_int)
      .def("device_allocation", &ArgPack::get_device_allocation)
      .def("nelement", &ArgPack::get_nelement)
      .def("data_type", &ArgPack::get_data_type)
      .def("set_arg_float", &ArgPack::set_arg_float)
      .def("set_arg_int", &ArgPack::set_arg_int)
      .def("set_arg_uint", &ArgPack::set_arg_uint)
      .def("set_arg_nested_argpack", &ArgPack::set_arg_nested_argpack)
      .def_readonly("dtype", &ArgPack::dtype);

  py::enum_<BufferFormat>(m, "Format")
#define PER_BUFFER_FORMAT(x) .value(#x, BufferFormat::x)
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_EXTENSION
      ;

  py::class_<Texture>(m, "Texture")
      .def("device_allocation_ptr", &Texture::get_device_allocation_ptr_as_int)
      .def("from_ndarray", &Texture::from_ndarray)
      .def("from_snode", &Texture::from_snode);

  py::enum_<aot::ArgKind>(m, "ArgKind")
      .value("SCALAR", aot::ArgKind::kScalar)
      .value("NDARRAY", aot::ArgKind::kNdarray)
      // Using this MATRIX as Scalar alias, we can move to native matrix type
      // when supported
      .value("MATRIX", aot::ArgKind::kMatrix)
      .value("TEXTURE", aot::ArgKind::kTexture)
      .value("RWTEXTURE", aot::ArgKind::kRWTexture)
      .export_values();

  py::class_<aot::Arg>(m, "Arg")
      .def(py::init<aot::ArgKind, std::string, DataType &, size_t,
                    std::vector<int>>(),
           py::arg("tag"), py::arg("name"), py::arg("dtype"),
           py::arg("field_dim"), py::arg("element_shape"))
      .def(py::init<aot::ArgKind, std::string, DataType &, size_t,
                    std::vector<int>>(),
           py::arg("tag"), py::arg("name"), py::arg("channel_format"),
           py::arg("num_channels"), py::arg("shape"))
      .def_readonly("tag", &aot::Arg::tag)
      .def_readonly("name", &aot::Arg::name)
      .def_readonly("element_shape", &aot::Arg::element_shape)
      .def_readonly("texture_shape", &aot::Arg::element_shape)
      .def_readonly("field_dim", &aot::Arg::field_dim)
      .def_readonly("num_channels", &aot::Arg::num_channels)
      .def("dtype", &aot::Arg::dtype)
      .def("element_dtype", &aot::Arg::element_dtype)
      .def("channel_format", &aot::Arg::dtype);

  py::class_<Node>(m, "Node");  // NOLINT(bugprone-unused-raii)

  py::class_<Sequential, Node>(m, "Sequential")
      .def(py::init<GraphBuilder *>())
      .def("append", &Sequential::append)
      .def("dispatch", &Sequential::dispatch);

  py::class_<GraphBuilder>(m, "GraphBuilder")
      .def(py::init<>())
      .def("dispatch", &GraphBuilder::dispatch)
      .def("compile", &GraphBuilder::compile)
      .def("create_sequential", &GraphBuilder::new_sequential_node,
           py::return_value_policy::reference)
      .def("seq", &GraphBuilder::seq, py::return_value_policy::reference);

  auto jit_run_graph = [](aot::CompiledGraph *self,
                          const CompileConfig &compile_config,
                          const py::dict &pyargs,
                          aot::CompiledGraphJITCache *cache) {
        std::unordered_map<std::string, aot::IValue> args;
        auto insert_scalar_arg = [&args](std::string arg_name,
                                         DataType expected_dtype,
                                         py::object pyarg) {
          auto type_id = expected_dtype->as<PrimitiveType>()->type;
          switch (type_id) {
#define PER_C_TYPE(type, ctype)                                           \
  case PrimitiveTypeID::type:                                             \
    args.insert({arg_name, aot::IValue::create(py::cast<ctype>(pyarg))}); \
    break;
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
            default:
              TI_ERROR("Unsupported scalar type {}",
                       expected_dtype->to_string());
          }
        };

        std::vector<std::unique_ptr<char[]>> matrix_buffers;
        matrix_buffers.reserve(self->args.size());
        std::vector<Matrix> matrices;
        // Reserve to avoid changes in element addresses
        matrices.reserve(self->args.size());
        auto matrix_dimensions = [](const py::buffer_info &buffer_info) {
          TI_ERROR_IF(buffer_info.ndim < 1 || buffer_info.ndim > 2,
                      "Graph Matrix runtime arguments must have rank 1 or 2, "
                      "but got rank {}",
                      buffer_info.ndim);
          return std::array<uint32_t, 3>{
              static_cast<uint32_t>(buffer_info.ndim),
              static_cast<uint32_t>(buffer_info.shape[0]),
              buffer_info.ndim == 2
                  ? static_cast<uint32_t>(buffer_info.shape[1])
                  : 0};
        };
        for (const auto &[arg_name, arg] : self->args) {
          auto tag = arg.tag;
          TI_ASSERT(pyargs.contains(arg_name.c_str()));
          auto pyarg = pyargs[arg_name.c_str()];
          if (tag == aot::ArgKind::kNdarray) {
            auto &val = pyarg.cast<Ndarray &>();
            args.insert({arg_name, aot::IValue::create(val)});
          } else if (tag == aot::ArgKind::kTexture ||
                     tag == aot::ArgKind::kRWTexture) {
            auto &val = pyarg.cast<Texture &>();
            args.insert({arg_name, aot::IValue::create(val)});
          } else if (tag == aot::ArgKind::kScalar) {
            auto expected_dtype = arg.dtype();
            insert_scalar_arg(arg_name, expected_dtype, pyarg);
          } else if (tag == aot::ArgKind::kMatrix) {
            auto type_id = arg.dtype()->as<PrimitiveType>()->type;
            switch (type_id) {
              case PrimitiveTypeID::f16: {
                auto arr = pyarg.cast<py::array_t<
                    float32, py::array::c_style | py::array::forcecast>>();
                py::buffer_info buffer_info = arr.request();
                auto length = buffer_info.size;
                auto ptr = reinterpret_cast<intptr_t>(buffer_info.ptr);
                auto dimensions = matrix_dimensions(buffer_info);
                auto byte_size = sizeof(uint16) * length;
                TI_ERROR_IF(byte_size > 128,
                            "Graph Matrix runtime argument {} uses {} bytes; "
                            "the limit is 128 bytes",
                            arg_name, byte_size);

                std::unique_ptr<char[]> data(new char[byte_size]);
                for (uint32_t i = 0; i < length; i++) {
                  uint16 half = fp16_ieee_from_fp32_value(
                      reinterpret_cast<float32 *>(ptr)[i]);
                  reinterpret_cast<uint16 *>(data.get())[i] = half;
                }
                matrix_buffers.emplace_back(std::move(data));

                matrices.emplace_back(Matrix(
                    length, arg.dtype(),
                    reinterpret_cast<intptr_t>(matrix_buffers.back().get()),
                    dimensions[0], dimensions[1], dimensions[2]));
                args.insert({arg_name, aot::IValue::create(matrices.back())});
                break;
              }
#define PER_C_TYPE(type, ctype)                                           \
  case PrimitiveTypeID::type: {                                           \
    auto arr = pyarg.cast<py::array_t<                                   \
        ctype, py::array::c_style | py::array::forcecast>>();            \
    py::buffer_info buffer_info = arr.request();                          \
    auto length = buffer_info.size;                                       \
    auto ptr = reinterpret_cast<intptr_t>(buffer_info.ptr);               \
    auto dimensions = matrix_dimensions(buffer_info);                     \
    auto byte_size = sizeof(ctype) * length;                              \
    TI_ERROR_IF(byte_size > 128,                                          \
                "Graph Matrix runtime argument {} uses {} bytes; the "    \
                "limit is 128 bytes",                                    \
                arg_name, byte_size);                                     \
                                                                           \
    std::unique_ptr<char[]> data(new char[byte_size]);                    \
    std::memcpy(data.get(), reinterpret_cast<char *>(ptr),                \
                byte_size);                                               \
    matrix_buffers.emplace_back(std::move(data));                         \
                                                                          \
    matrices.emplace_back(                                                \
        Matrix(length, arg.dtype(),                                       \
               reinterpret_cast<intptr_t>(matrix_buffers.back().get()),   \
               dimensions[0], dimensions[1], dimensions[2]));            \
    args.insert({arg_name, aot::IValue::create(matrices.back())});        \
    break;                                                                \
  }
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
              default:
                TI_ERROR("Unsupported scalar type {}",
                         arg.dtype()->to_string());
            }
          } else {
            TI_NOT_IMPLEMENTED;
          }
        }
        // Argument conversion above touches Python objects and must keep the
        // GIL. Once it is complete, graph execution is entirely native. In
        // particular, releasing the GIL here lets independent Python callers
        // reach CompiledGraphJITCache::run_mutex instead of being accidentally
        // serialized by Python. The cache owns the C++ transaction boundary;
        // Python objects passed in pyargs remain alive for this call.
        py::gil_scoped_release release;
        if (cache) {
          self->jit_run_cached(compile_config, args, *cache);
        } else {
          self->jit_run(compile_config, args);
        }
      };

  py::class_<aot::CompiledGraphJITCache>(m, "CompiledGraphJITCache")
      .def(py::init<>())
      .def("clear_runtime_state",
           &aot::CompiledGraphJITCache::clear_runtime_state)
      .def("_debug_graph_stats", [](aot::CompiledGraphJITCache &cache) {
        const auto snapshot = cache.debug_graph_stats();
        const auto &stats = snapshot.stats;
        auto backend_name = [](aot::CompiledGraphBackend backend) {
          switch (backend) {
            case aot::CompiledGraphBackend::cuda:
              return "cuda";
            case aot::CompiledGraphBackend::vulkan:
              return "vulkan";
            case aot::CompiledGraphBackend::none:
              return "none";
          }
          return "unknown";
        };
        auto path_name = [](aot::CompiledGraphExecutionPath path) {
          switch (path) {
            case aot::CompiledGraphExecutionPath::ordinary_fallback:
              return "ordinary_fallback";
            case aot::CompiledGraphExecutionPath::cuda_capture:
              return "cuda_capture";
            case aot::CompiledGraphExecutionPath::cuda_exact_replay:
              return "cuda_exact_replay";
            case aot::CompiledGraphExecutionPath::cuda_patched_replay:
              return "cuda_patched_replay";
            case aot::CompiledGraphExecutionPath::vulkan_record:
              return "vulkan_record";
            case aot::CompiledGraphExecutionPath::vulkan_replay:
              return "vulkan_replay";
            case aot::CompiledGraphExecutionPath::none:
              return "none";
          }
          return "unknown";
        };
        auto fallback_name = [](aot::CompiledGraphFallbackReason reason) {
          switch (reason) {
            case aot::CompiledGraphFallbackReason::debug_mode:
              return "debug_mode";
            case aot::CompiledGraphFallbackReason::insufficient_dispatches:
              return "insufficient_dispatches";
            case aot::CompiledGraphFallbackReason::unsupported_arguments:
              return "unsupported_arguments";
            case aot::CompiledGraphFallbackReason::resource_unavailable:
              return "resource_unavailable";
            case aot::CompiledGraphFallbackReason::structural_unsupported:
              return "structural_unsupported";
            case aot::CompiledGraphFallbackReason::transient_driver_failure:
              return "transient_driver_failure";
            case aot::CompiledGraphFallbackReason::fatal_driver_failure:
              return "fatal_driver_failure";
            case aot::CompiledGraphFallbackReason::retry_backoff:
              return "retry_backoff";
            case aot::CompiledGraphFallbackReason::runtime_mode:
              return "runtime_mode";
            case aot::CompiledGraphFallbackReason::replay_slot_saturated:
              return "replay_slot_saturated";
            case aot::CompiledGraphFallbackReason::none:
              return "none";
          }
          return "unknown";
        };
        py::dict result;
        result["backend"] = backend_name(stats.backend);
        result["last_path"] = path_name(stats.last_path);
        result["last_fallback_reason"] =
            fallback_name(stats.last_fallback_reason);
        result["attempts"] = stats.attempts;
        result["ordinary_fallbacks"] = stats.ordinary_fallbacks;
        result["capture_attempts"] = stats.capture_attempts;
        result["captures"] = stats.captures;
        result["exact_replays"] = stats.exact_replays;
        result["patched_replays"] = stats.patched_replays;
        result["recaptures"] = stats.recaptures;
        result["records"] = stats.records;
        result["replays"] = stats.replays;
        result["structural_fallbacks"] = stats.structural_fallbacks;
        result["transient_failures"] = stats.transient_failures;
        result["retry_backoff_fallbacks"] =
            stats.retry_backoff_fallbacks;
        result["replay_slot_saturation_fallbacks"] =
            stats.replay_slot_saturation_fallbacks;
        result["capture_exceptions"] = stats.capture_exceptions;
        result["zero_arg_captures"] = stats.zero_arg_captures;
        result["zero_arg_eligible"] = stats.zero_arg_eligible;
        result["known_persistent_argument_bytes"] =
            stats.known_persistent_argument_bytes;
        result["known_compiled_tasks"] =
            snapshot.known_compiled_tasks;
        result["known_compiled_dispatches"] =
            snapshot.known_compiled_dispatches;
        result["last_driver_error"] = stats.last_driver_error;
        result["retry_backoff_remaining"] =
            stats.retry_backoff_remaining;
        result["consecutive_transient_failures"] =
            stats.consecutive_transient_failures;
        result["diagnostics_previously_enabled"] =
            snapshot.diagnostics_previously_enabled;
        result["diagnostics_counters_complete"] =
            snapshot.diagnostics_counters_complete;
        return result;
      });

  py::class_<aot::CompiledGraph>(m, "CompiledGraph")
      .def_property_readonly(
          "_snode_tree_dependencies",
          [](const aot::CompiledGraph &graph) {
            py::list dependencies;
            for (const auto &dependency : graph.snode_tree_dependencies) {
              dependencies.append(
                  py::make_tuple(dependency.tree_id, dependency.generation));
            }
            return dependencies;
          })
      .def_property_readonly(
          "_snode_tree_dependency_info",
          [](const aot::CompiledGraph &graph) {
            py::list dependencies;
            for (const auto &dependency : graph.snode_tree_dependencies) {
              dependencies.append(py::make_tuple(
                  dependency.tree_id, dependency.generation,
                  dependency.layout_fingerprint));
            }
            return dependencies;
          })
      .def("jit_run",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs) {
             jit_run_graph(self, compile_config, pyargs, nullptr);
           })
      .def("jit_run_cached",
           [jit_run_graph](aot::CompiledGraph *self,
                           const CompileConfig &compile_config,
                           const py::dict &pyargs,
                           aot::CompiledGraphJITCache &cache) {
             jit_run_graph(self, compile_config, pyargs, &cache);
           });

  py::class_<Kernel>(m, "Kernel")
      .def("no_activate",
           [](Kernel *self, SNode *snode) {
             // TODO(#2193): Also apply to @ti.func?
             self->no_activate.push_back(snode);
           })
      .def("insert_scalar_param", &Kernel::insert_scalar_param)
      .def("insert_arr_param", &Kernel::insert_arr_param)
      .def("insert_ndarray_param", &Kernel::insert_ndarray_param)
      .def("insert_texture_param", &Kernel::insert_texture_param)
      .def("insert_pointer_param", &Kernel::insert_pointer_param)
      .def("insert_rw_texture_param", &Kernel::insert_rw_texture_param)
      .def("insert_argpack_param_and_push",
           &Kernel::insert_argpack_param_and_push)
      .def("pop_argpack_stack", &Kernel::pop_argpack_stack)
      .def("insert_ret", &Kernel::insert_ret)
      .def("finalize_rets", &Kernel::finalize_rets)
      .def("finalize_params", &Kernel::finalize_params)
      .def("make_launch_context", &Kernel::make_launch_context)
      .def("definition_retired", &Kernel::definition_retired)
      .def("set_compile_tier_override",
           &Kernel::set_compile_tier_override)
      .def("clear_compile_tier_override",
           &Kernel::clear_compile_tier_override)
      .def("get_compile_tier_override",
           [](Kernel *self) -> py::object {
             const auto &v = self->get_compile_tier_override();
             if (v.has_value()) {
               return py::cast(*v);
             }
             return py::none();
           })
      .def(
          "ast_builder",
          [](Kernel *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference);

  py::class_<LaunchContextBuilder>(m, "KernelLaunchContext")
      .def("set_arg_int", &LaunchContextBuilder::set_arg_int)
      .def("set_arg_uint", &LaunchContextBuilder::set_arg_uint)
      .def("set_arg_float", &LaunchContextBuilder::set_arg_float)
      .def("set_struct_arg_int", &LaunchContextBuilder::set_struct_arg<int64>)
      .def("set_struct_arg_uint", &LaunchContextBuilder::set_struct_arg<uint64>)
      .def("set_struct_arg_float",
           &LaunchContextBuilder::set_struct_arg<double>)
      .def("set_arg_external_array_with_shape",
           &LaunchContextBuilder::set_arg_external_array_with_shape)
      .def("set_arg_argpack", &LaunchContextBuilder::set_arg_argpack)
      .def("_debug_set_argpack_resource_handle",
           [](LaunchContextBuilder &ctx, const std::vector<int> &arg_id,
              std::uint64_t domain, std::uint32_t kind, std::uint32_t index,
              std::uint32_t generation) {
             ctx.debug_set_argpack_resource_handle(
                 arg_id, RuntimeResourceHandle{domain, kind, index,
                                               generation});
           })
      .def("set_arg_ndarray", &LaunchContextBuilder::set_arg_ndarray)
      .def("set_arg_ndarray_with_grad",
           &LaunchContextBuilder::set_arg_ndarray_with_grad)
      .def("_debug_set_ndarray_resource_handle",
           [](LaunchContextBuilder &ctx, const std::vector<int> &arg_id,
              std::uint64_t domain, std::uint32_t kind, std::uint32_t index,
              std::uint32_t generation) {
             ctx.debug_set_ndarray_resource_handle(
                 arg_id, RuntimeResourceHandle{domain, kind, index,
                                               generation});
           })
      .def("set_arg_texture", &LaunchContextBuilder::set_arg_texture)
      .def("_debug_set_texture_resource_handle",
           [](LaunchContextBuilder &ctx, const std::vector<int> &arg_id,
              std::uint64_t domain, std::uint32_t kind, std::uint32_t index,
              std::uint32_t generation) {
             ctx.debug_set_texture_resource_handle(
                 arg_id, RuntimeResourceHandle{domain, kind, index,
                                               generation});
           })
      .def("set_arg_rw_texture", &LaunchContextBuilder::set_arg_rw_texture)
      .def("get_struct_ret_int", &LaunchContextBuilder::get_struct_ret_int)
      .def("get_struct_ret_uint", &LaunchContextBuilder::get_struct_ret_uint)
      .def("get_struct_ret_float", &LaunchContextBuilder::get_struct_ret_float);

  py::class_<Function>(m, "Function")
      .def("insert_scalar_param", &Function::insert_scalar_param)
      .def("insert_arr_param", &Function::insert_arr_param)
      .def("insert_ndarray_param", &Function::insert_ndarray_param)
      .def("insert_texture_param", &Function::insert_texture_param)
      .def("insert_pointer_param", &Function::insert_pointer_param)
      .def("insert_rw_texture_param", &Function::insert_rw_texture_param)
      .def("insert_ret", &Function::insert_ret)
      .def("set_function_body",
           py::overload_cast<const std::function<void()> &>(
               &Function::set_function_body))
      .def("finalize_rets", &Function::finalize_rets)
      .def("finalize_params", &Function::finalize_params)
      .def(
          "ast_builder",
          [](Function *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference);

  py::class_<Expr> expr(m, "Expr");
  expr.def("snode", &Expr::snode, py::return_value_policy::reference)
      .def("is_external_tensor_expr",
           [](Expr *expr) { return expr->is<ExternalTensorExpression>(); })
      .def("is_index_expr",
           [](Expr *expr) { return expr->is<IndexExpression>(); })
      .def("is_primal",
           [](Expr *expr) {
             return expr->cast<FieldExpression>()->snode_grad_type ==
                    SNodeGradType::kPrimal;
           })
      .def("is_lvalue", [](Expr *expr) { return expr->expr->is_lvalue(); })
      .def("set_dbg_info", &Expr::set_dbg_info)
      .def("get_dbg_info", [](Expr *expr) { return expr->expr->dbg_info; })
      .def("set_name",
           [&](Expr *expr, std::string na) {
             expr->cast<FieldExpression>()->name = na;
           })
      .def("set_grad_type",
           [&](Expr *expr, SNodeGradType t) {
             expr->cast<FieldExpression>()->snode_grad_type = t;
           })
      .def("set_adjoint", &Expr::set_adjoint)
      .def("set_adjoint_checkbit", &Expr::set_adjoint_checkbit)
      .def("set_dual", &Expr::set_dual)
      .def("set_dynamic_index_stride",
           [&](Expr *expr, int dynamic_index_stride) {
             auto matrix_field = expr->cast<MatrixFieldExpression>();
             matrix_field->dynamic_indexable = true;
             matrix_field->dynamic_index_stride = dynamic_index_stride;
           })
      .def("get_dynamic_indexable",
           [&](Expr *expr) -> bool {
             return expr->cast<MatrixFieldExpression>()->dynamic_indexable;
           })
      .def("get_dynamic_index_stride",
           [&](Expr *expr) -> int {
             return expr->cast<MatrixFieldExpression>()->dynamic_index_stride;
           })
      .def(
          "get_dt",
          [&](Expr *expr) -> const Type * {
            return expr->cast<FieldExpression>()->dt;
          },
          py::return_value_policy::reference)
      .def("get_ret_type", &Expr::get_ret_type)
      .def("get_rvalue_type",
           [](Expr *expr) { return expr->get_rvalue_type(); })
      .def("is_tensor",
           [](Expr *expr) { return expr->get_rvalue_type()->is<TensorType>(); })
      .def("is_struct",
           [](Expr *expr) { return expr->get_rvalue_type()->is<StructType>(); })
      .def("get_shape",
           [](Expr *expr) -> std::optional<std::vector<int>> {
             auto tensor_type = expr->get_rvalue_type()->cast<TensorType>();
             if (tensor_type) {
               return std::optional<std::vector<int>>(tensor_type->get_shape());
             }
             return std::nullopt;
           })
      .def("type_check", &Expr::type_check)
      .def("get_expr_name",
           [](Expr *expr) { return expr->cast<FieldExpression>()->name; })
      .def("get_raw_address", [](Expr *expr) { return (uint64)expr; })
      .def("get_underlying_ptr_address", [](Expr *e) {
        // The reason that there are both get_raw_address() and
        // get_underlying_ptr_address() is that Expr itself is mostly wrapper
        // around its underlying |expr| (of type Expression). Expr |e| can be
        // temporary, while the underlying |expr| is mostly persistent.
        //
        // Same get_raw_address() implies that get_underlying_ptr_address() are
        // also the same. The reverse is not true.
        return (uint64)e->expr.get();
      });

  py::class_<ExprGroup>(m, "ExprGroup")
      .def(py::init<>())
      .def("size", [](ExprGroup *eg) { return eg->exprs.size(); })
      .def("push_back", &ExprGroup::push_back);

  py::class_<Stmt>(m, "Stmt");  // NOLINT(bugprone-unused-raii)

  m.def("insert_internal_func_call", [&](Operation *op, const ExprGroup &args) {
    return Expr::make<InternalFuncCallExpression>(op, args.exprs);
  });

  m.def("make_get_element_expr",
        Expr::make<GetElementExpression, const Expr &, std::vector<int>,
                   const DebugInfo &>);

  m.def("value_cast", static_cast<Expr (*)(const Expr &expr, DataType)>(cast));
  m.def("bits_cast",
        static_cast<Expr (*)(const Expr &expr, DataType)>(bit_cast));

  m.def("expr_atomic_add", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::add, a, b);
  });

  m.def("expr_atomic_sub", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::sub, a, b);
  });

  m.def("expr_atomic_min", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::min, a, b);
  });

  m.def("expr_atomic_max", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::max, a, b);
  });

  m.def("expr_atomic_bit_and", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_and, a, b);
  });

  m.def("expr_atomic_bit_or", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_or, a, b);
  });

  m.def("expr_atomic_bit_xor", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_xor, a, b);
  });

  m.def("expr_atomic_mul", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::mul, a, b);
  });

  m.def("expr_assume_in_range", assume_range);

  m.def("expr_loop_unique", loop_unique);

  m.def("expr_field", expr_field);

  m.def("expr_matrix_field", expr_matrix_field);

#define DEFINE_EXPRESSION_OP(x) m.def("expr_" #x, expr_##x);

  DEFINE_EXPRESSION_OP(neg)
  DEFINE_EXPRESSION_OP(sqrt)
  DEFINE_EXPRESSION_OP(round)
  DEFINE_EXPRESSION_OP(floor)
  DEFINE_EXPRESSION_OP(frexp)
  DEFINE_EXPRESSION_OP(ceil)
  DEFINE_EXPRESSION_OP(abs)
  DEFINE_EXPRESSION_OP(sin)
  DEFINE_EXPRESSION_OP(asin)
  DEFINE_EXPRESSION_OP(cos)
  DEFINE_EXPRESSION_OP(acos)
  DEFINE_EXPRESSION_OP(tan)
  DEFINE_EXPRESSION_OP(tanh)
  DEFINE_EXPRESSION_OP(inv)
  DEFINE_EXPRESSION_OP(rcp)
  DEFINE_EXPRESSION_OP(rsqrt)
  DEFINE_EXPRESSION_OP(exp)
  DEFINE_EXPRESSION_OP(log)
  DEFINE_EXPRESSION_OP(popcnt)
  DEFINE_EXPRESSION_OP(clz)

  DEFINE_EXPRESSION_OP(select)
  DEFINE_EXPRESSION_OP(ifte)

  DEFINE_EXPRESSION_OP(cmp_le)
  DEFINE_EXPRESSION_OP(cmp_lt)
  DEFINE_EXPRESSION_OP(cmp_ge)
  DEFINE_EXPRESSION_OP(cmp_gt)
  DEFINE_EXPRESSION_OP(cmp_ne)
  DEFINE_EXPRESSION_OP(cmp_eq)

  DEFINE_EXPRESSION_OP(bit_and)
  DEFINE_EXPRESSION_OP(bit_or)
  DEFINE_EXPRESSION_OP(bit_xor)
  DEFINE_EXPRESSION_OP(bit_shl)
  DEFINE_EXPRESSION_OP(bit_shr)
  DEFINE_EXPRESSION_OP(bit_sar)
  DEFINE_EXPRESSION_OP(bit_not)

  DEFINE_EXPRESSION_OP(logic_not)
  DEFINE_EXPRESSION_OP(logical_and)
  DEFINE_EXPRESSION_OP(logical_or)

  DEFINE_EXPRESSION_OP(add)
  DEFINE_EXPRESSION_OP(sub)
  DEFINE_EXPRESSION_OP(mul)
  DEFINE_EXPRESSION_OP(div)
  DEFINE_EXPRESSION_OP(truediv)
  DEFINE_EXPRESSION_OP(floordiv)
  DEFINE_EXPRESSION_OP(mod)
  DEFINE_EXPRESSION_OP(max)
  DEFINE_EXPRESSION_OP(min)
  DEFINE_EXPRESSION_OP(atan2)
  DEFINE_EXPRESSION_OP(pow)

#undef DEFINE_EXPRESSION_OP

  m.def("make_global_load_stmt", Stmt::make<GlobalLoadStmt, Stmt *>);
  m.def("make_global_store_stmt", Stmt::make<GlobalStoreStmt, Stmt *, Stmt *>);
  m.def("make_frontend_assign_stmt",
        Stmt::make<FrontendAssignStmt, const Expr &, const Expr &,
                   const DebugInfo &>);

  m.def("make_arg_load_expr",
        Expr::make<ArgLoadExpression, const std::vector<int> &,
                   const DataType &, bool, bool, int, const DebugInfo &>,
        "arg_id"_a, "dt"_a, "is_ptr"_a = false, "create_load"_a = true,
        "arg_depth"_a = 0, "dbg_info"_a = DebugInfo());

  m.def("make_reference",
        Expr::make<ReferenceExpression, const Expr &, const DebugInfo &>);

  m.def("make_external_tensor_expr",
        Expr::make<ExternalTensorExpression, const DataType &, int,
                   const std::vector<int> &, bool, int, const BoundaryMode &>);
  m.def("make_external_tensor_strided_expr",
        Expr::make<ExternalTensorExpression, const DataType &, int,
                   const std::vector<int> &, bool, int, const BoundaryMode &,
                   std::size_t, std::size_t>);
  m.def("make_external_tensor_member_expr",
        [](const Expr &expr, const DataType &dt, std::size_t byte_offset,
           std::size_t byte_stride) {
          TI_ASSERT(expr.is<ExternalTensorExpression>());
          auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
          return Expr::make<ExternalTensorExpression>(
              dt, external_tensor_expr->ndim, external_tensor_expr->arg_id,
              external_tensor_expr->needs_grad,
              external_tensor_expr->arg_depth, external_tensor_expr->boundary,
              byte_offset, byte_stride);
        });

  m.def("make_external_tensor_grad_expr",
        Expr::make<ExternalTensorExpression, Expr *>);

  m.def("make_rand_expr",
        Expr::make<RandExpression, const DataType &, const DebugInfo &>);

  m.def("make_const_expr_bool",
        Expr::make<ConstExpression, const DataType &, uint1>);

  m.def("make_const_expr_int",
        Expr::make<ConstExpression, const DataType &, int64>);

  m.def("make_const_expr_fp",
        Expr::make<ConstExpression, const DataType &, float64>);

  m.def("make_texture_ptr_expr",
        Expr::make<TexturePtrExpression, const std::vector<int> &, int, int,
                   const DebugInfo &>);
  m.def("make_rw_texture_ptr_expr",
        Expr::make<TexturePtrExpression, const std::vector<int> &, int, int,
                   const BufferFormat &, int, const DebugInfo &>);

  auto &&texture =
      py::enum_<TextureOpType>(m, "TextureOpType", py::arithmetic());
  for (int t = 0; t <= (int)TextureOpType::kStore; t++)
    texture.value(texture_op_type_name(TextureOpType(t)).c_str(),
                  TextureOpType(t));
  texture.export_values();

  auto &&bin = py::enum_<BinaryOpType>(m, "BinaryOpType", py::arithmetic());
  for (int t = 0; t <= (int)BinaryOpType::undefined; t++)
    bin.value(binary_op_type_name(BinaryOpType(t)).c_str(), BinaryOpType(t));
  bin.export_values();
  m.def("make_binary_op_expr",
        Expr::make<BinaryOpExpression, const BinaryOpType &, const Expr &,
                   const Expr &>);

  auto &&unary = py::enum_<UnaryOpType>(m, "UnaryOpType", py::arithmetic());
  for (int t = 0; t <= (int)UnaryOpType::undefined; t++)
    unary.value(unary_op_type_name(UnaryOpType(t)).c_str(), UnaryOpType(t));
  unary.export_values();
  m.def("make_unary_op_expr",
        Expr::make<UnaryOpExpression, const UnaryOpType &, const Expr &>);
#define PER_TYPE(x)                                                  \
  m.attr(("DataType_" + data_type_name(PrimitiveType::x)).c_str()) = \
      PrimitiveType::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

  m.def("data_type_size", data_type_size);
  m.def("data_type_alignment", data_type_alignment);
  m.def("data_type_element_offset",
        [](DataType dt, const std::vector<int> &indices) {
          return dt.ptr_removed()->as<StructType>()->get_element_offset(indices);
        });
  m.def("is_quant", is_quant);
  m.def("is_integral", is_integral);
  m.def("is_signed", is_signed);
  m.def("is_real", is_real);
  m.def("is_unsigned", is_unsigned);
  m.def("is_tensor", is_tensor);

  m.def("data_type_name", data_type_name);

  m.def(
      "subscript_with_multiple_indices",
      Expr::make<IndexExpression, const Expr &, const std::vector<ExprGroup> &,
                 const std::vector<int> &, const DebugInfo &>);

  m.def("get_external_tensor_element_dim", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    // FIXME: no need to make it negative since we don't support SOA
    auto dtype = expr.cast<ExternalTensorExpression>()->dt;
    return dtype->is<TensorType>()
               ? -dtype->cast<TensorType>()->get_shape().size()
               : 0;
  });

  m.def("get_external_tensor_needs_grad", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->needs_grad;
  });

  m.def("get_external_tensor_arg_id", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->arg_id;
  });

  m.def("get_external_tensor_element_type", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
    return external_tensor_expr->dt;
  });

  m.def("get_external_tensor_element_shape", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
    return external_tensor_expr->dt.get_shape();
  });

  m.def("get_external_tensor_dim", [](const Expr &expr) {
    if (expr.is<ExternalTensorExpression>()) {
      return expr.cast<ExternalTensorExpression>()->ndim;
    } else if (expr.is<TexturePtrExpression>()) {
      return expr.cast<TexturePtrExpression>()->num_dims;
    } else {
      TI_ASSERT(false);
      return 0;
    }
  });

  m.def("get_external_tensor_shape_along_axis",
        Expr::make<ExternalTensorShapeAlongAxisExpression, const Expr &, int,
                   const DebugInfo &>);

  m.def("get_external_tensor_real_func_args",
        [](const Expr &expr, const DebugInfo &dbg_info = DebugInfo()) {
          TI_ASSERT(expr.is<ExternalTensorExpression>());
          auto external_tensor_expr = expr.cast<ExternalTensorExpression>();

          std::vector<Expr> args;
          for (int i = 0; i < external_tensor_expr->ndim; i++) {
            args.push_back(Expr::make<ExternalTensorShapeAlongAxisExpression>(
                expr, i, expr->dbg_info));
            args.back()->type_check(nullptr);
          }

          args.push_back(Expr::make<ExternalTensorBasePtrExpression>(
              expr, /*is_grad=*/false, dbg_info));
          args.back()->type_check(nullptr);

          if (external_tensor_expr->needs_grad) {
            args.push_back(Expr::make<ExternalTensorBasePtrExpression>(
                expr, /*is_grad=*/true, dbg_info));
            args.back()->type_check(nullptr);
          }

          return args;
        });

  // Mesh related.
  m.def("get_relation_size", [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx,
                                mesh::MeshElementType to_type,
                                const DebugInfo &dbg_info = DebugInfo()) {
    return Expr::make<MeshRelationAccessExpression>(
        mesh_ptr.ptr.get(), mesh_idx, to_type, dbg_info);
  });

  m.def("get_relation_access",
        [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx,
           mesh::MeshElementType to_type, const Expr &neighbor_idx,
           const DebugInfo &dbg_info = DebugInfo()) {
          return Expr::make<MeshRelationAccessExpression>(
              mesh_ptr.ptr.get(), mesh_idx, to_type, neighbor_idx, dbg_info);
        });

  py::class_<FunctionKey>(m, "FunctionKey")
      .def(py::init<const std::string &, int, int>())
      .def_readonly("instance_id", &FunctionKey::instance_id);

  m.def("test_throw", [] {
    try {
      throw IRModified();
    } catch (IRModified) {
      TI_INFO("caught");
    }
  });

  m.def("test_throw", [] { throw IRModified(); });

#if TI_WITH_LLVM
  m.def("libdevice_path", libdevice_path);
#endif

  m.def("host_arch", host_arch);
  m.def("arch_uses_llvm", arch_uses_llvm);

  m.def("set_lib_dir", [&](const std::string &dir) { compiled_lib_dir = dir; });
  m.def("set_tmp_dir", [&](const std::string &dir) { runtime_tmp_dir = dir; });

  m.def("get_commit_hash", get_commit_hash);
  m.def("get_version_string", get_version_string);
  m.def("get_version_major", get_version_major);
  m.def("get_version_minor", get_version_minor);
  m.def("get_version_patch", get_version_patch);
  m.def("get_llvm_target_support", [] {
#if defined(TI_WITH_LLVM)
    return LLVM_VERSION_STRING;
#else
    return "targets unsupported";
#endif
  });
  m.def("test_printf", [] { printf("test_printf\n"); });
  m.def("test_logging", [] { TI_INFO("test_logging"); });
  m.def("trigger_crash", [] { *(int *)(1) = 0; });
  m.def("get_max_num_indices", [] { return taichi_max_num_indices; });
  m.def("get_max_num_args", [] { return taichi_max_num_args; });
  m.def("test_threading", test_threading);
  m.def("is_extension_supported", is_extension_supported);

  m.def("query_int64", [](const std::string &key) {
    if (key == "cuda_compute_capability") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDAContext::get_instance().get_compute_capability());
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_driver_lock_sampled_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .lock.sampled_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_driver_lock_contended_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .lock.contended_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_context_lock_sampled_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDAContext::get_instance()
              .get_lock_telemetry_snapshot()
              .sampled_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_context_lock_contended_acquisitions") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDAContext::get_instance()
              .get_lock_telemetry_snapshot()
              .contended_acquisitions);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_async_allocation_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDADriver::get_instance().get_telemetry_snapshot().async_allocation_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_sync_allocation_fallback_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .sync_allocation_fallback_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_async_free_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(
          CUDADriver::get_instance().get_telemetry_snapshot().async_free_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_sync_free_fallback_calls") {
#if defined(TI_WITH_CUDA)
      return static_cast<int64_t>(CUDADriver::get_instance()
                                      .get_telemetry_snapshot()
                                      .sync_free_fallback_calls);
#else
      TI_NOT_IMPLEMENTED
#endif
    } else if (key == "vulkan_graph_replay_slot_saturation_fallbacks") {
#if defined(TI_WITH_VULKAN)
      return static_cast<int64_t>(
          gfx::get_graph_replay_slot_saturation_fallbacks());
#else
      TI_NOT_IMPLEMENTED
#endif
    } else {
      TI_ERROR("Key {} not supported in query_int64", key);
    }
  });

  // Type system

  py::class_<Type>(m, "Type").def("to_string", &Type::to_string);

  m.def("promoted_type", promoted_type);

  // Note that it is important to specify py::return_value_policy::reference for
  // the factory methods, otherwise pybind11 will delete the Types owned by
  // TypeFactory on Python-scope pointer destruction.
  py::class_<TypeFactory>(m, "TypeFactory")
      .def("get_quant_int_type", &TypeFactory::get_quant_int_type,
           py::arg("num_bits"), py::arg("is_signed"), py::arg("compute_type"),
           py::return_value_policy::reference)
      .def("get_quant_fixed_type", &TypeFactory::get_quant_fixed_type,
           py::arg("digits_type"), py::arg("compute_type"), py::arg("scale"),
           py::return_value_policy::reference)
      .def("get_quant_float_type", &TypeFactory::get_quant_float_type,
           py::arg("digits_type"), py::arg("exponent_type"),
           py::arg("compute_type"), py::return_value_policy::reference)
      .def(
          "get_tensor_type",
          [&](TypeFactory *factory, std::vector<int> shape,
              const DataType &element_type) {
            return factory->create_tensor_type(shape, element_type);
          },
          py::return_value_policy::reference)
      .def(
          "get_struct_type",
          [&](TypeFactory *factory,
              std::vector<std::pair<DataType, std::string>> elements) {
            std::vector<AbstractDictionaryMember> members;
            for (auto &[type, name] : elements) {
              members.push_back({type, name});
            }
            return DataType(factory->get_struct_type(members));
          },
          py::return_value_policy::reference)
      .def("get_rwtexture_struct_type", &TypeFactory::get_rwtexture_struct_type,
           py::return_value_policy::reference)
      .def("get_ndarray_struct_type", &TypeFactory::get_ndarray_struct_type,
           py::arg("dt"), py::arg("ndim"), py::arg("needs_grad"),
           py::return_value_policy::reference)
      .def("get_struct_type_for_argpack_ptr",
           &TypeFactory::get_struct_type_for_argpack_ptr, py::arg("dt"),
           py::arg("layout") = "none", py::return_value_policy::reference)
      .def(
          "get_argpack_type",
          [&](TypeFactory *factory,
              std::vector<std::pair<DataType, std::string>> elements) {
            std::vector<AbstractDictionaryMember> members;
            size_t pos = 0;
            for (auto &[type, name] : elements) {
              members.push_back({type, name, ++pos});
            }
            return DataType(factory->get_argpack_type(members));
          },
          py::return_value_policy::reference);

  m.def("get_type_factory_instance", TypeFactory::get_instance,
        py::return_value_policy::reference);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<BitStructType>(m, "BitStructType");
  py::class_<BitStructTypeBuilder>(m, "BitStructTypeBuilder")
      .def(py::init<int>())
      .def("begin_placing_shared_exponent",
           &BitStructTypeBuilder::begin_placing_shared_exponent)
      .def("end_placing_shared_exponent",
           &BitStructTypeBuilder::end_placing_shared_exponent)
      .def("add_member", &BitStructTypeBuilder::add_member)
      .def("build", &BitStructTypeBuilder::build,
           py::return_value_policy::reference);

  py::class_<SNodeRegistry>(m, "SNodeRegistry")
      .def(py::init<>())
      .def("create_root", &SNodeRegistry::create_root,
           py::return_value_policy::reference);

  m.def(
      "finalize_snode_tree",
      [](SNodeRegistry *registry, const SNode *root, Program *program,
         bool compile_only) -> SNodeTree * {
        return program->add_snode_tree(registry->finalize(root), compile_only);
      },
      py::return_value_policy::reference);

  // Sparse Matrix
  py::class_<SparseMatrixBuilder>(m, "SparseMatrixBuilder")
      .def(py::init<int, int, int, DataType, const std::string &>(),
           py::arg("rows"), py::arg("cols"), py::arg("max_num_triplets"),
           py::arg("dt") = PrimitiveType::f32,
           py::arg("storage_format") = "col_major")
      .def("print_triplets_eigen", &SparseMatrixBuilder::print_triplets_eigen)
      .def("print_triplets_cuda", &SparseMatrixBuilder::print_triplets_cuda)
      .def("create_ndarray",
           [&](SparseMatrixBuilder *builder, Program *prog) {
             return builder->create_ndarray(prog);
           })
      .def("delete_ndarray",
           [&](SparseMatrixBuilder *builder, Program *prog) {
             return builder->delete_ndarray(prog);
           })
      .def("get_ndarray_data_ptr", &SparseMatrixBuilder::get_ndarray_data_ptr)
      .def("get_ndarray", &SparseMatrixBuilder::get_ndarray,
           py::return_value_policy::reference)
      .def("build", &SparseMatrixBuilder::build)
      .def("build_cuda", &SparseMatrixBuilder::build_cuda)
      .def("build_vulkan", &SparseMatrixBuilder::build_vulkan)
      .def("get_addr", [](SparseMatrixBuilder *mat) { return uint64(mat); });

  py::class_<SparsePattern, std::shared_ptr<SparsePattern>>(m,
                                                            "SparsePattern");
  auto sparse_pattern_runtime_stats =
      [](const SparsePattern &pattern) {
        const auto stats = pattern.debug_runtime_statistics();
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["storage_format"] = stats.storage_format;
        identity["index_dtype"] = stats.index_dtype;
        identity["value_order"] = stats.value_order;
        identity["rows"] = stats.rows;
        identity["cols"] = stats.cols;
        identity["nnz"] = stats.nnz;
        identity["block_rows"] = stats.block_rows;
        identity["block_cols"] = stats.block_cols;
        identity["block_size"] = stats.block_size;
        identity["block_nnz"] = stats.block_nnz;
        identity["pattern_id"] = stats.pattern_id;
        identity["pattern_version"] = stats.pattern_version;

        py::dict lifecycle;
        lifecycle["immutable"] = stats.immutable;
        lifecycle["pattern_builds"] = stats.pattern_builds;
        lifecycle["operator_references"] = stats.operator_references;
        lifecycle["program_bound"] = true;

        py::dict resources;
        resources["pattern_reserved_bytes"] =
            stats.pattern_reserved_bytes;
        resources["ownership_scope"] =
            "shared_immutable_pattern_storage";
        resources["sum_once_across_operators"] = true;

        py::dict transfers;
        transfers["host_to_device_bytes"] = stats.host_to_device_bytes;
        transfers["device_to_host_bytes"] = stats.device_to_host_bytes;
        transfers["device_to_device_bytes"] =
            stats.device_to_device_bytes;
        transfers["scope"] = "pattern_creation_only";

        py::dict result;
        result["schema_version"] = 1;
        result["identity"] = std::move(identity);
        result["lifecycle"] = std::move(lifecycle);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        return result;
      };
  py::class_<SparseCsrPattern, SparsePattern,
             std::shared_ptr<SparseCsrPattern>>(m, "SparseCsrPattern")
      .def("_debug_runtime_stats", sparse_pattern_runtime_stats);
  py::class_<SparseBsrPattern, SparsePattern,
             std::shared_ptr<SparseBsrPattern>>(m, "SparseBsrPattern")
      .def("_debug_runtime_stats", sparse_pattern_runtime_stats);

  py::class_<SparseMatrix>(m, "SparseMatrix")
      .def(py::init<>())
      .def(py::init<int, int, DataType>(), py::arg("rows"), py::arg("cols"),
           py::arg("dt") = PrimitiveType::f32)
      .def(py::init<SparseMatrix &>())
      .def("to_string", &SparseMatrix::to_string)
      .def("get_element", &SparseMatrix::get_element<float32>)
      .def("set_element", &SparseMatrix::set_element<float32>)
      .def("mmwrite", &SparseMatrix::mmwrite)
      .def("num_rows", &SparseMatrix::num_rows)
      .def("num_cols", &SparseMatrix::num_cols)
      .def("num_nonzero", &SparseMatrix::num_nonzero)
      .def("update_values", &SparseMatrix::update_values)
      .def("_debug_runtime_stats", [](const SparseMatrix &matrix) {
        const auto stats = matrix.debug_runtime_statistics();
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["storage_format"] = stats.storage_format;
        identity["dtype"] = stats.dtype;
        identity["rows"] = stats.rows;
        identity["cols"] = stats.cols;
        identity["nnz"] = stats.nnz;
        if (stats.block_size > 0) {
          identity["block_rows"] = stats.block_rows;
          identity["block_cols"] = stats.block_cols;
          identity["block_size"] = stats.block_size;
          identity["block_nnz"] = stats.block_nnz;
        } else {
          identity["block_rows"] = py::none();
          identity["block_cols"] = py::none();
          identity["block_size"] = py::none();
          identity["block_nnz"] = py::none();
        }
        identity["pattern_version"] = stats.pattern_version;
        identity["numeric_version"] = stats.numeric_version;
        if (stats.pattern_storage_shared) {
          identity["pattern_id"] = stats.shared_pattern_id;
        } else {
          identity["pattern_id"] = py::none();
        }

        py::dict operations;
        operations["pattern_builds"] = stats.pattern_builds;
        operations["numeric_updates"] = stats.numeric_updates;
        operations["numeric_update_bytes"] = stats.numeric_update_bytes;
        operations["spmv_calls"] = stats.spmv_calls;
        operations["spmv_plan_builds"] = stats.spmv_plan_builds;
        operations["spmv_plan_reuses"] = stats.spmv_plan_reuses;
        operations["spmv_handle_creations"] =
            stats.spmv_handle_creations;
        operations["dense_vector_descriptor_creations"] =
            stats.dense_vector_descriptor_creations;
        operations["dense_vector_descriptor_rebinds"] =
            stats.dense_vector_descriptor_rebinds;
        operations["spmv_workspace_allocations"] =
            stats.spmv_workspace_allocations;
        operations["resource_generations_published"] =
            stats.resource_generations_published;
        operations["resource_generations_retired"] =
            stats.resource_generations_retired;
        operations["resource_generations_released"] =
            stats.resource_generations_released;

        py::dict resources;
        resources["pattern_reserved_bytes"] =
            stats.pattern_reserved_bytes;
        resources["values_reserved_bytes"] = stats.values_reserved_bytes;
        resources["spmv_workspace_reserved_bytes"] =
            stats.spmv_workspace_reserved_bytes;
        resources["operator_owned_reserved_bytes"] =
            stats.operator_owned_reserved_bytes;
        resources["numeric_update_peak_temporary_bytes"] =
            stats.numeric_update_peak_temporary_bytes;
        resources["resource_generation_active_leases"] =
            stats.resource_generation_active_leases;
        resources["resource_generation_current"] =
            stats.resource_generation_current;
        resources["operator_exclusive_reserved_bytes"] =
            stats.pattern_storage_shared
                ? stats.operator_exclusive_reserved_bytes
                : stats.operator_owned_reserved_bytes;
        resources["pattern_storage_shared"] =
            stats.pattern_storage_shared;
        if (stats.pattern_storage_shared) {
          resources["shared_pattern_operator_references"] =
              stats.shared_pattern_operator_references;
        } else {
          resources["shared_pattern_operator_references"] = py::none();
        }
        resources["sum_operator_owned_bytes_across_operators_safe"] =
            !stats.pattern_storage_shared;
        resources["matrix_descriptor_count"] =
            stats.matrix_descriptor_count;
        resources["dense_vector_descriptor_count"] =
            stats.dense_vector_descriptor_count;
        resources["spmv_handle_count"] = stats.spmv_handle_count;
        resources["opaque_provider_resource_bytes"] = py::none();
        resources["ownership_scope"] =
            stats.pattern_storage_shared
                ? "matrix_values_plan_and_shared_pattern_reference"
                : "matrix_pattern_values_and_persistent_spmv_plan";
        resources["excluded"] =
            "builder_staging_transients_input_output_vectors_and_solver_"
            "workspace";

        py::dict transfers;
        transfers["host_to_device_bytes"] = stats.host_to_device_bytes;
        transfers["device_to_host_bytes"] = stats.device_to_host_bytes;
        transfers["device_to_device_bytes"] =
            stats.device_to_device_bytes;
        transfers["scope"] = "direct_backend_copies_attributed_to_operator";

        py::dict provider;
        provider["name"] = stats.provider_name;
        if (stats.provider_version_major >= 0 &&
            stats.provider_version_minor >= 0 &&
            stats.provider_version_patch >= 0) {
          py::dict version;
          version["major"] = stats.provider_version_major;
          version["minor"] = stats.provider_version_minor;
          version["patch"] = stats.provider_version_patch;
          provider["library_version"] = std::move(version);
        } else {
          provider["library_version"] = py::none();
        }
        provider["bsr_descriptor_available"] =
            stats.provider_bsr_descriptor_available;
        provider["generic_bsr_spmv_available"] =
            stats.provider_generic_bsr_spmv_available;
        provider["selected_storage_format"] = stats.storage_format;
        provider["capability_scope"] =
            "loaded_library_symbols_and_version_not_performance";

        py::dict result;
        result["schema_version"] = 1;
        result["identity"] = std::move(identity);
        result["operations"] = std::move(operations);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        result["provider"] = std::move(provider);
        return result;
      })
      .def("get_data_type", &SparseMatrix::get_data_type);

#define MAKE_SPARSE_MATRIX(TYPE, STORAGE, VTYPE)                             \
  using STORAGE##TYPE##EigenMatrix =                                         \
      Eigen::SparseMatrix<float##TYPE, Eigen::STORAGE>;                      \
  py::class_<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>, SparseMatrix>(   \
      m, #VTYPE #STORAGE "_EigenSparseMatrix")                               \
      .def(py::init<int, int, DataType>())                                   \
      .def(py::init<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix> &>())      \
      .def(py::init<const STORAGE##TYPE##EigenMatrix &>())                   \
      .def(py::self += py::self)                                             \
      .def(py::self + py::self)                                              \
      .def(py::self -= py::self)                                             \
      .def(py::self - py::self)                                              \
      .def(py::self *= float##TYPE())                                        \
      .def(py::self *float##TYPE())                                          \
      .def(float##TYPE() * py::self)                                         \
      .def(py::self *py::self)                                               \
      .def("matmul", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::matmul) \
      .def("spmv", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::spmv)     \
      .def("transpose",                                                      \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::transpose)        \
      .def("get_element",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::get_element<      \
               float##TYPE>)                                                 \
      .def("set_element",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::set_element<      \
               float##TYPE>)                                                 \
      .def("mat_vec_mul",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::mat_vec_mul<      \
               Eigen::VectorX##VTYPE>);

  MAKE_SPARSE_MATRIX(32, ColMajor, f);
  MAKE_SPARSE_MATRIX(32, RowMajor, f);
  MAKE_SPARSE_MATRIX(64, ColMajor, d);
  MAKE_SPARSE_MATRIX(64, RowMajor, d);

  py::class_<CpuSparseCsrMatrix, SparseMatrix>(m, "CpuSparseCsrMatrix")
      .def("spmv", &CpuSparseCsrMatrix::nd_spmv);
  py::class_<CpuSparseBsrMatrix, SparseMatrix>(m, "CpuSparseBsrMatrix")
      .def("spmv", &CpuSparseBsrMatrix::nd_spmv);

  py::class_<CompiledKernelLinearOperator, SparseMatrix>(
      m, "CompiledKernelLinearOperator")
      .def("spmv", &CompiledKernelLinearOperator::nd_spmv)
      .def("update_numeric_data",
           &CompiledKernelLinearOperator::update_numeric_data);

  py::class_<CompiledGraphLinearOperator, SparseMatrix>(
      m, "CompiledGraphLinearOperator")
      .def("spmv", &CompiledGraphLinearOperator::nd_spmv)
      .def(
          "update_numeric_data",
          [](CompiledGraphLinearOperator &operator_, Program *program,
             const py::dict &numeric_args,
             std::uint64_t expected_topology_version,
             std::uint64_t expected_numeric_version) {
            CompiledGraphLinearOperator::NdarrayArguments numeric;
            for (const auto &item : numeric_args) {
              numeric.emplace(py::cast<std::string>(item.first),
                              &py::cast<const Ndarray &>(item.second));
            }
            operator_.update_numeric_arguments(
                program, std::move(numeric), expected_topology_version,
                expected_numeric_version);
          });

  py::class_<CuSparseMatrix, SparseMatrix>(m, "CuSparseMatrix")
      .def(py::init<int, int, DataType>())
      .def(py::init<const CuSparseMatrix &>())
      .def("spmv", &CuSparseMatrix::nd_spmv)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * float32())
      .def(float32() * py::self)
      .def("matmul", &CuSparseMatrix::matmul)
      .def("transpose", &CuSparseMatrix::transpose)
      .def("get_element", &CuSparseMatrix::get_element)
      .def("to_string", &CuSparseMatrix::to_string);

  py::class_<CuSparseBsrMatrix, SparseMatrix>(m, "CuSparseBsrMatrix")
      .def("spmv", &CuSparseBsrMatrix::nd_spmv);

  py::class_<VulkanSparseMatrix, SparseMatrix>(m, "VulkanSparseMatrix")
      .def("spmv", &VulkanSparseMatrix::nd_spmv);
  py::class_<VulkanSparseBsrMatrix, SparseMatrix>(
      m, "VulkanSparseBsrMatrix")
      .def("spmv", &VulkanSparseBsrMatrix::nd_spmv);

  py::class_<VulkanSparseAssemblyPlan>(m, "VulkanSparseAssemblyPlan")
      .def("build", &VulkanSparseAssemblyPlan::build,
           py::call_guard<py::gil_scoped_release>())
      .def("_debug_runtime_stats",
           [](const VulkanSparseAssemblyPlan &plan) {
             const auto stats = plan.debug_runtime_statistics();
             py::dict identity;
             identity["backend_family"] = "vulkan";
             identity["method"] = "radix_sort_segment_reduce_csr";
             identity["rows"] = stats.rows;
             identity["cols"] = stats.cols;
             identity["triplet_capacity"] = stats.capacity;

             py::dict status;
             status["last_status"] = stats.last_status;
             status["last_input_triplets"] = stats.last_input_triplets;
             status["last_unique_nnz"] = stats.last_unique_nnz;
             status["last_duplicate_triplets"] =
                 stats.last_duplicate_triplets;
             status["codes"] = py::dict(
                 "ok"_a = 0, "index_out_of_range"_a = 1,
                 "nonfinite_input"_a = 2,
                 "nonfinite_duplicate_sum"_a = 3,
                 "invalid_device_state"_a = 4,
                 "active_count_exceeds_capacity"_a = 5,
                 "publish_failed"_a = 6);

             py::dict operations;
             operations["build_calls"] = stats.build_calls;
             operations["successful_builds"] = stats.successful_builds;
             operations["failed_builds"] = stats.failed_builds;
             operations["workspace_builds"] = stats.workspace_builds;
             operations["workspace_reuses"] = stats.workspace_reuses;
             operations["workspace_growth_synchronizations"] =
                 stats.workspace_growth_synchronizations;
             operations["host_synchronizations"] =
                 stats.host_synchronizations;
             operations["host_control_readbacks"] =
                 stats.host_control_readbacks;
             operations["host_scalar_readbacks"] =
                 stats.host_scalar_readbacks;

             py::dict resources;
             resources["persistent_workspace_reserved_bytes"] =
                 stats.persistent_workspace_reserved_bytes;
             resources["shared_radix_sort_workspace_reserved_bytes"] =
                 stats.shared_radix_sort_workspace_reserved_bytes;
             resources["shared_scan_workspace_reserved_bytes"] =
                 stats.shared_scan_workspace_reserved_bytes;
             resources["last_output_pattern_bytes"] =
                 stats.last_output_pattern_bytes;
             resources["last_output_value_bytes"] =
                 stats.last_output_value_bytes;
             resources["workspace_ownership"] =
                 "plan_staging_plus_program_shared_sort_scan";

             py::dict transfers;
             transfers["device_to_host_bytes"] =
                 stats.device_to_host_bytes;
             transfers["device_to_device_bytes"] =
                 stats.device_to_device_bytes;
             transfers["host_to_device_bytes"] = 0;
             transfers["device_payload_readback_bytes"] = 0;

             py::dict contract;
             contract["fixed_capacity"] = true;
             contract["device_resident_triplet_payload"] = true;
             contract["transactional_publish"] = true;
             contract["exact_sized_published_csr"] = true;
             contract["sorted_unique_columns_per_row"] = true;
             contract["duplicate_reduce_order"] =
                 "sorted_segment_sequential";
             contract["control_readback_bytes_per_build"] = 8;
             contract["public_sparse_builder"] = false;
             contract["bsr_output"] = false;

             py::dict result;
             result["schema_version"] = 1;
             result["identity"] = std::move(identity);
             result["status"] = std::move(status);
             result["operations"] = std::move(operations);
             result["resources"] = std::move(resources);
             result["transfers"] = std::move(transfers);
             result["contract"] = std::move(contract);
             return result;
           });
  m.def("_make_vulkan_sparse_assembly_plan",
        [](Program *program, int rows, int cols, int capacity) {
          return std::make_unique<VulkanSparseAssemblyPlan>(
              program, rows, cols, capacity);
        });

  py::class_<CudaSparseAssemblyPlan>(m, "CudaSparseAssemblyPlan")
      .def("build", &CudaSparseAssemblyPlan::build,
           py::call_guard<py::gil_scoped_release>())
      .def("_debug_runtime_stats",
           [](const CudaSparseAssemblyPlan &plan) {
             const auto stats = plan.debug_runtime_statistics();
             py::dict identity;
             identity["backend_family"] = "cuda";
             identity["method"] = "radix_sort_segment_reduce_csr";
             identity["rows"] = stats.rows;
             identity["cols"] = stats.cols;
             identity["triplet_capacity"] = stats.capacity;

             py::dict status;
             status["last_status"] = stats.last_status;
             status["last_input_triplets"] = stats.last_input_triplets;
             status["last_unique_nnz"] = stats.last_unique_nnz;
             status["last_duplicate_triplets"] =
                 stats.last_duplicate_triplets;
             status["codes"] = py::dict(
                 "ok"_a = 0, "index_out_of_range"_a = 1,
                 "nonfinite_input"_a = 2,
                 "nonfinite_duplicate_sum"_a = 3,
                 "invalid_device_state"_a = 4,
                 "active_count_out_of_range"_a = 5,
                 "publish_failed"_a = 6);

             py::dict operations;
             operations["build_calls"] = stats.build_calls;
             operations["successful_builds"] = stats.successful_builds;
             operations["failed_builds"] = stats.failed_builds;
             operations["workspace_builds"] = stats.workspace_builds;
             operations["workspace_reuses"] = stats.workspace_reuses;
             operations["workspace_growth_synchronizations"] =
                 stats.workspace_growth_synchronizations;
             operations["host_synchronizations"] =
                 stats.host_synchronizations;
             operations["host_control_readbacks"] =
                 stats.host_control_readbacks;
             operations["host_scalar_readbacks"] =
                 stats.host_scalar_readbacks;

             py::dict resources;
             resources["persistent_workspace_reserved_bytes"] =
                 stats.persistent_workspace_reserved_bytes;
             resources["shared_radix_sort_workspace_reserved_bytes"] =
                 stats.shared_radix_sort_workspace_reserved_bytes;
             resources["shared_scan_workspace_reserved_bytes"] =
                 stats.shared_scan_workspace_reserved_bytes;
             resources["last_output_pattern_bytes"] =
                 stats.last_output_pattern_bytes;
             resources["last_output_value_bytes"] =
                 stats.last_output_value_bytes;
             resources["workspace_ownership"] =
                 "plan_staging_plus_program_shared_sort_scan";

             py::dict transfers;
             transfers["device_to_host_bytes"] =
                 stats.device_to_host_bytes;
             transfers["device_to_device_bytes"] =
                 stats.device_to_device_bytes;
             transfers["host_to_device_bytes"] = 0;
             transfers["device_payload_readback_bytes"] = 0;

             py::dict contract;
             contract["fixed_capacity"] = true;
             contract["device_active_count"] = true;
             contract["empty_matrix"] = true;
             contract["device_resident_triplet_payload"] = true;
             contract["transactional_publish"] = true;
             contract["exact_sized_published_csr"] = true;
             contract["sorted_unique_columns_per_row"] = true;
             contract["duplicate_reduce_order"] =
                 "stable_sorted_segment_sequential";
             contract["control_readback_bytes_per_build"] = 8;
             contract["public_sparse_builder"] = true;
             contract["bsr_output"] = false;
             contract["cuda_toolkit_required"] = false;

             py::dict result;
             result["schema_version"] = 1;
             result["identity"] = std::move(identity);
             result["status"] = std::move(status);
             result["operations"] = std::move(operations);
             result["resources"] = std::move(resources);
             result["transfers"] = std::move(transfers);
             result["contract"] = std::move(contract);
             return result;
           });
  m.def("_make_cuda_sparse_assembly_plan",
        [](Program *program, int rows, int cols, int capacity) {
          return std::make_unique<CudaSparseAssemblyPlan>(
              program, rows, cols, capacity);
        });

  py::class_<SparseSolver>(m, "SparseSolver")
      .def("compute", &SparseSolver::compute)
      .def("analyze_pattern", &SparseSolver::analyze_pattern)
      .def("factorize", &SparseSolver::factorize)
      .def("validate_factorization", &SparseSolver::validate_factorization)
      .def("info", &SparseSolver::info);

#define REGISTER_EIGEN_SOLVER(dt, type, order, fd)                           \
  py::class_<EigenSparseSolver##dt##type##order, SparseSolver>(              \
      m, "EigenSparseSolver" #dt #type #order)                               \
      .def("compute", &EigenSparseSolver##dt##type##order::compute)          \
      .def("analyze_pattern",                                                \
           &EigenSparseSolver##dt##type##order::analyze_pattern)             \
      .def("factorize", &EigenSparseSolver##dt##type##order::factorize)      \
      .def("solve",                                                          \
           &EigenSparseSolver##dt##type##order::solve<Eigen::VectorX##fd>)   \
      .def("solve_rf",                                                       \
           &EigenSparseSolver##dt##type##order::solve_rf<Eigen::VectorX##fd, \
                                                         dt>)                \
      .def("info", &EigenSparseSolver##dt##type##order::info);

  REGISTER_EIGEN_SOLVER(float32, LLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float64, LLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, COLAMD, d)

  py::class_<CuSparseSolver, SparseSolver>(m, "CuSparseSolver")
      .def("compute", &CuSparseSolver::compute)
      .def("analyze_pattern", &CuSparseSolver::analyze_pattern)
      .def("factorize", &CuSparseSolver::factorize)
      .def("solve_rf", &CuSparseSolver::solve_rf)
      .def("info", &CuSparseSolver::info);

  m.def("make_sparse_solver", &make_sparse_solver);
  m.def("make_cusparse_solver", &make_cusparse_solver);

  // Conjugate Gradient solver
  auto sparse_solve_result_to_dict = [](const SparseSolveResult &result) {
    const char *termination_reason = "not_run";
    switch (result.status) {
      case SparseSolveStatus::kMaxIterations:
        termination_reason = "max_iterations";
        break;
      case SparseSolveStatus::kBreakdown:
        termination_reason = "breakdown";
        break;
      case SparseSolveStatus::kConverged:
        termination_reason = "converged";
        break;
      case SparseSolveStatus::kNotRun:
        break;
    }
    py::dict snapshot;
    snapshot["status_code"] = static_cast<int>(result.status);
    snapshot["termination_reason"] = termination_reason;
    snapshot["converged"] = result.converged();
    snapshot["breakdown"] = result.breakdown();
    snapshot["reached_max_iterations"] =
        result.reached_max_iterations();
    snapshot["iterations"] = result.iterations;
    snapshot["initial_residual_norm"] = result.initial_residual_norm;
    snapshot["residual_norm"] = result.residual_norm;
    snapshot["absolute_tolerance"] = result.absolute_tolerance;
    snapshot["relative_tolerance"] = result.relative_tolerance;
    snapshot["relative_reference_norm"] =
        result.relative_reference_norm;
    snapshot["effective_tolerance"] = result.effective_tolerance;
    return snapshot;
  };
  auto sparse_solve_plan_stats_to_dict =
      [](const SparseSolvePlanRuntimeStatistics &stats) {
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["method"] = stats.method;
        identity["dtype"] = stats.dtype;
        identity["rows"] = stats.rows;
        identity["cols"] = stats.cols;
        identity["max_iterations"] = stats.max_iterations;
        identity["absolute_tolerance"] = stats.absolute_tolerance;
        identity["relative_tolerance"] = stats.relative_tolerance;
        identity["last_relative_reference_norm"] =
            stats.last_relative_reference_norm;
        identity["last_effective_tolerance"] =
            stats.last_effective_tolerance;
        identity["preconditioner_method"] =
            stats.preconditioner_method;
        identity["operator_action_provider"] =
            stats.operator_action_provider;
        identity["operator_asynchronous_submit"] =
            stats.operator_asynchronous_submit;
        identity["operator_execution_kind"] =
            stats.operator_execution_kind;
        identity["operator_backend_execution_path"] =
            stats.operator_backend_execution_path;
        identity["solver_execution_policy"] =
            stats.solver_execution_policy;
        identity["host_check_interval"] = stats.host_check_interval;
        identity["solver_graph_enabled"] =
            stats.solver_graph_enabled;
        identity["preconditioner_action_provider"] =
            stats.preconditioner_action_provider;
        identity["preconditioner_behavior"] =
            stats.preconditioner_behavior;
        identity["preconditioner_asynchronous_submit"] =
            stats.preconditioner_asynchronous_submit;
        identity["operator_pattern_version"] =
            stats.operator_pattern_version;
        identity["operator_numeric_version"] =
            stats.operator_numeric_version;
        identity["last_solve_pattern_version"] =
            stats.last_solve_pattern_version;
        identity["last_solve_numeric_version"] =
            stats.last_solve_numeric_version;
        identity["operator_pattern_changed_since_last_solve"] =
            stats.operator_pattern_changed_since_last_solve;
        identity["operator_numeric_changed_since_last_solve"] =
            stats.operator_numeric_changed_since_last_solve;

        py::dict operations;
        operations["solve_calls"] = stats.solve_calls;
        operations["total_iterations"] = stats.total_iterations;
        operations["workspace_builds"] = stats.workspace_builds;
        operations["workspace_reuses"] = stats.workspace_reuses;
        operations["operator_apply_calls"] =
            stats.operator_apply_calls_available
                ? py::cast(stats.operator_apply_calls)
                : py::none();
        operations["host_scalar_reductions"] =
            stats.host_scalar_reductions;
        operations["device_scalar_operations"] =
            stats.device_scalar_operations;
        operations["host_scalar_readbacks"] =
            stats.host_scalar_readbacks;
        operations["host_synchronizations"] =
            stats.host_synchronizations;
        operations["fixed_iteration_only"] =
            stats.fixed_iteration_only;
        operations["bounded_masked_execution"] =
            stats.bounded_masked_execution;
        operations["preconditioner_apply_calls"] =
            stats.preconditioner_apply_calls_available
                ? py::cast(stats.preconditioner_apply_calls)
                : py::none();
        operations["operator_generation_pins"] =
            stats.operator_generation_pins;
        operations["operator_generation_changes"] =
            stats.operator_generation_changes;
        operations["operator_numeric_generation_changes"] =
            stats.operator_numeric_generation_changes;
        operations["operator_binding_generation_changes"] =
            stats.operator_binding_generation_changes;
        operations["operator_plan_invalidations"] =
            stats.operator_plan_invalidations;
        operations["operator_execution_plan_builds"] =
            stats.operator_execution_plan_builds;
        operations["operator_execution_plan_reuses"] =
            stats.operator_execution_plan_reuses;
        operations["operator_binding_rebinds"] =
            stats.operator_binding_rebinds;
        operations["operator_sequence_submissions"] =
            stats.operator_sequence_submissions;
        operations["operator_compiled_graph_submissions"] =
            stats.operator_compiled_graph_submissions;
        operations["operator_runtime_capture_submissions"] =
            stats.operator_runtime_capture_submissions;
        operations["operator_backend_captures"] =
            stats.operator_backend_captures;
        operations["operator_backend_replays"] =
            stats.operator_backend_replays;
        operations["operator_ordinary_fallbacks"] =
            stats.operator_ordinary_fallbacks;
        operations["operator_cache_invalidations"] =
            stats.operator_cache_invalidations;
        operations["preconditioner_generation_pins"] =
            stats.preconditioner_generation_pins;
        operations["preconditioner_generation_changes"] =
            stats.preconditioner_generation_changes;
        operations["preconditioner_numeric_generation_changes"] =
            stats.preconditioner_numeric_generation_changes;
        operations["preconditioner_binding_generation_changes"] =
            stats.preconditioner_binding_generation_changes;
        operations["preconditioner_plan_invalidations"] =
            stats.preconditioner_plan_invalidations;
        operations["preconditioner_setup_calls"] =
            stats.preconditioner_setup_calls;
        operations["preconditioner_update_calls"] =
            stats.preconditioner_update_calls;
        operations["preconditioner_update_successes"] =
            stats.preconditioner_update_successes;
        operations["preconditioner_update_noops"] =
            stats.preconditioner_update_noops;
        operations["preconditioner_update_failures"] =
            stats.preconditioner_update_failures;

        py::dict resources;
        resources["persistent_vector_count"] =
            stats.persistent_vector_count;
        resources["persistent_vector_reserved_bytes"] =
            stats.persistent_vector_reserved_bytes;
        resources["persistent_scalar_count"] =
            stats.persistent_scalar_count;
        resources["persistent_scalar_reserved_bytes"] =
            stats.persistent_scalar_reserved_bytes;
        resources["cublas_handle_count"] = stats.cublas_handle_count;
        resources["external_preconditioner"] =
            stats.external_preconditioner;
        resources["preconditioner_ownership_scope"] =
            stats.preconditioner_ownership_scope;
        resources["opaque_provider_resource_bytes"] = py::none();
        resources["solver_state_rebuilt_each_solve"] =
            stats.solver_state_rebuilt_each_solve;
        resources["transient_solver_workspace_bytes"] =
            stats.transient_solver_workspace_bytes_available
                ? py::cast(stats.transient_solver_workspace_bytes)
                : py::none();
        resources["ownership_scope"] =
            "solve_plan_vectors_scalars_and_provider_handle";
        resources["excluded"] =
            "operator_resources_rhs_solution_and_caller_vectors";

        py::dict transfers;
        transfers["device_to_device_bytes"] =
            stats.device_to_device_bytes;
        transfers["device_to_host_bytes"] =
            stats.device_to_host_bytes;
        transfers["host_to_device_bytes"] =
            stats.host_to_device_bytes;
        transfers["scope"] = "copies_issued_directly_by_solve_plan";

        py::dict result;
        result["schema_version"] = 1;
        result["identity"] = std::move(identity);
        result["operations"] = std::move(operations);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        return result;
      };
  auto sparse_preconditioner_stats_to_dict =
      [](const SparsePreconditionerPlanRuntimeStatistics &stats) {
        py::dict identity;
        identity["backend_family"] = stats.backend_family;
        identity["method"] = stats.method;
        identity["dtype"] = stats.dtype;
        identity["rows"] = stats.rows;
        if (stats.block_size > 0) {
          identity["block_rows"] = stats.block_rows;
          identity["block_size"] = stats.block_size;
        } else {
          identity["block_rows"] = py::none();
          identity["block_size"] = py::none();
        }
        identity["operator_pattern_version_at_build"] =
            stats.operator_pattern_version_at_build;
        identity["operator_numeric_version_at_build"] =
            stats.operator_numeric_version_at_build;
        identity["operator_pattern_version_current"] =
            stats.operator_pattern_version_current;
        identity["operator_numeric_version_current"] =
            stats.operator_numeric_version_current;
        identity["operator_stale"] = stats.operator_stale;
        identity["preconditioner_pattern_version_at_build"] =
            stats.preconditioner_pattern_version_at_build;
        identity["preconditioner_numeric_version_at_build"] =
            stats.preconditioner_numeric_version_at_build;
        identity["preconditioner_pattern_version_current"] =
            stats.preconditioner_pattern_version_current;
        identity["preconditioner_numeric_version_current"] =
            stats.preconditioner_numeric_version_current;
        identity["preconditioner_stale"] = stats.preconditioner_stale;

        py::dict operations;
        operations["apply_calls"] = stats.apply_calls;
        operations["numeric_refresh_calls"] =
            stats.numeric_refresh_calls;
        operations["numeric_refresh_successes"] =
            stats.numeric_refresh_successes;
        operations["numeric_refresh_noops"] =
            stats.numeric_refresh_noops;
        operations["numeric_refresh_failures"] =
            stats.numeric_refresh_failures;

        py::dict resources;
        resources["persistent_inverse_count"] =
            stats.persistent_inverse_count;
        resources["persistent_inverse_reserved_bytes"] =
            stats.persistent_inverse_reserved_bytes;
        resources["refresh_peak_temporary_host_bytes"] =
            stats.refresh_peak_temporary_host_bytes;
        resources["refresh_peak_temporary_device_bytes"] =
            stats.refresh_peak_temporary_device_bytes;
        const bool external_inverse_operator =
            stats.method == "compiled_kernel_inverse_apply" ||
            stats.method == "compiled_graph_inverse_apply";
        resources["ownership_scope"] =
            external_inverse_operator ? "external_inverse_operator"
                                      : "preconditioner_inverse";
        resources["excluded"] =
            external_inverse_operator
                ? "target_inverse_operators_input_output_and_program_shared_cache"
                : "operator_input_output_and_program_shared_cache";

        py::dict transfers;
        transfers["construction_device_to_host_bytes"] =
            stats.construction_device_to_host_bytes;
        transfers["construction_host_to_device_bytes"] =
            stats.construction_host_to_device_bytes;
        transfers["construction_host_synchronizations"] =
            stats.construction_host_synchronizations;
        transfers["refresh_device_to_host_bytes"] =
            stats.refresh_device_to_host_bytes;
        transfers["refresh_host_to_device_bytes"] =
            stats.refresh_host_to_device_bytes;
        transfers["refresh_host_synchronizations"] =
            stats.refresh_host_synchronizations;
        transfers["apply_host_transfer_bytes"] = 0;

        py::dict contract;
        contract["fixed_csr_only"] = stats.method == "jacobi";
        contract["fixed_bsr_only"] = stats.method == "block_jacobi";
        contract["in_place_apply_supported"] =
            stats.in_place_apply_supported;
        contract["numeric_refresh_supported"] =
            stats.numeric_refresh_supported;
        contract["numeric_update_requires_refresh"] =
            stats.numeric_refresh_supported;
        contract["numeric_update_requires_rebuild"] =
            !stats.numeric_refresh_supported;
        contract["pattern_update_requires_rebuild"] = true;
        contract["public_solver_integration"] = false;

        py::dict result;
        result["schema_version"] = 2;
        result["identity"] = std::move(identity);
        result["operations"] = std::move(operations);
        result["resources"] = std::move(resources);
        result["transfers"] = std::move(transfers);
        result["contract"] = std::move(contract);
        return result;
      };

  py::class_<ExperimentalLinearOperatorHandle>(
      m, "ExperimentalLinearOperatorHandle")
      .def("_apply", &ExperimentalLinearOperatorHandle::apply,
           py::keep_alive<1, 2>(), py::arg("program"),
           py::arg("input"), py::arg("output"))
      .def("_metadata", [](const ExperimentalLinearOperatorHandle &handle) {
        const auto &descriptor = handle.descriptor();
        const auto &capabilities = handle.capabilities();
        const auto &traits = handle.mathematical_traits();
        const auto stamp = handle.resource_stamp();
        const auto claim_to_dict = [](const OperatorTraitClaim &claim) {
          py::dict result;
          result["known"] = claim.known();
          result["value"] = claim.known()
                                  ? py::cast(claim.value)
                                  : py::none();
          const char *provenance = "unspecified";
          switch (claim.provenance) {
            case OperatorTraitProvenance::asserted_by_user:
              provenance = "asserted_by_user";
              break;
            case OperatorTraitProvenance::derived_structurally:
              provenance = "derived_structurally";
              break;
            case OperatorTraitProvenance::constructed_by_framework:
              provenance = "constructed_by_framework";
              break;
            case OperatorTraitProvenance::empirically_checked:
              provenance = "empirically_checked";
              break;
            case OperatorTraitProvenance::unspecified:
              break;
          }
          result["provenance"] = provenance;
          result["validity_scope"] = claim.validity_scope;
          return result;
        };
        py::dict capabilities_dict;
        capabilities_dict["forward_apply"] = capabilities.forward_apply;
        capabilities_dict["adjoint_apply"] = capabilities.adjoint_apply;
        capabilities_dict["native_generalized_apply"] =
            capabilities.native_generalized_apply;
        capabilities_dict["asynchronous_submit"] =
            capabilities.asynchronous_submit;
        capabilities_dict["explicit_sequence"] =
            capabilities.explicit_sequence;
        capabilities_dict["compiled_graph"] = capabilities.compiled_graph;
        capabilities_dict["runtime_capture"] = capabilities.runtime_capture;
        capabilities_dict["binding_rebind"] = capabilities.binding_rebind;
        capabilities_dict["persistent_workspace"] =
            capabilities.persistent_workspace;
        py::dict traits_dict;
        traits_dict["self_adjoint"] =
            claim_to_dict(traits.self_adjoint);
        traits_dict["positive_definite"] =
            claim_to_dict(traits.positive_definite);
        traits_dict["positive_semidefinite"] =
            claim_to_dict(traits.positive_semidefinite);
        traits_dict["singular"] = claim_to_dict(traits.singular);
        py::dict resource_stamp;
        resource_stamp["program_generation"] = stamp.program_generation;
        resource_stamp["schema_revision"] = stamp.schema_revision;
        resource_stamp["topology_revision"] = stamp.topology_revision;
        resource_stamp["numeric_revision"] = stamp.numeric_revision;
        resource_stamp["binding_revision"] = stamp.binding_revision;
        py::dict result;
        result["schema_version"] = 1;
        result["provider"] = handle.provider_name();
        result["dtype"] = data_type_name(descriptor.domain.scalar_type);
        result["shape"] = py::make_tuple(
            descriptor.range.scalar_extent,
            descriptor.domain.scalar_extent);
        result["entry_shape"] = descriptor.domain.entry_shape;
        result["execution_kind"] =
            operator_execution_kind_name(handle.execution_kind());
        result["capabilities"] = std::move(capabilities_dict);
        result["traits"] = std::move(traits_dict);
        result["resource_stamp"] = std::move(resource_stamp);
        return result;
      })
      .def("_debug_runtime_stats",
           [](const ExperimentalLinearOperatorHandle &handle) {
             const auto stats = handle.debug_runtime_statistics();
             py::dict result;
             result["schema_version"] = 1;
             result["provider"] = handle.provider_name();
             result["execution_kind"] =
                 operator_execution_kind_name(stats.execution_kind);
             result["submissions"] = stats.submissions;
             result["primitive_apply_calls"] =
                 stats.primitive_apply_calls;
             result["generalized_lowerings"] =
                 stats.generalized_lowerings;
             result["scratch_builds"] = stats.scratch_builds;
             result["scratch_reuses"] = stats.scratch_reuses;
             result["scratch_reserved_bytes"] =
                 stats.scratch_reserved_bytes;
             result["generation_pins"] = stats.generation_pins;
             result["generation_changes"] = stats.generation_changes;
             result["invalidations"] = stats.invalidations;
             result["execution_plan_builds"] =
                 stats.execution_plan_builds;
             result["execution_plan_reuses"] =
                 stats.execution_plan_reuses;
             result["binding_rebinds"] = stats.binding_rebinds;
             result["backend_path"] =
                 operator_backend_execution_path_name(
                     stats.last_backend_path);
             return result;
           });

  m.def(
      "_make_experimental_linear_operator",
      [](Program *program, SparseMatrix &matrix, int self_adjoint,
         int positive_definite, int positive_semidefinite, int singular) {
        return make_experimental_linear_operator_handle(
            program, matrix,
            make_asserted_operator_traits(
                self_adjoint, positive_definite,
                positive_semidefinite, singular));
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("self_adjoint") = -1,
      py::arg("positive_definite") = -1,
      py::arg("positive_semidefinite") = -1,
      py::arg("singular") = -1);
  m.def(
      "_make_experimental_identity_operator",
      [](Program *program, DataType dtype, std::size_t size) {
        return make_experimental_identity_operator_handle(
            program, OperatorSpaceDesc{dtype, size});
      },
      py::keep_alive<0, 1>(), py::arg("program"),
      py::arg("dtype"), py::arg("size"));
  m.def("_make_experimental_adjoint_operator",
        make_experimental_adjoint_operator_handle,
        py::keep_alive<0, 1>(), py::arg("operand"));
  m.def("_make_experimental_scaled_operator",
        make_experimental_scaled_operator_handle,
        py::keep_alive<0, 2>(), py::arg("scale"), py::arg("operand"));
  m.def("_make_experimental_sum_operator",
        make_experimental_sum_operator_handle,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("left"), py::arg("right"));
  m.def("_make_experimental_composed_operator",
        make_experimental_composed_operator_handle,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::arg("outer"), py::arg("inner"));
  m.def(
      "_make_experimental_block_diagonal_operator",
      [](const py::iterable &items) {
        std::vector<ExperimentalLinearOperatorHandle *> blocks;
        for (const auto &item : items) {
          blocks.push_back(
              py::cast<ExperimentalLinearOperatorHandle *>(item));
        }
        return make_experimental_block_diagonal_operator_handle(blocks);
      },
      py::arg("blocks"));

  py::class_<SparseJacobiPreconditionerPlan>(
      m, "SparseJacobiPreconditionerPlan")
      .def("_refresh_numeric",
           &SparseJacobiPreconditionerPlan::refresh_numeric)
      .def("apply", &SparseJacobiPreconditionerPlan::apply)
      .def("_debug_runtime_stats",
           [sparse_preconditioner_stats_to_dict](
               const SparseJacobiPreconditionerPlan &plan) {
             return sparse_preconditioner_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_sparse_jacobi_preconditioner_plan",
        make_sparse_jacobi_preconditioner_plan, py::keep_alive<0, 1>(),
        py::keep_alive<0, 2>());
  py::class_<SparseBlockJacobiPreconditionerPlan>(
      m, "SparseBlockJacobiPreconditionerPlan")
      .def("apply", &SparseBlockJacobiPreconditionerPlan::apply)
      .def("_refresh_numeric",
           &SparseBlockJacobiPreconditionerPlan::refresh_numeric)
      .def("_debug_runtime_stats",
           [sparse_preconditioner_stats_to_dict](
               const SparseBlockJacobiPreconditionerPlan &plan) {
             return sparse_preconditioner_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_sparse_block_jacobi_preconditioner_plan",
        make_sparse_block_jacobi_preconditioner_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
  py::class_<CompiledKernelPreconditionerPlan>(
      m, "CompiledKernelPreconditionerPlan")
      .def("apply", &CompiledKernelPreconditionerPlan::apply)
      .def("_debug_runtime_stats",
           [sparse_preconditioner_stats_to_dict](
               const CompiledKernelPreconditionerPlan &plan) {
             return sparse_preconditioner_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_compiled_kernel_preconditioner_plan",
        make_compiled_kernel_preconditioner_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>(), py::arg("program"),
        py::arg("target_operator"), py::arg("inverse_apply_operator"),
        py::arg("assume_symmetric_positive_definite"));
  py::class_<CG<Eigen::VectorXf, float>>(m, "CGf")
      .def(py::init<SparseMatrix &, int, float, bool>())
      .def("solve", &CG<Eigen::VectorXf, float>::solve)
      .def("set_x", &CG<Eigen::VectorXf, float>::set_x)
      .def("reset_x", &CG<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &CG<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &CG<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray", &CG<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success", &CG<Eigen::VectorXf, float>::is_success)
      .def("get_status", &CG<Eigen::VectorXf, float>::get_status)
      .def("get_iterations", &CG<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &CG<Eigen::VectorXf, float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &CG<Eigen::VectorXf, float>::get_residual_norm)
      .def("_get_last_result", [sparse_solve_result_to_dict](
                                    const CG<Eigen::VectorXf, float> &cg) {
        return sparse_solve_result_to_dict(cg.get_last_result());
      })
      .def("_debug_runtime_stats", [sparse_solve_plan_stats_to_dict](
                                         const CG<Eigen::VectorXf, float> &cg) {
        return sparse_solve_plan_stats_to_dict(
            cg.debug_runtime_statistics());
      });
  py::class_<CG<Eigen::VectorXd, double>>(m, "CGd")
      .def(py::init<SparseMatrix &, int, double, bool>())
      .def("solve", &CG<Eigen::VectorXd, double>::solve)
      .def("set_x", &CG<Eigen::VectorXd, double>::set_x)
      .def("reset_x", &CG<Eigen::VectorXd, double>::reset_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXd, double>::set_x_ndarray)
      .def("get_x", &CG<Eigen::VectorXd, double>::get_x)
      .def("set_b_ndarray", &CG<Eigen::VectorXd, double>::set_b_ndarray)
      .def("set_b", &CG<Eigen::VectorXd, double>::set_b)
      .def("is_success", &CG<Eigen::VectorXd, double>::is_success)
      .def("get_status", &CG<Eigen::VectorXd, double>::get_status)
      .def("get_iterations", &CG<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &CG<Eigen::VectorXd, double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &CG<Eigen::VectorXd, double>::get_residual_norm)
      .def("_get_last_result", [sparse_solve_result_to_dict](
                                    const CG<Eigen::VectorXd, double> &cg) {
        return sparse_solve_result_to_dict(cg.get_last_result());
      })
      .def("_debug_runtime_stats", [sparse_solve_plan_stats_to_dict](
                                         const CG<Eigen::VectorXd, double> &cg) {
        return sparse_solve_plan_stats_to_dict(
            cg.debug_runtime_statistics());
      });
  m.def(
      "make_float_cg_solver",
      [](SparseMatrix &A, int max_iters, float absolute_tolerance,
         bool verbose, float relative_tolerance) {
        return make_cg_solver<Eigen::VectorXf, float>(
            A, max_iters, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "make_double_cg_solver",
      [](SparseMatrix &A, int max_iters, double absolute_tolerance,
         bool verbose, double relative_tolerance) {
        return make_cg_solver<Eigen::VectorXd, double>(
            A, max_iters, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<SparseBiCGSTAB<Eigen::VectorXf, float>>(
      m, "SparseBiCGSTABf")
      .def("solve", &SparseBiCGSTAB<Eigen::VectorXf, float>::solve)
      .def("set_x", &SparseBiCGSTAB<Eigen::VectorXf, float>::set_x)
      .def("reset_x", &SparseBiCGSTAB<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &SparseBiCGSTAB<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &SparseBiCGSTAB<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::is_success)
      .def("get_status",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::get_status)
      .def("get_iterations",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXf,
                           float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXf, float>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<SparseBiCGSTAB<Eigen::VectorXd, double>>(
      m, "SparseBiCGSTABd")
      .def("solve", &SparseBiCGSTAB<Eigen::VectorXd, double>::solve)
      .def("set_x", &SparseBiCGSTAB<Eigen::VectorXd, double>::set_x)
      .def("reset_x", &SparseBiCGSTAB<Eigen::VectorXd, double>::reset_x)
      .def("get_x", &SparseBiCGSTAB<Eigen::VectorXd, double>::get_x)
      .def("set_x_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::set_x_ndarray)
      .def("set_b", &SparseBiCGSTAB<Eigen::VectorXd, double>::set_b)
      .def("set_b_ndarray",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::set_b_ndarray)
      .def("is_success",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::is_success)
      .def("get_status",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::get_status)
      .def("get_iterations",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXd,
                           double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseBiCGSTAB<Eigen::VectorXd, double>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "make_float_sparse_bicgstab_solver",
      [](SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<SparseBiCGSTAB<Eigen::VectorXf, float>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "make_double_sparse_bicgstab_solver",
      [](SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<SparseBiCGSTAB<Eigen::VectorXd, double>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<FixedSparseBiCGSTAB<Eigen::VectorXf, float>>(
      m, "FixedSparseBiCGSTABf")
      .def("solve", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::solve)
      .def("solve_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::solve_ndarray)
      .def("set_x", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_x)
      .def("reset_x",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::is_success)
      .def("get_status",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::get_status)
      .def("get_iterations",
           &FixedSparseBiCGSTAB<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXf,
                                float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXf,
                                float>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<FixedSparseBiCGSTAB<Eigen::VectorXd, double>>(
      m, "FixedSparseBiCGSTABd")
      .def("solve", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::solve)
      .def("solve_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::solve_ndarray)
      .def("set_x", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_x)
      .def("reset_x",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::reset_x)
      .def("get_x", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::get_x)
      .def("set_x_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_x_ndarray)
      .def("set_b", &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_b)
      .def("set_b_ndarray",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::set_b_ndarray)
      .def("is_success",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::is_success)
      .def("get_status",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::get_status)
      .def("get_iterations",
           &FixedSparseBiCGSTAB<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXd,
                                double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &FixedSparseBiCGSTAB<Eigen::VectorXd,
                                double>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const FixedSparseBiCGSTAB<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "_make_float_cpu_fixed_sparse_bicgstab_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXf, float>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_fixed_sparse_bicgstab_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXd, double>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_float_cpu_experimental_linear_operator_bicgstab_solver",
      [](Program *program, ExperimentalLinearOperatorHandle &operator_handle,
         int max_iterations, float absolute_tolerance,
         float relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "BiCGSTAB operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXf, float>>(
            program, operator_handle.binding(), max_iterations,
            absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_experimental_linear_operator_bicgstab_solver",
      [](Program *program, ExperimentalLinearOperatorHandle &operator_handle,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        TI_ERROR_IF(operator_handle.program() != program,
                    "BiCGSTAB operator belongs to a different Program.");
        return std::make_unique<
            FixedSparseBiCGSTAB<Eigen::VectorXd, double>>(
            program, operator_handle.binding(), max_iterations,
            absolute_tolerance, false, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<SparseMINRES<Eigen::VectorXf, float>>(m, "SparseMINRESf")
      .def("solve", &SparseMINRES<Eigen::VectorXf, float>::solve)
      .def("set_x", &SparseMINRES<Eigen::VectorXf, float>::set_x)
      .def("reset_x", &SparseMINRES<Eigen::VectorXf, float>::reset_x)
      .def("get_x", &SparseMINRES<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray",
           &SparseMINRES<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &SparseMINRES<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray",
           &SparseMINRES<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success", &SparseMINRES<Eigen::VectorXf, float>::is_success)
      .def("get_status", &SparseMINRES<Eigen::VectorXf, float>::get_status)
      .def("get_iterations",
           &SparseMINRES<Eigen::VectorXf, float>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseMINRES<Eigen::VectorXf,
                         float>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseMINRES<Eigen::VectorXf, float>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseMINRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseMINRES<Eigen::VectorXf, float> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  py::class_<SparseMINRES<Eigen::VectorXd, double>>(m, "SparseMINRESd")
      .def("solve", &SparseMINRES<Eigen::VectorXd, double>::solve)
      .def("set_x", &SparseMINRES<Eigen::VectorXd, double>::set_x)
      .def("reset_x", &SparseMINRES<Eigen::VectorXd, double>::reset_x)
      .def("get_x", &SparseMINRES<Eigen::VectorXd, double>::get_x)
      .def("set_x_ndarray",
           &SparseMINRES<Eigen::VectorXd, double>::set_x_ndarray)
      .def("set_b", &SparseMINRES<Eigen::VectorXd, double>::set_b)
      .def("set_b_ndarray",
           &SparseMINRES<Eigen::VectorXd, double>::set_b_ndarray)
      .def("is_success", &SparseMINRES<Eigen::VectorXd, double>::is_success)
      .def("get_status", &SparseMINRES<Eigen::VectorXd, double>::get_status)
      .def("get_iterations",
           &SparseMINRES<Eigen::VectorXd, double>::get_iterations)
      .def("get_initial_residual_norm",
           &SparseMINRES<Eigen::VectorXd,
                         double>::get_initial_residual_norm)
      .def("get_residual_norm",
           &SparseMINRES<Eigen::VectorXd, double>::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](
               const SparseMINRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_result_to_dict(solver.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const SparseMINRES<Eigen::VectorXd, double> &solver) {
             return sparse_solve_plan_stats_to_dict(
                 solver.debug_runtime_statistics());
           });
  m.def(
      "make_float_sparse_minres_solver",
      [](SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXf, float>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "make_double_sparse_minres_solver",
      [](SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXd, double>>(
            matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_float_cpu_fixed_sparse_minres_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXf, float>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_double_cpu_fixed_sparse_minres_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, bool verbose,
         double relative_tolerance) {
        return std::make_unique<SparseMINRES<Eigen::VectorXd, double>>(
            program, matrix, max_iterations, absolute_tolerance, verbose,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0);

  py::class_<CUCG>(m, "CUCG")
      .def("solve", &CUCG::solve)
      .def("is_success", &CUCG::is_success)
      .def("get_status", &CUCG::get_status)
      .def("get_iterations", &CUCG::get_iterations)
      .def("get_initial_residual_norm", &CUCG::get_initial_residual_norm)
      .def("get_residual_norm", &CUCG::get_residual_norm)
      .def("_get_last_result", [sparse_solve_result_to_dict](const CUCG &cg) {
        return sparse_solve_result_to_dict(cg.get_last_result());
      })
      .def("_debug_runtime_stats", [sparse_solve_plan_stats_to_dict](
                                         const CUCG &cg) {
        return sparse_solve_plan_stats_to_dict(
            cg.debug_runtime_statistics());
      });
  m.def(
      "make_cucg_solver",
      [](SparseMatrix &matrix, int max_iterations,
         float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return make_cucg_solver(matrix, max_iterations,
                                absolute_tolerance, verbose,
                                relative_tolerance);
      },
      py::arg("matrix"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseJacobiPreconditionerPlan &preconditioner,
         int max_iterations, float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return make_cuda_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, verbose, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_block_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseBlockJacobiPreconditionerPlan &preconditioner,
         int max_iterations, float absolute_tolerance, bool verbose,
         float relative_tolerance) {
        return make_cuda_block_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, verbose, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_compiled_kernel_cg_solver",
      make_cuda_compiled_kernel_cg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_compiled_graph_cg_solver",
      make_cuda_compiled_graph_cg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("verbose"), py::arg("relative_tolerance") = 0.0f);
  m.def(
      "_make_cuda_compiled_kernel_pcg_solver",
      make_cuda_compiled_kernel_pcg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"), py::arg("verbose"),
      py::arg("relative_tolerance") = 0.0f);

  py::class_<CpuSparseCGPlan>(m, "CpuSparseCGPlan")
      .def("solve", &CpuSparseCGPlan::solve)
      .def("is_success", &CpuSparseCGPlan::is_success)
      .def("get_status", &CpuSparseCGPlan::get_status)
      .def("get_iterations", &CpuSparseCGPlan::get_iterations)
      .def("get_initial_residual_norm",
           &CpuSparseCGPlan::get_initial_residual_norm)
      .def("get_residual_norm", &CpuSparseCGPlan::get_residual_norm)
      .def("_get_last_result",
           [sparse_solve_result_to_dict](const CpuSparseCGPlan &plan) {
             return sparse_solve_result_to_dict(plan.get_last_result());
           })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](const CpuSparseCGPlan &plan) {
             return sparse_solve_plan_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.attr("CpuBsrCGPlan") = m.attr("CpuSparseCGPlan");
  m.def(
      "_make_cpu_operator_cg_solver",
      [](Program *program, SparseMatrix &matrix, int max_iterations,
         double absolute_tolerance, double relative_tolerance) {
        return make_cpu_operator_cg_solver(
            program, matrix, max_iterations, absolute_tolerance,
            relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("matrix"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_experimental_linear_operator_cg_solver",
      make_cpu_experimental_linear_operator_cg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::arg("program"), py::arg("operator"),
      py::arg("max_iterations"), py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseJacobiPreconditionerPlan &preconditioner,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        return make_cpu_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_block_jacobi_pcg_solver",
      [](Program *program, SparseMatrix &matrix,
         SparseBlockJacobiPreconditionerPlan &preconditioner,
         int max_iterations, double absolute_tolerance,
         double relative_tolerance) {
        return make_cpu_block_jacobi_pcg_solver(
            program, matrix, preconditioner, max_iterations,
            absolute_tolerance, relative_tolerance);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);
  m.def(
      "_make_cpu_compiled_kernel_pcg_solver",
      make_cpu_compiled_kernel_pcg_solver,
      py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
      py::keep_alive<0, 3>(), py::arg("program"), py::arg("matrix"),
      py::arg("preconditioner"), py::arg("max_iterations"),
      py::arg("absolute_tolerance"),
      py::arg("relative_tolerance") = 0.0);

  py::class_<VulkanCGIterationPlan>(m, "VulkanCGIterationPlan")
      .def("solve", &VulkanCGIterationPlan::solve)
      .def("is_success", &VulkanCGIterationPlan::is_success)
      .def("get_iterations",
           &VulkanCGIterationPlan::get_iterations)
      .def("get_initial_residual_norm",
           &VulkanCGIterationPlan::get_initial_residual_norm)
      .def("get_residual_norm",
           &VulkanCGIterationPlan::get_residual_norm)
      .def("get_status", &VulkanCGIterationPlan::get_status)
      .def("_get_last_result", [sparse_solve_result_to_dict](
                                    const VulkanCGIterationPlan &plan) {
        return sparse_solve_result_to_dict(plan.get_last_result());
      })
      .def("_debug_runtime_stats",
           [sparse_solve_plan_stats_to_dict](
               const VulkanCGIterationPlan &plan) {
             return sparse_solve_plan_stats_to_dict(
                 plan.debug_runtime_statistics());
           });
  m.def("_make_vulkan_cg_iteration_plan",
        make_vulkan_cg_iteration_plan);
  m.def("_make_vulkan_cg_convergence_plan",
        make_vulkan_cg_convergence_plan);
  m.def("_make_vulkan_jacobi_pcg_convergence_plan",
        make_vulkan_jacobi_pcg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>());
  m.def("_make_vulkan_block_jacobi_pcg_convergence_plan",
        make_vulkan_block_jacobi_pcg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>());
  m.def("_make_vulkan_compiled_kernel_cg_convergence_plan",
        make_vulkan_compiled_kernel_cg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
  m.def("_make_vulkan_compiled_graph_cg_convergence_plan",
        make_vulkan_compiled_graph_cg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
  m.def("_make_vulkan_compiled_kernel_pcg_convergence_plan",
        make_vulkan_compiled_kernel_pcg_convergence_plan,
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
        py::keep_alive<0, 3>());

  // Mesh Class
  // Mesh related.
  py::enum_<mesh::MeshTopology>(m, "MeshTopology", py::arithmetic())
      .value("Triangle", mesh::MeshTopology::Triangle)
      .value("Tetrahedron", mesh::MeshTopology::Tetrahedron)
      .export_values();

  py::enum_<mesh::MeshElementType>(m, "MeshElementType", py::arithmetic())
      .value("Vertex", mesh::MeshElementType::Vertex)
      .value("Edge", mesh::MeshElementType::Edge)
      .value("Face", mesh::MeshElementType::Face)
      .value("Cell", mesh::MeshElementType::Cell)
      .export_values();

  py::enum_<mesh::MeshRelationType>(m, "MeshRelationType", py::arithmetic())
      .value("VV", mesh::MeshRelationType::VV)
      .value("VE", mesh::MeshRelationType::VE)
      .value("VF", mesh::MeshRelationType::VF)
      .value("VC", mesh::MeshRelationType::VC)
      .value("EV", mesh::MeshRelationType::EV)
      .value("EE", mesh::MeshRelationType::EE)
      .value("EF", mesh::MeshRelationType::EF)
      .value("EC", mesh::MeshRelationType::EC)
      .value("FV", mesh::MeshRelationType::FV)
      .value("FE", mesh::MeshRelationType::FE)
      .value("FF", mesh::MeshRelationType::FF)
      .value("FC", mesh::MeshRelationType::FC)
      .value("CV", mesh::MeshRelationType::CV)
      .value("CE", mesh::MeshRelationType::CE)
      .value("CF", mesh::MeshRelationType::CF)
      .value("CC", mesh::MeshRelationType::CC)
      .export_values();

  py::enum_<mesh::ConvType>(m, "ConvType", py::arithmetic())
      .value("l2g", mesh::ConvType::l2g)
      .value("l2r", mesh::ConvType::l2r)
      .value("g2r", mesh::ConvType::g2r)
      .export_values();

  py::class_<mesh::Mesh>(m, "Mesh");        // NOLINT(bugprone-unused-raii)
  py::class_<mesh::MeshPtr>(m, "MeshPtr");  // NOLINT(bugprone-unused-raii)

  m.def("element_order", mesh::element_order);
  m.def("from_end_element_order", mesh::from_end_element_order);
  m.def("to_end_element_order", mesh::to_end_element_order);
  m.def("relation_by_orders", mesh::relation_by_orders);
  m.def("inverse_relation", mesh::inverse_relation);
  m.def("element_type_name", mesh::element_type_name);

  m.def(
      "create_mesh",
      []() {
        auto mesh_shared = std::make_shared<mesh::Mesh>();
        mesh::MeshPtr mesh_ptr = mesh::MeshPtr{mesh_shared};
        return mesh_ptr;
      },
      py::return_value_policy::reference);

  // ad-hoc setters
  m.def("set_owned_offset",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
          mesh_ptr.ptr->owned_offset.insert(std::pair(type, snode));
        });
  m.def("set_total_offset",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
          mesh_ptr.ptr->total_offset.insert(std::pair(type, snode));
        });
  m.def("set_num_patches", [](mesh::MeshPtr &mesh_ptr, int num_patches) {
    mesh_ptr.ptr->num_patches = num_patches;
  });

  m.def("set_num_elements", [](mesh::MeshPtr &mesh_ptr,
                               mesh::MeshElementType type, int num_elements) {
    mesh_ptr.ptr->num_elements.insert(std::pair(type, num_elements));
  });

  m.def("get_num_elements",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type) {
          return mesh_ptr.ptr->num_elements.find(type)->second;
        });

  m.def("set_patch_max_element_num",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type,
           int max_element_num) {
          mesh_ptr.ptr->patch_max_element_num.insert(
              std::pair(type, max_element_num));
        });

  m.def("set_index_mapping",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType element_type,
           mesh::ConvType conv_type, SNode *snode) {
          mesh_ptr.ptr->index_mapping.insert(
              std::make_pair(std::make_pair(element_type, conv_type), snode));
        });

  m.def("set_relation_fixed",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value) {
          mesh_ptr.ptr->relations.insert(
              std::pair(type, mesh::MeshLocalRelation(value)));
        });

  m.def("set_relation_dynamic",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value,
           SNode *patch_offset, SNode *offset) {
          mesh_ptr.ptr->relations.insert(std::pair(
              type, mesh::MeshLocalRelation(value, patch_offset, offset)));
        });

  m.def("wait_for_debugger", []() {
#ifdef WIN32
    while (!::IsDebuggerPresent())
      ::Sleep(100);
#endif
  });

  auto operationClass = py::class_<Operation>(m, "Operation");
  auto internalOpClass = py::class_<InternalOp>(m, "InternalOp");

#define PER_INTERNAL_OP(x)                                           \
  internalOpClass.def_property_readonly_static(                      \
      #x, [](py::object) { return Operations::get(InternalOp::x); }, \
      py::return_value_policy::reference);
#include "taichi/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
}

}  // namespace taichi
