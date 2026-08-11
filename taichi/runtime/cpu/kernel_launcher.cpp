#include "taichi/runtime/cpu/kernel_launcher.h"

#include <chrono>
#include <ctime>
#include <utility>

#include "taichi/rhi/arch.h"
#include "taichi/system/profiler.h"
#include "taichi/system/profiler_annotation.h"
#include "taichi/util/environ_config.h"

namespace taichi::lang {
namespace cpu {

namespace {

using LaunchClock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(LaunchClock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(LaunchClock::now() -
                                                           start)
          .count());
}

}  // namespace

KernelLauncher::KernelLauncher(Config config) : Base(std::move(config)) {
  launch_attribution_.enabled =
      get_environ_config("TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION", 0) != 0;
}

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  const bool attribute = launch_attribution_.enabled;
  const auto launch_start =
      attribute ? LaunchClock::now() : LaunchClock::time_point{};
  const auto launch_cpu_start = attribute ? std::clock() : std::clock_t{};
  if (attribute) {
    launch_attribution_.launches.fetch_add(1, std::memory_order_relaxed);
    if (Profiling::is_tracing_enabled()) {
      launch_attribution_.compile_profiler_enabled_launches.fetch_add(
          1, std::memory_order_relaxed);
    }
  }
  std::unique_lock<std::mutex> execution_lock(execution_mutex_,
                                               std::defer_lock);
  if (attribute) {
    const auto start = LaunchClock::now();
    execution_lock.lock();
    launch_attribution_.execution_lock_wait_ns.fetch_add(
        elapsed_ns(start), std::memory_order_relaxed);
  } else {
    execution_lock.lock();
  }
  const auto execution_lock_acquired =
      attribute ? LaunchClock::now() : LaunchClock::time_point{};
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex(),
                                                   std::defer_lock);
  if (attribute) {
    const auto start = LaunchClock::now();
    launch_lock.lock();
    launch_attribution_.registration_lock_wait_ns.fetch_add(
        elapsed_ns(start), std::memory_order_relaxed);
  } else {
    launch_lock.lock();
  }
  const auto registration_lock_acquired =
      attribute ? LaunchClock::now() : LaunchClock::time_point{};
  std::shared_ptr<const Context> launcher_ctx;
  if (attribute) {
    const auto start = LaunchClock::now();
    auto iter = contexts_.find(handle.get_launch_id());
    TI_ASSERT(iter != contexts_.end());
    launcher_ctx = iter->second;
    launch_attribution_.context_lookup_ns.fetch_add(
        elapsed_ns(start), std::memory_order_relaxed);
  } else {
    auto iter = contexts_.find(handle.get_launch_id());
    TI_ASSERT(iter != contexts_.end());
    launcher_ctx = iter->second;
  }
  auto *executor = get_runtime_executor();

  ctx.get_context().runtime = executor->get_llvm_runtime();
  // For taichi ndarrays, context.array_ptrs saves pointer to its
  // |DeviceAllocation|, CPU backend actually want to use the raw ptr here.
  const auto &parameters = launcher_ctx->parameters;
  {
    const auto attribution_start =
        attribute ? LaunchClock::now() : LaunchClock::time_point{};
    TI_COMPILE_PROFILER("cpu_launch_bind_args");
    for (int i = 0; i < (int)parameters.size(); i++) {
      const auto &kv = parameters[i];
      const auto &key = kv.first;
      const auto &parameter = kv.second;
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      std::vector<int> grad_ptr_idx = key;
      grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);

      if (parameter.is_array && ctx.device_allocation_type[key] ==
                                    LaunchContextBuilder::DevAllocType::kNone) {
        ctx.set_ndarray_ptrs(key, (uint64)ctx.array_ptrs[data_ptr_idx],
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
      }
      if (parameter.is_array &&
          ctx.device_allocation_type[key] !=
              LaunchContextBuilder::DevAllocType::kNone &&
          ctx.array_runtime_sizes[key] > 0) {
        const auto alloc_type = ctx.device_allocation_type[key];
        uint64 host_ptr = 0;
        if (alloc_type ==
            LaunchContextBuilder::DevAllocType::kDenseStorage) {
          const auto &binding = ctx.get_resolved_dense_storage(key);
          auto *base = reinterpret_cast<char *>(
              executor->get_device_alloc_info_ptr(binding.allocation));
          host_ptr = reinterpret_cast<uint64>(base + binding.byte_offset);
        } else {
          DeviceAllocation *ptr =
              static_cast<DeviceAllocation *>(ctx.array_ptrs[data_ptr_idx]);
          host_ptr =
              (uint64)executor->get_device_alloc_info_ptr(*ptr);
        }
        ctx.set_array_device_allocation_type(
            key, LaunchContextBuilder::DevAllocType::kNone);

        auto grad_ptr = alloc_type ==
                                LaunchContextBuilder::DevAllocType::kDenseStorage
                            ? nullptr
                            : ctx.array_ptrs[grad_ptr_idx];
        uint64 host_ptr_grad =
            grad_ptr == nullptr
                ? 0
                : (uint64)executor->get_device_alloc_info_ptr(
                      *static_cast<DeviceAllocation *>(grad_ptr));
        ctx.set_ndarray_ptrs(key, host_ptr, host_ptr_grad);
      }
      if (parameter.is_argpack) {
        data_ptr_idx = key;
        data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
        auto *argpack = ctx.argpack_ptrs[key];
        auto argpack_ptr = argpack->get_device_allocation();
        uint64 host_ptr =
            (uint64)executor->get_device_alloc_info_ptr(argpack_ptr);
        if (key.size() == 1) {
          ctx.set_argpack_ptr(key, host_ptr);
        } else {
          auto key_parent = key;
          key_parent.pop_back();
          auto *argpack_parent = ctx.argpack_ptrs[key_parent];
          argpack_parent->set_arg_nested_argpack_ptr(key.back(), host_ptr);
        }
      }
    }
    if (attribute) {
      launch_attribution_.argument_binding_ns.fetch_add(
          elapsed_ns(attribution_start), std::memory_order_relaxed);
    }
  }
  {
    const auto attribution_start =
        attribute ? LaunchClock::now() : LaunchClock::time_point{};
    TI_COMPILE_PROFILER("cpu_launch_tasks");
    const bool labeled = !ctx.dispatch_label().empty();
    if (attribute) {
      launch_attribution_.task_invocations.fetch_add(
          launcher_ctx->task_funcs.size(), std::memory_order_relaxed);
      if (labeled) {
        launch_attribution_.labeled_launches.fetch_add(
            1, std::memory_order_relaxed);
      }
      if (sparse_listgen_telemetry_enabled_) {
        launch_attribution_.sparse_telemetry_launches.fetch_add(
            1, std::memory_order_relaxed);
      }
    }
    const auto run_task = [&](std::size_t index) {
      const auto start =
          attribute ? LaunchClock::now() : LaunchClock::time_point{};
      launcher_ctx->task_funcs[index](&ctx.get_context());
      if (!attribute) {
        return;
      }
      const auto duration = elapsed_ns(start);
      switch (launcher_ctx->task_types[index]) {
        case OffloadedTaskType::serial:
          launch_attribution_.serial_task_invocations.fetch_add(
              1, std::memory_order_relaxed);
          launch_attribution_.serial_task_execution_ns.fetch_add(
              duration, std::memory_order_relaxed);
          break;
        case OffloadedTaskType::range_for:
          launch_attribution_.range_task_invocations.fetch_add(
              1, std::memory_order_relaxed);
          launch_attribution_.range_task_execution_ns.fetch_add(
              duration, std::memory_order_relaxed);
          break;
        default:
          launch_attribution_.other_task_invocations.fetch_add(
              1, std::memory_order_relaxed);
          launch_attribution_.other_task_execution_ns.fetch_add(
              duration, std::memory_order_relaxed);
          break;
      }
    };
    if (!sparse_listgen_telemetry_enabled_) {
      if (!labeled) {
        for (std::size_t i = 0; i < launcher_ctx->task_funcs.size(); ++i) {
          run_task(i);
        }
      } else {
        TI_ASSERT(launcher_ctx->task_funcs.size() ==
                  launcher_ctx->task_trace_metadata.size());
        for (std::size_t i = 0; i < launcher_ctx->task_funcs.size(); ++i) {
          const auto &[task_name, task_id] =
              launcher_ctx->task_trace_metadata[i];
          const auto trace_name = make_labeled_task_name(
              task_name, task_id, ctx.dispatch_label());
          ScopedKernelProfilerName profiler_name(trace_name);
          ScopedExternalProfilerAnnotation annotation(trace_name);
          run_task(i);
        }
      }
    } else {
      TI_ASSERT(launcher_ctx->task_funcs.size() ==
                launcher_ctx->sparse_task_metadata.size());
      for (std::size_t i = 0; i < launcher_ctx->task_funcs.size(); ++i) {
        const auto &metadata = launcher_ctx->sparse_task_metadata[i];
        const bool is_listgen =
            metadata.sparse_list_op == OffloadedTask::kSparseListOpListgen &&
            metadata.snode_id >= 0;
        if (is_listgen) {
          auto &telemetry = sparse_listgen_telemetry_[metadata.snode_id];
          telemetry.snode_id = metadata.snode_id;
          telemetry.parent_snode_id = metadata.parent_snode_id;
          ++telemetry.requests;
          executor->begin_cpu_sparse_listgen_work();
        }
        if (!labeled) {
          run_task(i);
        } else {
          TI_ASSERT(launcher_ctx->task_funcs.size() ==
                    launcher_ctx->task_trace_metadata.size());
          const auto &[task_name, task_id] =
              launcher_ctx->task_trace_metadata[i];
          const auto trace_name = make_labeled_task_name(
              task_name, task_id, ctx.dispatch_label());
          ScopedKernelProfilerName profiler_name(trace_name);
          ScopedExternalProfilerAnnotation annotation(trace_name);
          run_task(i);
        }
        if (is_listgen) {
          const auto work = executor->read_cpu_sparse_listgen_work();
          if (work.available) {
            auto &telemetry = sparse_listgen_telemetry_[metadata.snode_id];
            if (work.reused) {
              ++telemetry.reuse_hits;
              continue;
            }
            const bool cold = telemetry.rebuilds == 0;
            ++telemetry.rebuilds;
            telemetry.last_rebuild_reason =
                cold ? "cold" : "runtime_topology_or_parent_changed";
            if (!telemetry.scanned_elements.available) {
              telemetry.scanned_elements = {0, true};
              telemetry.emitted_elements = {0, true};
              telemetry.serial_rebuilds = {0, true};
              telemetry.parallel_rebuilds = {0, true};
            }
            telemetry.scanned_elements.value += work.scanned_elements;
            telemetry.emitted_elements.value += work.emitted_elements;
            if (work.parallel) {
              ++telemetry.parallel_rebuilds.value;
            } else {
              ++telemetry.serial_rebuilds.value;
            }
          }
        }
      }
    }
    if (attribute) {
      launch_attribution_.task_execution_ns.fetch_add(
          elapsed_ns(attribution_start), std::memory_order_relaxed);
    }
  }
  if (attribute) {
    const auto finish = LaunchClock::now();
    launch_attribution_.registration_lock_hold_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                finish - registration_lock_acquired)
                .count()),
        std::memory_order_relaxed);
    launch_attribution_.execution_lock_hold_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                finish - execution_lock_acquired)
                .count()),
        std::memory_order_relaxed);
    launch_attribution_.launch_wall_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                finish - launch_start)
                .count()),
        std::memory_order_relaxed);
    const auto cpu_ticks = std::clock() - launch_cpu_start;
    if (cpu_ticks > 0) {
      launch_attribution_.launch_cpu_ns.fetch_add(
          static_cast<std::uint64_t>(cpu_ticks) * 1000000000ull /
              CLOCKS_PER_SEC,
          std::memory_order_relaxed);
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(arch_is_cpu(compiled.arch()));
  std::unique_lock<std::shared_mutex> lock(registration_mutex());

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();

    auto ctx = std::make_shared<Context>();
    auto *executor = get_runtime_executor();

    const auto &internal_data = compiled.get_internal_data();
    auto data = internal_data.compiled_data.clone();
    auto parameters = internal_data.args;
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Construct task_funcs
    using TaskFunc = int32 (*)(void *);
    std::vector<TaskFunc> task_funcs;
    std::vector<OffloadedTaskType> task_types;
    std::vector<std::pair<std::string, std::string>> task_trace_metadata;
    std::vector<Context::SparseTaskMetadata> sparse_task_metadata;
    task_funcs.reserve(data.tasks.size());
    task_types.reserve(data.tasks.size());
    task_trace_metadata.reserve(data.tasks.size());
    sparse_task_metadata.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      TI_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found",
                     task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
      task_types.push_back(task.task_type);
      task_trace_metadata.emplace_back(task.name, task.task_id);
      sparse_task_metadata.push_back({task.sparse_list_op,
                                      task.sparse_list_snode_id,
                                      task.sparse_list_parent_snode_id});
    }

    // Populate ctx
    ctx->jit_module = jit_module;
    ctx->snode_tree_ids = compiled.snode_tree_ids();
    ctx->parameters = std::move(parameters);
    ctx->task_funcs = std::move(task_funcs);
    ctx->task_types = std::move(task_types);
    ctx->task_trace_metadata = std::move(task_trace_metadata);
    ctx->sparse_task_metadata = std::move(sparse_task_metadata);
    const bool was_inserted = contexts_.emplace(index, std::move(ctx)).second;
    TI_ASSERT(was_inserted);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

void KernelLauncher::retire_snode_tree(int tree_id) {
  std::unique_lock<std::mutex> execution_lock(execution_mutex_);
  std::unique_lock<std::shared_mutex> registration_lock(registration_mutex());
  auto *executor = get_runtime_executor();
  std::vector<int> retired_sparse_snode_ids;
  for (auto iter = contexts_.begin(); iter != contexts_.end();) {
    const auto &context = iter->second;
    if (!std::binary_search(context->snode_tree_ids.begin(),
                            context->snode_tree_ids.end(), tree_id)) {
      ++iter;
      continue;
    }
    for (const auto &metadata : context->sparse_task_metadata) {
      if (metadata.snode_id >= 0) {
        retired_sparse_snode_ids.push_back(metadata.snode_id);
      }
    }
    auto module = context->jit_module;
    iter = contexts_.erase(iter);
    executor->remove_jit_module(module);
  }
  for (int snode_id : retired_sparse_snode_ids) {
    sparse_listgen_telemetry_.erase(snode_id);
  }
}

std::size_t KernelLauncher::debug_registered_kernel_count() {
  std::shared_lock<std::shared_mutex> lock(registration_mutex());
  return contexts_.size();
}

void KernelLauncher::debug_reset_sparse_listgen_statistics() {
  std::unique_lock<std::mutex> lock(execution_mutex_);
  sparse_listgen_telemetry_.clear();
  sparse_listgen_telemetry_enabled_ = true;
}

SparseSNodeTreeListgenStatistics
KernelLauncher::debug_sparse_listgen_statistics(
    const std::vector<int> &snode_ids) {
  std::unique_lock<std::mutex> lock(execution_mutex_);
  SparseSNodeTreeListgenStatistics result;
  result.available = sparse_listgen_telemetry_enabled_;
  if (!result.available) {
    return result;
  }
  for (const auto &[snode_id, telemetry] : sparse_listgen_telemetry_) {
    if (std::binary_search(snode_ids.begin(), snode_ids.end(), snode_id)) {
      result.nodes.push_back(telemetry);
    }
  }
  std::sort(result.nodes.begin(), result.nodes.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.snode_id < rhs.snode_id;
            });
  return result;
}

void KernelLauncher::debug_reset_launch_attribution() {
  auto &stats = launch_attribution_;
#define TI_RESET_CPU_LAUNCH_COUNTER(name) \
  stats.name.store(0, std::memory_order_relaxed)
  TI_RESET_CPU_LAUNCH_COUNTER(launches);
  TI_RESET_CPU_LAUNCH_COUNTER(launch_wall_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(launch_cpu_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(execution_lock_wait_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(execution_lock_hold_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(registration_lock_wait_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(registration_lock_hold_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(context_lookup_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(argument_binding_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(task_execution_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(task_invocations);
  TI_RESET_CPU_LAUNCH_COUNTER(serial_task_invocations);
  TI_RESET_CPU_LAUNCH_COUNTER(serial_task_execution_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(range_task_invocations);
  TI_RESET_CPU_LAUNCH_COUNTER(range_task_execution_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(other_task_invocations);
  TI_RESET_CPU_LAUNCH_COUNTER(other_task_execution_ns);
  TI_RESET_CPU_LAUNCH_COUNTER(labeled_launches);
  TI_RESET_CPU_LAUNCH_COUNTER(sparse_telemetry_launches);
  TI_RESET_CPU_LAUNCH_COUNTER(compile_profiler_enabled_launches);
#undef TI_RESET_CPU_LAUNCH_COUNTER
}

std::unordered_map<std::string, std::uint64_t>
KernelLauncher::debug_launch_attribution() const {
  const auto &stats = launch_attribution_;
  auto load = [](const std::atomic<std::uint64_t> &value) {
    return value.load(std::memory_order_relaxed);
  };
  return {
      {"enabled", stats.enabled ? 1u : 0u},
      {"launches", load(stats.launches)},
      {"launch_wall_ns", load(stats.launch_wall_ns)},
      {"launch_cpu_ns", load(stats.launch_cpu_ns)},
      {"execution_lock_wait_ns", load(stats.execution_lock_wait_ns)},
      {"execution_lock_hold_ns", load(stats.execution_lock_hold_ns)},
      {"registration_lock_wait_ns", load(stats.registration_lock_wait_ns)},
      {"registration_lock_hold_ns", load(stats.registration_lock_hold_ns)},
      {"context_lookup_ns", load(stats.context_lookup_ns)},
      {"argument_binding_ns", load(stats.argument_binding_ns)},
      {"task_execution_ns", load(stats.task_execution_ns)},
      {"task_invocations", load(stats.task_invocations)},
      {"serial_task_invocations", load(stats.serial_task_invocations)},
      {"serial_task_execution_ns", load(stats.serial_task_execution_ns)},
      {"range_task_invocations", load(stats.range_task_invocations)},
      {"range_task_execution_ns", load(stats.range_task_execution_ns)},
      {"other_task_invocations", load(stats.other_task_invocations)},
      {"other_task_execution_ns", load(stats.other_task_execution_ns)},
      {"labeled_launches", load(stats.labeled_launches)},
      {"sparse_telemetry_launches", load(stats.sparse_telemetry_launches)},
      {"compile_profiler_enabled_launches",
       load(stats.compile_profiler_enabled_launches)},
  };
}

}  // namespace cpu
}  // namespace taichi::lang
