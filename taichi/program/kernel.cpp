#include "taichi/program/kernel.h"

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/codegen/codegen.h"
#include "taichi/common/logging.h"
#include "taichi/common/serialization.h"
#include "taichi/common/task.h"
#include "taichi/inc/constants.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/bit.h"

#include "picosha2.h"

#include <utility>

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

class Function;

Kernel::Kernel(Program &program,
               const std::function<void()> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  this->init(program, func, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               const std::function<void(Kernel *)> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  // due to #6362, we cannot write [func, this] { return func(this); }
  this->init(program, [&] { return func(this); }, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               std::unique_ptr<IRNode> &&ir,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  this->arch = program.compile_config().arch;
  this->autodiff_mode = autodiff_mode;
  this->ir = std::move(ir);
  this->program = &program;
  is_accessor = false;
  ir_is_ast_ = false;  // CHI IR

  TI_ASSERT(this->ir->is<Block>());
  this->ir->as<Block>()->set_parent_callable(this);

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  } else if (autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    name = primal_name + "_validate_grad";
  } else {
    TI_ERROR("Unsupported autodiff mode");
  }
}

LaunchContextBuilder Kernel::make_launch_context() {
  LaunchContextBuilder builder(this);
  const OffloadExecutionPlan *plan = nullptr;
  if (offload_execution_plan_frozen_.load(std::memory_order_acquire)) {
    TI_ASSERT(offload_execution_plan_.has_value());
    plan = &*offload_execution_plan_;
  } else {
    std::lock_guard<std::mutex> lock(offload_execution_plan_mutex_);
    if (offload_execution_plan_.has_value()) {
      plan = &*offload_execution_plan_;
      // Publish the immutable plan before later launches take the lock-free
      // path. Once published, the setter fails closed instead of invalidating
      // any LaunchContextBuilder's borrowed references.
      offload_execution_plan_frozen_.store(true, std::memory_order_release);
    }
  }
  if (plan != nullptr) {
    builder.bind_cuda_task_execution_plan(
        plan->execution_identity, plan->launch_content_digest, plan->task_kinds,
        plan->grid_residency_waves, plan->range_work_per_thread_targets);
  }
  return builder;
}

template <typename T>
T Kernel::fetch_ret(DataType dt, int i) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (T)program->fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (T)program->fetch_result<float64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (T)program->fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (T)program->fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (T)program->fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (T)program->fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return (T)program->fetch_result<uint1>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (T)program->fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (T)program->fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (T)program->fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (T)program->fetch_result<uint64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    // use f32 to interact with python
    return (T)program->fetch_result<float32>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

std::string Kernel::get_name() const {
  return name;
}

void Kernel::set_kernel_key_for_cache(const std::string &kernel_key) const {
  kernel_key_ = kernel_key;
  kernel_key_valid_ = true;
}

const std::string &Kernel::get_cached_kernel_key() const {
  if (kernel_key_valid_) {
    return kernel_key_;
  }
  static const std::string empty;
  return empty;
}

void Kernel::invalidate_kernel_key_for_cache() const {
  kernel_key_valid_ = false;
}

bool Kernel::has_cached_offline_cache_body() const {
  return offline_cache_body_.has_value();
}

const std::string &Kernel::get_cached_offline_cache_body() const {
  return *offline_cache_body_;
}

void Kernel::set_offline_cache_body(std::string body) const {
  offline_cache_body_ = std::move(body);
}

void Kernel::set_compile_tier_override(const std::string &tier) {
  compile_tier_override_ = tier;
  invalidate_kernel_key_for_cache();
}

void Kernel::clear_compile_tier_override() {
  compile_tier_override_.reset();
  invalidate_kernel_key_for_cache();
}

const std::optional<std::string> &Kernel::get_compile_tier_override() const {
  return compile_tier_override_;
}

void Kernel::set_task_launch_policy(
    const std::string &mode,
    int block_dim,
    bool injected_block_dim,
    const std::string &optimization_spec_identity,
    const std::string &thread_local_mode,
    int cuda_min_blocks_per_sm,
    int cuda_max_registers) {
  TI_ERROR_IF(block_dim <= 0 || block_dim > taichi_max_gpu_block_dim,
              "TaskLaunchPolicy block_dim must be in [1, {}], got {}",
              taichi_max_gpu_block_dim, block_dim);
  TI_ERROR_IF((block_dim % 32 != 0) && !bit::is_power_of_two(block_dim),
              "TaskLaunchPolicy block_dim must be a power of two or a "
              "multiple of 32, got {}",
              block_dim);
  TaskLaunchPolicy policy;
  if (mode == "hint") {
    policy.mode = TaskLaunchPolicyMode::hint;
  } else if (mode == "require") {
    policy.mode = TaskLaunchPolicyMode::require;
  } else {
    TI_ERROR("TaskLaunchPolicy mode must be 'hint' or 'require', got {}", mode);
  }
  if (thread_local_mode == "auto") {
    policy.thread_local_mode = TaskLaunchThreadLocalMode::automatic;
  } else if (thread_local_mode == "on") {
    policy.thread_local_mode = TaskLaunchThreadLocalMode::enabled;
  } else if (thread_local_mode == "off") {
    policy.thread_local_mode = TaskLaunchThreadLocalMode::disabled;
  } else {
    TI_ERROR("TaskLaunchPolicy thread-local mode must be 'auto', 'on', or "
             "'off', got {}",
             thread_local_mode);
  }
  policy.block_dim = block_dim;
  TI_ERROR_IF(cuda_min_blocks_per_sm != 1 && cuda_min_blocks_per_sm != 2 &&
                  cuda_min_blocks_per_sm != 4,
              "CUDA min blocks per SM must be 1, 2, or 4, got {}",
              cuda_min_blocks_per_sm);
  TI_ERROR_IF(cuda_max_registers < -1 || cuda_max_registers > 255 ||
                  (cuda_max_registers > 0 && cuda_max_registers < 16),
              "CUDA max registers must be -1, 0, or in [16, 255], got {}",
              cuda_max_registers);
  policy.cuda_min_blocks_per_sm = cuda_min_blocks_per_sm;
  policy.cuda_max_registers = cuda_max_registers;
  policy.injected_block_dim = injected_block_dim;
  TI_ERROR_IF(optimization_spec_identity.empty(),
              "TaskLaunchPolicy requires a non-empty optimization spec "
              "identity");
  policy.optimization_spec_identity = optimization_spec_identity;
  task_launch_policy_ = policy;
  set_kernel_optimization_spec(
      optimization_spec_identity, thread_local_mode,
      cuda_min_blocks_per_sm, cuda_max_registers);
  invalidate_kernel_key_for_cache();
}

const std::optional<Kernel::TaskLaunchPolicy> &Kernel::get_task_launch_policy()
    const {
  return task_launch_policy_;
}

std::string Kernel::task_launch_policy_cache_key() const {
  if (!task_launch_policy_.has_value()) {
    return {};
  }
  const char mode =
      task_launch_policy_->mode == TaskLaunchPolicyMode::require ? 'r' : 'h';
  const char thread_local_mode =
      task_launch_policy_->thread_local_mode ==
              TaskLaunchThreadLocalMode::enabled
          ? 'e'
          : (task_launch_policy_->thread_local_mode ==
                     TaskLaunchThreadLocalMode::disabled
                 ? 'd'
                 : 'a');
  return fmt::format(
      "{}:{}:{}:{}:{}:{}:{}", mode, task_launch_policy_->block_dim,
      task_launch_policy_->injected_block_dim ? 'i' : 's', thread_local_mode,
      task_launch_policy_->cuda_min_blocks_per_sm,
      task_launch_policy_->cuda_max_registers,
      task_launch_policy_->optimization_spec_identity);
}

void Kernel::set_kernel_optimization_spec(
    const std::string &identity,
    const std::string &thread_local_mode,
    int cuda_min_blocks_per_sm,
    int cuda_max_registers) {
  TI_ERROR_IF(offload_execution_plan_.has_value(),
              "legacy kernel optimization metadata cannot be combined with "
              "an offload execution plan");
  TI_ERROR_IF(identity.empty(),
              "kernel optimization spec requires a non-empty identity");
  KernelOptimizationSpec spec;
  if (thread_local_mode == "auto") {
    spec.thread_local_mode = TaskLaunchThreadLocalMode::automatic;
  } else if (thread_local_mode == "on") {
    spec.thread_local_mode = TaskLaunchThreadLocalMode::enabled;
  } else if (thread_local_mode == "off") {
    spec.thread_local_mode = TaskLaunchThreadLocalMode::disabled;
  } else {
    TI_ERROR("kernel optimization thread-local mode must be 'auto', 'on', or "
             "'off', got {}",
             thread_local_mode);
  }
  TI_ERROR_IF(cuda_min_blocks_per_sm != 1 && cuda_min_blocks_per_sm != 2 &&
                  cuda_min_blocks_per_sm != 4,
              "CUDA min blocks per SM must be 1, 2, or 4, got {}",
              cuda_min_blocks_per_sm);
  TI_ERROR_IF(cuda_max_registers < -1 || cuda_max_registers > 255 ||
                  (cuda_max_registers > 0 && cuda_max_registers < 16),
              "CUDA max registers must be -1, 0, or in [16, 255], got {}",
              cuda_max_registers);
  spec.cuda_min_blocks_per_sm = cuda_min_blocks_per_sm;
  spec.cuda_max_registers = cuda_max_registers;
  spec.identity = identity;
  kernel_optimization_spec_ = std::move(spec);
  invalidate_kernel_key_for_cache();
}

const std::optional<Kernel::KernelOptimizationSpec> &
Kernel::get_kernel_optimization_spec() const {
  return kernel_optimization_spec_;
}

std::string Kernel::optimization_spec_cache_key() const {
  if (offload_execution_plan_.has_value()) {
    return fmt::format("oep:{}",
                       offload_execution_plan_->compilation_identity);
  }
  if (!kernel_optimization_spec_.has_value()) {
    return {};
  }
  if (task_launch_policy_.has_value()) {
    return task_launch_policy_cache_key();
  }
  const char thread_local_mode =
      kernel_optimization_spec_->thread_local_mode ==
              TaskLaunchThreadLocalMode::enabled
          ? 'e'
          : (kernel_optimization_spec_->thread_local_mode ==
                     TaskLaunchThreadLocalMode::disabled
                 ? 'd'
                 : 'a');
  return fmt::format(
      "k:{}:{}:{}:{}", thread_local_mode,
      kernel_optimization_spec_->cuda_min_blocks_per_sm,
      kernel_optimization_spec_->cuda_max_registers,
      kernel_optimization_spec_->identity);
}

const std::string &Kernel::optimization_spec_identity() const {
  static const std::string kEmptyIdentity;
  if (offload_execution_plan_.has_value()) {
    return offload_execution_plan_->compilation_identity;
  }
  return kernel_optimization_spec_.has_value()
             ? kernel_optimization_spec_->identity
             : kEmptyIdentity;
}

void Kernel::set_offload_execution_plan(
    const std::string &compilation_identity,
    const std::string &execution_identity,
    const std::vector<int> &task_indices,
    const std::vector<std::string> &task_kinds,
    const std::vector<int> &workgroup_sizes,
    const std::vector<std::string> &thread_local_modes,
    const std::vector<int> &cuda_min_blocks_per_sm,
    const std::vector<int> &cuda_max_registers,
    const std::vector<int> &grid_residency_waves,
    const std::vector<int> &range_work_per_thread_targets,
    const std::vector<std::string> &memory_strategies) {
  TI_ERROR_IF(
      offload_execution_plan_frozen_.load(std::memory_order_acquire),
      "offload execution plan cannot be replaced after launch-context "
      "materialization");
  TI_ERROR_IF(compilation_identity.empty() || execution_identity.empty(),
              "offload execution plan requires non-empty compilation and "
              "execution identities");
  TI_ERROR_IF(task_launch_policy_.has_value() ||
                  kernel_optimization_spec_.has_value(),
              "offload execution plans cannot be combined with legacy "
              "kernel optimization metadata");
  const std::size_t task_count = task_indices.size();
  TI_ERROR_IF(
      task_count == 0 || task_kinds.size() != task_count ||
          workgroup_sizes.size() != task_count ||
          thread_local_modes.size() != task_count ||
          cuda_min_blocks_per_sm.size() != task_count ||
          cuda_max_registers.size() != task_count ||
          grid_residency_waves.size() != task_count ||
          range_work_per_thread_targets.size() != task_count ||
          memory_strategies.size() != task_count,
      "offload execution plan vectors must have one entry per physical task");

  OffloadExecutionPlan plan;
  plan.compilation_identity = compilation_identity;
  plan.execution_identity = execution_identity;
  plan.tasks.reserve(task_count);
  plan.task_kinds.reserve(task_count);
  plan.grid_residency_waves.reserve(task_count);
  plan.range_work_per_thread_targets.reserve(task_count);
  for (std::size_t index = 0; index < task_count; ++index) {
    TI_ERROR_IF(task_indices[index] != static_cast<int>(index),
                "offload execution plan task indices must be contiguous; "
                "expected {}, got {}",
                index, task_indices[index]);
    TI_ERROR_IF(task_kinds[index].empty(),
                "offload execution plan task {} has no task kind", index);
    const int block_dim = workgroup_sizes[index];
    TI_ERROR_IF(block_dim < 0 || block_dim > taichi_max_gpu_block_dim ||
                    (block_dim != 0 && (block_dim % 32 != 0) &&
                     !bit::is_power_of_two(block_dim)),
                "offload task workgroup size must be zero, a power of two, "
                "or a multiple of 32 in the device limit; got {}",
                block_dim);

    OffloadTaskOptimizationSpec task;
    task.task_index = static_cast<std::uint32_t>(index);
    task.task_kind = task_kinds[index];
    task.workgroup_size = block_dim;
    if (thread_local_modes[index] == "auto") {
      task.thread_local_mode = TaskLaunchThreadLocalMode::automatic;
    } else if (thread_local_modes[index] == "on") {
      task.thread_local_mode = TaskLaunchThreadLocalMode::enabled;
    } else if (thread_local_modes[index] == "off") {
      task.thread_local_mode = TaskLaunchThreadLocalMode::disabled;
    } else {
      TI_ERROR("offload task thread-local mode must be 'auto', 'on', or "
               "'off', got {}",
               thread_local_modes[index]);
    }
    TI_ERROR_IF(cuda_min_blocks_per_sm[index] != 1 &&
                    cuda_min_blocks_per_sm[index] != 2 &&
                    cuda_min_blocks_per_sm[index] != 4,
                "CUDA min blocks per SM must be 1, 2, or 4, got {}",
                cuda_min_blocks_per_sm[index]);
    TI_ERROR_IF(cuda_max_registers[index] < -1 ||
                    cuda_max_registers[index] > 255 ||
                    (cuda_max_registers[index] > 0 &&
                     cuda_max_registers[index] < 16),
                "CUDA max registers must be -1, 0, or in [16, 255], got {}",
                cuda_max_registers[index]);
    TI_ERROR_IF(grid_residency_waves[index] != 0 &&
                    grid_residency_waves[index] != 1 &&
                    grid_residency_waves[index] != 2 &&
                    grid_residency_waves[index] != 4,
                "CUDA grid residency waves must be 0, 1, 2, or 4");
    TI_ERROR_IF(range_work_per_thread_targets[index] != 1 &&
                    range_work_per_thread_targets[index] != 2 &&
                    range_work_per_thread_targets[index] != 4 &&
                    range_work_per_thread_targets[index] != 8,
                "CUDA range work-per-thread target must be 1, 2, 4, or 8");
    task.cuda_min_blocks_per_sm = cuda_min_blocks_per_sm[index];
    task.cuda_max_registers = cuda_max_registers[index];
    task.grid_residency_waves = grid_residency_waves[index];
    task.range_work_per_thread_target =
        range_work_per_thread_targets[index];
    TI_ERROR_IF(memory_strategies[index] != "direct" &&
                    memory_strategies[index] != "shared_staged_1d",
                "offload task memory strategy must be 'direct' or "
                "'shared_staged_1d', got {}",
                memory_strategies[index]);
    task.memory_strategy = memory_strategies[index];
    TI_ERROR_IF(
        task.memory_strategy == "shared_staged_1d" &&
            (task.task_kind != "range_for" || task.workgroup_size == 0 ||
             task.grid_residency_waves != 0 ||
             task.range_work_per_thread_target != 1),
        "shared_staged_1d requires a range_for task, an exact "
        "workgroup size, automatic grid residency, and one item per "
        "thread");
    const bool nonbaseline =
        task.workgroup_size != 0 ||
        task.thread_local_mode != TaskLaunchThreadLocalMode::automatic ||
        task.cuda_min_blocks_per_sm != 2 || task.cuda_max_registers != -1 ||
        task.grid_residency_waves != 0 ||
        task.range_work_per_thread_target != 1 ||
        task.memory_strategy != "direct";
    TI_ERROR_IF(task.task_kind != "range_for" && nonbaseline,
                "offload execution plan v1 can tune only range_for tasks; "
                "task {} is {}",
                index, task.task_kind);
    plan.task_kinds.push_back(task.task_kind);
    plan.grid_residency_waves.push_back(task.grid_residency_waves);
    plan.range_work_per_thread_targets.push_back(
        task.range_work_per_thread_target);
    plan.tasks.push_back(std::move(task));
  }
  // A content-derived native digest prevents a caller from forging an
  // existing public execution identity with different launch vectors. CUDA's
  // steady MRU compares this fixed-size value; exact vectors remain a cold
  // validation under the context cache lock.
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(plan.task_kinds);
  serializer(plan.grid_residency_waves);
  serializer(plan.range_work_per_thread_targets);
  serializer.finalize();
  picosha2::hash256(serializer.data, plan.launch_content_digest);
  {
    std::lock_guard<std::mutex> lock(offload_execution_plan_mutex_);
    TI_ERROR_IF(
        offload_execution_plan_frozen_.load(std::memory_order_relaxed),
        "offload execution plan cannot be replaced after launch-context "
        "materialization");
    offload_execution_plan_ = std::move(plan);
    invalidate_kernel_key_for_cache();
  }
}

const std::optional<Kernel::OffloadExecutionPlan> &
Kernel::get_offload_execution_plan() const {
  return offload_execution_plan_;
}

const Kernel::OffloadTaskOptimizationSpec &
Kernel::offload_task_optimization_spec(std::size_t task_index,
                                       OffloadedTaskType task_type) const {
  TI_ERROR_IF(!offload_execution_plan_.has_value(),
              "kernel has no offload execution plan");
  TI_ERROR_IF(task_index >= offload_execution_plan_->tasks.size(),
              "offload execution plan has no task at physical ordinal {}",
              task_index);
  const auto &task = offload_execution_plan_->tasks[task_index];
  const auto actual_kind = offloaded_task_type_name(task_type);
  TI_ERROR_IF(task.task_index != task_index || task.task_kind != actual_kind,
              "offload execution plan topology mismatch at task {}: "
              "expected {}, got {}",
              task_index, task.task_kind, actual_kind);
  return task;
}

void Kernel::set_snode_tree_dependencies(
    const std::vector<int> &dependencies) const {
  std::lock_guard<std::mutex> lock(snode_tree_dependencies_mutex_);
  if (snode_tree_dependency_state_.load(std::memory_order_relaxed) !=
      SNodeTreeDependencyState::unknown) {
    TI_ASSERT(snode_tree_dependencies_ == dependencies);
    return;
  }
  snode_tree_dependencies_ = dependencies;
  snode_tree_dependency_state_.store(
      dependencies.empty() ? SNodeTreeDependencyState::none
                           : SNodeTreeDependencyState::present,
      std::memory_order_release);
}

void Kernel::retire_definition(bool preserve_relocatable_abi) {
  ir.reset();
  context.reset();
  std::vector<SNode *>().swap(no_activate);
  name.clear();
  name.shrink_to_fit();
  offline_cache_body_.reset();
  if (preserve_relocatable_abi) {
    // The immutable callable ABI, source cache key and dependency slots are
    // the frontend half of a verified relocatable template. They contain no
    // SNode pointer or generation-owned allocation. Keeping them lets a new
    // structurally equivalent Field specialization reuse the retired shell
    // without rebuilding Python AST/CHI IR.
    return;
  }
  std::vector<Parameter>().swap(parameter_list);
  decltype(nested_parameters)().swap(nested_parameters);
  decltype(argpack_types)().swap(argpack_types);
  std::vector<Ret>().swap(rets);
  args_type = nullptr;
  args_size = 0;
  ret_type = nullptr;
  ret_size = 0;
  kernel_key_.clear();
  kernel_key_.shrink_to_fit();
  kernel_key_valid_ = false;
  compile_tier_override_.reset();
  task_launch_policy_.reset();
  kernel_optimization_spec_.reset();
  offload_execution_plan_.reset();
  snode_tree_dependency_state_.store(SNodeTreeDependencyState::unknown,
                                     std::memory_order_release);
  std::lock_guard<std::mutex> lock(snode_tree_dependencies_mutex_);
  std::vector<int>().swap(snode_tree_dependencies_);
}

void Kernel::init(Program &program,
                  const std::function<void()> &func,
                  const std::string &primal_name,
                  AutodiffMode autodiff_mode) {
  this->autodiff_mode = autodiff_mode;
  this->program = &program;

  is_accessor = false;
  context = std::make_unique<FrontendContext>(program.compile_config().arch,
                                              /*is_kernel_=*/true);
  ir = context->get_root();

  TI_ASSERT(ir->is<Block>());
  ir->as<Block>()->set_parent_callable(this);

  ir_is_ast_ = true;
  arch = program.compile_config().arch;

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    name = primal_name + "_validate_grad";
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }

  func();
}
}  // namespace taichi::lang
