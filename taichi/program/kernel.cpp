#include "taichi/program/kernel.h"

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/codegen/codegen.h"
#include "taichi/common/logging.h"
#include "taichi/common/task.h"
#include "taichi/inc/constants.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/bit.h"

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
  return LaunchContextBuilder(this);
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

void Kernel::set_task_launch_policy(const std::string &mode,
                                    int block_dim,
                                    bool injected_block_dim) {
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
  policy.block_dim = block_dim;
  policy.injected_block_dim = injected_block_dim;
  task_launch_policy_ = policy;
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
  return fmt::format("{}:{}:{}", mode, task_launch_policy_->block_dim,
                     task_launch_policy_->injected_block_dim ? 'i' : 's');
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

void Kernel::retire_definition() {
  ir.reset();
  context.reset();
  std::vector<SNode *>().swap(no_activate);
  std::vector<Parameter>().swap(parameter_list);
  decltype(nested_parameters)().swap(nested_parameters);
  decltype(argpack_types)().swap(argpack_types);
  std::vector<Ret>().swap(rets);
  args_type = nullptr;
  args_size = 0;
  ret_type = nullptr;
  ret_size = 0;
  name.clear();
  name.shrink_to_fit();
  kernel_key_.clear();
  kernel_key_.shrink_to_fit();
  kernel_key_valid_ = false;
  offline_cache_body_.reset();
  compile_tier_override_.reset();
  task_launch_policy_.reset();
  snode_tree_dependency_state_.store(SNodeTreeDependencyState::unknown,
                                     std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(snode_tree_dependencies_mutex_);
    std::vector<int>().swap(snode_tree_dependencies_);
  }
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
