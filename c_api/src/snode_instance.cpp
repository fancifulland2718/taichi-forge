#include "snode_instance.h"

#include <algorithm>
#include <cstring>
#include <set>
#include "native_resource_registry.h"
#include "taichi/aot/execution_plan.h"

namespace capi {
using namespace taichi::lang;
namespace {
struct InstanceBindingInfo {
  int code_tree, state_tree;
  std::uint64_t code_generation, state_generation;
  TI_IO_DEF(code_tree, state_tree, code_generation, state_generation);
};
}  // namespace

SNodeInstance::SNodeInstance(AotModule &module)
    : code_(module.retain_code()), device_(&module.runtime().get()) {
#ifdef TI_WITH_VULKAN
  if (code_->arch() == taichi::Arch::vulkan)
    return;
#endif
#ifdef TI_WITH_LLVM
  module_ = dynamic_cast<LLVM::LlvmAotModule *>(code_.get());
  if (module_ && (arch_is_cpu(module_->arch()) || module_->arch() == taichi::Arch::cuda)) {
    executor_ = module_->get_runtime_executor();
    return;
  }
#endif
  throw MemoryBindingError("SNode instances require a loaded CPU/CUDA/Vulkan AOT module");
}

void SNodeInstance::initialize(const std::set<std::string> *entries) {
#ifdef TI_WITH_VULKAN
  if (arch() == taichi::Arch::vulkan) {
    gfx_instance_ = entries
        ? gfx::make_aot_snode_instance(code_, std::vector<std::string>(entries->begin(), entries->end()))
        : gfx::make_aot_snode_instance(code_);
    return;
  }
#endif
#ifdef TI_WITH_LLVM
  const auto prototypes = module_->snode_instance_prototypes(entries);
  // Reserve host containers first. A failed clone leaves exactly the
  // successfully initialized prefix owned by this object.
  states_.reserve(prototypes.size());
  bindings_.reserve(prototypes.size());
  reserved_ = executor_->reserve_aot_snode_tree_ids(prototypes.size());
  for (std::size_t i = 0; i < prototypes.size(); ++i) {
    auto state = module_->clone_instance_tree(prototypes[i], reserved_[i]);
    bindings_.push_back({prototypes[i].tree_id, state.tree_id,
                         state.tree_generation, prototypes[i].generation});
    states_.push_back(std::move(state));
  }
#endif
}

SNodeInstance::~SNodeInstance() {
#ifdef TI_WITH_LLVM
  if (!executor_)
    return;
  try {
    // Registry retirement drains this resource's recordings before dropping
    // the last owner. Other instance views keep their own loans and states.
    for (auto it = states_.rbegin(); it != states_.rend(); ++it)
      executor_->destroy_snode_tree_runtime(*it);
    executor_->release_aot_snode_tree_ids(reserved_);
  } catch (...) {
    // Failed retirement cannot justify recycling IDs or prototype backing.
    (void)new std::shared_ptr<aot::Module>(std::move(code_));
  }
#endif
}

taichi::Arch SNodeInstance::arch() const { return code_->arch(); }
Device &SNodeInstance::device() const { return *device_; }

void SNodeInstance::validate_owner(AotModule &module) const {
  if (&module.get() != code_.get())
    throw MemoryBindingError("SNode instance belongs to another loaded module");
#ifdef TI_WITH_LLVM
  if (executor_)
    executor_->validate_snode_tree_instance_bindings(bindings_);
#endif
}

aot::Kernel *SNodeInstance::kernel_view(const std::string &entry) const {
#ifdef TI_WITH_VULKAN
  if (gfx_instance_)
    return gfx_instance_->get_kernel(entry);
#endif
  return nullptr;
}

#ifdef TI_WITH_LLVM
std::vector<LlvmSparseRuntimeViewBinding> SNodeInstance::bindings_for(
    LlvmRuntimeExecutor &executor, const std::vector<int> &code_trees) const {
  if (&executor != executor_)
    throw MemoryBindingError("SNode instance belongs to another runtime");
  std::vector<LlvmSparseRuntimeViewBinding> selected;
  std::set<int> unique(code_trees.begin(), code_trees.end());
  for (int tree : unique) {
    const auto found = std::lower_bound(bindings_.begin(), bindings_.end(), tree,
        [](const auto &binding, int id) { return binding.tree_id < id; });
    if (found == bindings_.end() || found->tree_id != tree)
      throw MemoryBindingError("Kernel tree is outside the SNode instance's selected closure");
    selected.push_back(*found);
  }
  executor_->validate_snode_tree_instance_bindings(selected);
  return selected;
}
#endif

std::string SNodeInstance::info() const {
#ifdef TI_WITH_VULKAN
  if (gfx_instance_) {
    // GFX tree IDs are diagnostic mappings, not generation-bearing handles.
    // The native resource registry owns this instance's lifetime and epochs.
    liong::json::JsonArray result;
    const auto &roots = gfx_instance_->root_bindings();
    for (std::size_t i = 0; i < roots.size(); ++i) {
      if (roots[i] < 0)
        continue;
      liong::json::JsonObject entry;
      entry.inner.emplace("code_tree", static_cast<int>(i));
      entry.inner.emplace("state_tree", roots[i]);
      result.inner.emplace_back(std::move(entry));
    }
    return liong::json::print(liong::json::JsonValue(std::move(result)));
  }
#endif
  std::vector<InstanceBindingInfo> result;
#ifdef TI_WITH_LLVM
  for (const auto &binding : bindings_)
    result.push_back({binding.tree_id, binding.source_tree_id,
                      binding.code_generation, binding.generation});
#endif
  return liong::json::print(liong::json::serialize(result));
}

#ifdef TI_WITH_LLVM
std::shared_ptr<cpu::SparseRuntimeView> SNodeInstance::cpu_view(
    const std::vector<int> &code_trees) const {
  return std::make_shared<cpu::SparseRuntimeView>(
      *executor_, bindings_for(*executor_, code_trees));
}
#endif
}  // namespace capi

TiNativeResource ti_create_aot_snode_instance_ext(TiRuntime runtime,
                                                TiAotModule module) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto *owner = reinterpret_cast<AotModule *>(module);
  if (!rt.owns_module(owner))
    throw MemoryBindingError("SNode instance module is not live in this runtime");
  auto instance = std::make_shared<capi::SNodeInstance>(*owner);
  instance->initialize();
  return rt.native_resource_registry().publish(std::move(instance), {}, {}, {}, true);
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_get_aot_snode_instance_info_ext(TiRuntime runtime, TiNativeResource instance,
                                       uint64_t *size, char *json) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(size);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto state = std::dynamic_pointer_cast<capi::SNodeInstance>(
      rt.native_resource_registry().get(instance));
  if (!state)
    throw MemoryBindingError("Native resource is not an SNode instance");
  const auto text = state->info();
  const auto capacity = *size;
  *size = text.size() + 1;
  if (json) {
    if (capacity < *size)
      throw MemoryBindingError("SNode instance info buffer is too small");
    std::memcpy(json, text.c_str(), *size);
  }
  TI_CAPI_TRY_CATCH_END();
}

TiNativeResource ti_create_aot_snode_instance_for_plan_ext(
    TiRuntime runtime, TiAotModule module, uint64_t description_size,
    const char *description) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module);
  TI_CAPI_ARGUMENT_NULL_RV(description);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto *owner = reinterpret_cast<AotModule *>(module);
  if (!rt.owns_module(owner))
    throw MemoryBindingError("SNode instance module is not live in this runtime");
  const auto plan = taichi::lang::aot::ExecutionPlan::parse(
      std::string(description, description_size));
  const auto entries = plan.kernel_entries();
  auto instance = std::make_shared<capi::SNodeInstance>(*owner);
  instance->initialize(&entries);
  return rt.native_resource_registry().publish(std::move(instance), {}, {}, {}, true);
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_get_aot_snode_plan_layouts_ext(
    TiRuntime runtime, TiAotModule module, uint64_t description_size,
    const char *description, uint64_t *size, char *json) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(module);
  TI_CAPI_ARGUMENT_NULL(description);
  TI_CAPI_ARGUMENT_NULL(size);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto *owner = reinterpret_cast<AotModule *>(module);
  if (!rt.owns_module(owner))
    throw MemoryBindingError("SNode layout module is not live in this runtime");
  const auto plan = taichi::lang::aot::ExecutionPlan::parse(std::string(description, description_size));
  const auto entries = plan.kernel_entries();
  std::string text;
#ifdef TI_WITH_LLVM
  if (auto *llvm = dynamic_cast<taichi::lang::LLVM::LlvmAotModule *>(&owner->get()))
    text = llvm->fixed_snode_plan_layouts(entries);
#endif
#ifdef TI_WITH_VULKAN
  if (rt.arch == taichi::Arch::vulkan)
    text = taichi::lang::gfx::fixed_snode_plan_layouts(
        &owner->get(), std::vector<std::string>(entries.begin(), entries.end()));
#endif
  if (text.empty())
    throw MemoryBindingError("Fixed SNode plan layouts require CPU/CUDA/Vulkan");
  const auto capacity = *size;
  *size = text.size() + 1;
  if (json) {
    if (capacity < *size)
      throw MemoryBindingError("SNode layout buffer is too small");
    std::memcpy(json, text.c_str(), *size);
  }
  TI_CAPI_TRY_CATCH_END();
}
