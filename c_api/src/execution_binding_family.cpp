#include "execution_binding_family.h"
#include "native_resource_registry.h"

#include "taichi/program/runtime_fault.h"
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/binding_frame_replay.h"
#endif

namespace capi {
using namespace taichi::lang;

ExecutionBindingFamily::ExecutionBindingFamily(Runtime &owner,
                                               AotModule &code,
                                               aot::ExecutionPlan description)
    : runtime(&owner),
      module(&code),
      identity(Runtime::new_handle_token()),
      plan(std::move(description)),
      frame_resources(plan.resources),
      frame_arguments(plan.replay_bindings),
      arrays(std::make_shared<SharedExecutionResources>()) {
  if (runtime->arch != taichi::Arch::cuda || !plan.cuda_binding_frames)
    throw MemoryBindingError(
        "CUDA binding family requires its selected CUDA execution strategy");
#ifdef TI_WITH_CUDA
  replay = std::make_shared<cuda::BindingFrameReplay>(
      identity, runtime->runtime_fault_domain());
#else
  throw MemoryBindingError(
      "CUDA binding families are unavailable in this runtime");
#endif
  preparation = make_native_preparation_context(*runtime);
  for (auto gradient = frame_resources.gradients.begin();
       gradient != frame_resources.gradients.end();) {
    if (!plan.replay_bindings.count(gradient->first))
      gradient = frame_resources.gradients.erase(gradient);
    else
      ++gradient;
  }
  frame_resources.analyze(frame_arguments);
  std::set<std::string> native_resources;
  std::unordered_set<std::string> private_names;
  for (const auto index : plan.resources.order) {
    const auto &resource = plan.resources.declarations[index];
    private_names.insert(resource.name);
    if (std::holds_alternative<NativeOperationDescription>(resource.value) &&
        plan.replay_bindings.count(resource.name))
      native_resources.insert(resource.name);
  }
  for (const auto &name :
       plan.resources.runtime_bindings(std::move(native_resources))) {
    arrays->stable_names.insert(name);
    if (!private_names.count(name))
      persistent_public_arguments.insert(name);
  }
}

ExecutionBindingFamily::~ExecutionBindingFamily() {
  // Prepared-call retirement completes outstanding work before dropping its
  // family. An unpublished family can also contain capture preparation work.
  initialization.reset();
  replay.reset();
  initialization_operands.reset();
  objects.clear();
}

void ExecutionBindingFamily::initialize() {
  if (failed)
    throw MemoryBindingError(
        "Binding family initialization failed; prepare a new family");
  if (initialized)
    return;
  try {
    if (initialization) {
      initialization->run();
      runtime->wait();
      initialization.reset();
    }
    initialized = true;
    initialization_operands.reset();
  } catch (...) {
    failed = true;
    throw;
  }
}

}  // namespace capi
