#pragma once

#include "execution_resources.h"
#include "taichi/aot/execution_plan.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_operation_registry.h"

namespace capi {

// Cold family facts and shared private state. Published calls own this context;
// it never owns the calls or their backend recordings, avoiding a replay cycle.
struct ExecutionBindingFamily {
  Runtime *runtime;
  AotModule *module;
  std::uint64_t identity;
  taichi::lang::aot::ExecutionPlan plan;
  taichi::lang::aot::ExecutionResources frame_resources;
  std::set<std::string> frame_arguments;
  std::unordered_map<std::string, TiArgument> original_arguments;
  std::unordered_set<std::string> persistent_public_arguments;
  std::shared_ptr<SharedExecutionResources> arrays;
  std::unique_ptr<taichi::lang::NativePreparationContext> preparation;
  std::map<std::string, std::shared_ptr<taichi::lang::NativeResource>> objects;
  std::unordered_set<TiMemory> initialization_memories;
  std::unordered_set<TiMemory> persistent_memories;
  std::unordered_set<TiNativeResource> initialization_external_resources;
  std::unordered_set<TiNativeResource> persistent_external_resources;
  taichi::lang::NativeRecordingState initial_facts;
  std::shared_ptr<void> initialization_operands;
  std::unique_ptr<taichi::lang::FixedExecution> initialization;
  std::shared_ptr<taichi::lang::cuda::BindingFrameReplay> replay;
  bool prepared{false};
  bool initialized{false};
  bool failed{false};

  ExecutionBindingFamily(Runtime &runtime,
                         AotModule &module,
                         taichi::lang::aot::ExecutionPlan plan);
  ~ExecutionBindingFamily();
  void initialize();
};

}  // namespace capi
