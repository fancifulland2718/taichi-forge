#pragma once

#ifdef TI_WITH_VULKAN
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_execution_domain.h"

namespace taichi::lang {

// Cold Vulkan bindings keep queue ownership out of persistent execution plans.
// As with CUDA, one enclosing owner may reserve initialization and replay as one
// logical instance. Code and live operands must outlive the complete instance.
struct FixedExecution::VulkanExecutionBinding {
  std::shared_ptr<vulkan::ExecutionDomain> domain;
  std::shared_ptr<RuntimeCompletionResources> operands;
  std::vector<std::shared_ptr<RuntimeClient>> *retirement_parts{nullptr};
  std::shared_ptr<vulkan::ExecutionInstance> instance;
  // Optional workspace shared by phases of the same logical instance. Its
  // owner records initialization after all phases have been prepared and
  // orders that initialization before their first submission. Without one,
  // FixedExecution initializes its own workspace on its first accepted submit.
  std::shared_ptr<gfx::GfxRuntime::FixedGraphWorkspace> workspace;
  std::size_t parameter_count{0};
};

}  // namespace taichi::lang
#endif
