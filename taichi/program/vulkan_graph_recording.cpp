#include <algorithm>

#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/gfx/graph_recording.h"

namespace taichi::lang {
namespace {
bool is_fixed_dense_tree(const SNode &node) {
  if (node.type != SNodeType::root && node.type != SNodeType::dense &&
      node.type != SNodeType::place) {
    return false;
  }
  return std::all_of(node.ch.begin(), node.ch.end(), [](const auto &child) {
    return is_fixed_dense_tree(*child);
  });
}
}  // namespace

bool Program::snode_tree_dependencies_are_fixed_dense(
    const std::vector<SNodeTreeDependency> &dependencies) const {
  validate_snode_tree_dependencies(dependencies, "Prepared Vulkan Graph");
  return std::all_of(dependencies.begin(), dependencies.end(),
                     [this](const auto &dependency) {
                       return is_fixed_dense_tree(
                           *snode_trees_[dependency.tree_id]->root());
                     });
}
}  // namespace taichi::lang

#ifdef TI_WITH_VULKAN
#include "taichi/runtime/gfx/kernel_launcher.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/program/storage_view.h"

namespace taichi::lang::gfx {
aot::CompiledGraph graph_recording_argument_schema(
    const std::vector<GraphRecordingSource> &sources) {
  aot::CompiledGraph result;
  auto merge = [&](const aot::Arg &arg) {
    auto [entry, inserted] = result.args.emplace(arg.name, arg);
    const auto &prior = entry->second;
    const auto is_image = [](aot::ArgKind kind) {
      return kind == aot::ArgKind::kTexture || kind == aot::ArgKind::kRWTexture;
    };
    // Native image uses and a kernel's sampled/storage view can share one
    // Texture binding. Preserve the stricter storage format, while keeping
    // incompatible storage declarations rejected by the ordinary equality test.
    if (!inserted && is_image(arg.tag) && is_image(prior.tag) &&
        arg.element_shape.size() == prior.element_shape.size() &&
        arg.tag != prior.tag) {
      if (arg.tag == aot::ArgKind::kRWTexture) {
        entry->second = arg;
      }
      return;
    }
    TI_ERROR_IF(
        !inserted && arg != prior,
        "Prepared Vulkan Graph has conflicting argument declarations: {}",
        arg.name);
  };
  for (const auto &source : sources) {
    if (const auto *graph = std::get_if<aot::CompiledGraph *>(&source.value)) {
      TI_ERROR_IF(!*graph, "Prepared Vulkan Graph segment is null");
      for (const auto &[name, arg] : (*graph)->args) {
        merge(arg);
      }
    } else {
      const auto &command =
          std::get<std::shared_ptr<ExternalGraphCommand>>(source.value);
      TI_ERROR_IF(!command, "Prepared Vulkan Graph command is null");
      for (const auto &arg : command->arguments()) {
        merge(arg);
      }
    }
  }
  return result;
}

FixedGraphRecording::FixedGraphRecording(
    Program &program,
    std::unique_ptr<GraphReplayRegistration> registration,
    bool has_snode_tree_dependencies)
    : program_(&program),
      has_snode_tree_dependencies_(has_snode_tree_dependencies),
      registration_(std::move(registration)) {
}
FixedGraphRecording::~FixedGraphRecording() = default;
bool FixedGraphRecording::supports_snode_tree_dependencies(
    Program &program,
    const aot::CompiledGraph &graph) {
  auto guard = program.acquire_snode_tree_lifecycle_read_guard();
  const bool fixed_dense = program.snode_tree_dependencies_are_fixed_dense(
      graph.snode_tree_dependencies);
  for (const auto &dispatch : graph.dispatches) {
    TI_ERROR_IF(!dispatch.ti_kernel || dispatch.ti_kernel->program != &program,
                "Prepared Vulkan Graph source belongs to another runtime");
  }
  return fixed_dense;
}
void FixedGraphRecording::run() {
  // Tree destruction owns the writer through its device completion and root
  // retirement transaction. No dependency traversal occurs during replay.
  std::optional<Program::SNodeTreeLifecycleReadGuard> tree_guard;
  if (has_snode_tree_dependencies_) {
    tree_guard.emplace(program_->acquire_snode_tree_lifecycle_read_guard());
  }
  auto guard = program_->acquire_runtime_resource_submission_guard();
  std::lock_guard<std::mutex> lock(mutex_);
  TI_ERROR_IF(!registration_, "Prepared Vulkan Graph is closed");
  registration_->launch_prepared();
  program_->mark_runtime_submission_pending();
}
void FixedGraphRecording::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  registration_.reset();
}
std::uint64_t FixedGraphRecording::argument_bytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return registration_
             ? registration_->snapshot_stats().known_persistent_argument_bytes
             : 0;
}
bool FixedGraphRecording::uses_secondary_commands() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return registration_ && registration_->snapshot_stats().fixed_secondary;
}
}  // namespace taichi::lang::gfx

namespace taichi::lang {
std::shared_ptr<gfx::FixedGraphRecording>
Program::create_vulkan_graph_recording(
    const std::vector<gfx::GraphRecordingSource> &sources,
    const std::unordered_map<std::string, aot::IValue> &args) {
  auto tree_guard = acquire_snode_tree_lifecycle_read_guard();
  auto scope = acquire_runtime_resource_graph_scope();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || compile_config().debug ||
                  compile_config().kernel_profiler,
              "Prepared Vulkan Graph requires non-debug Vulkan execution");
  auto *launcher = dynamic_cast<gfx::KernelLauncher *>(&get_kernel_launcher());
  TI_ERROR_IF(!launcher, "Prepared Vulkan Graph launcher is unavailable");
  std::vector<const Ndarray *> arrays;
  auto textures = std::make_shared<std::vector<TextureResourceLease>>();
  for (const auto &[name, value] : args) {
    if (value.tag == aot::ArgKind::kNdarray) {
      TI_ERROR_IF(!value.val,
                  "Prepared Vulkan Graph requires Program ndarray owners: {}",
                  name);
      const auto *array = reinterpret_cast<const Ndarray *>(value.val);
      TI_ERROR_IF(array->owning_program() != this,
                  "Prepared Vulkan Graph ndarray belongs to another runtime");
      arrays.push_back(array);
      if (value.runtime_storage) {
        TI_ERROR_IF(
            value.runtime_storage->descriptor().owner().kind !=
                    storage::StorageOwnerKind::kProgramNdarray ||
                value.runtime_storage->descriptor().owner().ndarray_handle !=
                    array->runtime_resource_handle(),
            "Prepared Vulkan Graph descriptor must match its retained ndarray "
            "owner");
        const auto *argument = value.runtime_storage;
        retain_runtime_storage_for_graph_submission(&argument, 1);
      }
    } else if (value.tag == aot::ArgKind::kAccelerationStructure) {
      TI_ERROR_IF(value.resource_owner != this || value.val == 0,
                  "Prepared Vulkan Graph AS belongs to another Program: {}", name);
      // The existing kernel binding retains TLAS, BLAS and all backing Vulkan
      // objects in the recorded command, independently of the open handle table.
    } else if (value.tag == aot::ArgKind::kTextureCollection) {
      const auto *collection =
          reinterpret_cast<const TextureCollection *>(value.val);
      TI_ERROR_IF(!collection || collection->owning_program() != this,
                  "Prepared Vulkan Graph TextureCollection belongs to another "
                  "Program: {}",
                  name);
      // This is cold recording preparation. Retain every member once; the
      // immutable recorded descriptor array and leases are reused without a
      // member scan on each replay.
      for (const auto &member : collection->members()) {
        TI_ERROR_IF(member.texture == nullptr ||
                        member.texture->is_cuda_texture(),
                    "Prepared Vulkan Graph requires Vulkan "
                    "TextureCollection members: {}",
                    name);
        textures->push_back(acquire_texture_external_lease(member.texture));
      }
    } else if (value.tag == aot::ArgKind::kTexture) {
      const auto *texture = reinterpret_cast<const Texture *>(value.val);
      TI_ERROR_IF(!texture || texture->owning_program() != this ||
                      texture->is_cuda_texture(),
                  "Prepared Vulkan Graph requires a Program-owned image: {}",
                  name);
      textures->push_back(acquire_texture_external_lease(texture));
    } else {
      TI_ERROR_IF(
          value.tag != aot::ArgKind::kScalar &&
              value.tag != aot::ArgKind::kMatrix,
          "Prepared Vulkan Graph argument kind is unsupported");
    }
  }
  std::vector<std::shared_ptr<void>> owners;
  owners.push_back(std::move(textures));
  owners.push_back(
      std::make_shared<NdarrayLaunchLeases>(acquire_ndarray_leases(arrays)));
  std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
  std::vector<gfx::GfxRuntime::GraphRecordingOperation> operations;
  std::vector<int> tree_ids;
  auto retain_dependencies = [&](const auto &dependencies) {
    TI_ERROR_IF(!snode_tree_dependencies_are_fixed_dense(dependencies),
                "Prepared Vulkan Graph requires fixed dense SNodeTree "
                "dependencies; sparse or packed trees are unsupported");
    for (const auto &dependency : dependencies) {
      tree_ids.push_back(dependency.tree_id);
    }
  };
  for (const auto &source : sources) {
    if (const auto *graph_pointer =
            std::get_if<aot::CompiledGraph *>(&source.value)) {
      const auto &graph = **graph_pointer;
      TI_ERROR_IF(graph.has_indirect_dispatches() ||
                      graph.has_cuda_parallel_dispatch_groups(),
                  "Prepared Vulkan Graph requires flat fixed-dispatch segments");
      retain_dependencies(graph.snode_tree_dependencies);
      // Retain synthetic kernels and their compiled payloads independently of
      // Python builders and subsequent compilation-cache eviction.
      for (auto &kernel : graph.owned_jit_kernels) {
        owners.push_back(kernel);
      }
      for (const auto &dispatch : graph.dispatches) {
        TI_ERROR_IF(!dispatch.ti_kernel ||
                        dispatch.ti_kernel->program != this ||
                        dispatch.cuda_capture_command ||
                        dispatch.cuda_bounded_dispatch ||
                        dispatch.cpu_bounded_dispatch,
                    "Prepared Vulkan Graph cannot lower this dispatch");
        auto kernel = compile_kernel_execution_handle(
            compile_config(), get_device_caps(), *dispatch.ti_kernel);
        auto handle = launcher->get_or_register_kernel(kernel->compiled());
        owners.push_back(std::move(kernel));
        auto context =
            std::make_unique<LaunchContextBuilder>(dispatch.ti_kernel);
        graph.init_runtime_context(dispatch.symbolic_args, args, *context);
        resolve_ndarray_launch_context_under_guard(*context);
        resolve_runtime_storage_launch_context_under_guard(*context);
        resolve_texture_launch_context_under_guard(*context);
        operations.push_back({{handle, context.get()}, {}});
        contexts.push_back(std::move(context));
      }
    } else {
      const auto &command =
          std::get<std::shared_ptr<gfx::ExternalGraphCommand>>(source.value);
      command->validate(*this, args);
      retain_dependencies(command->snode_tree_dependencies());
      owners.push_back(command);
      operations.push_back(
          {{}, [command](Device *device, CommandList *commands) {
             command->record(device, commands);
           }, command->supports_inline_recording(), command->image_uses()});
    }
  }
  std::sort(tree_ids.begin(), tree_ids.end());
  tree_ids.erase(std::unique(tree_ids.begin(), tree_ids.end()), tree_ids.end());
  const bool has_tree_dependencies = !tree_ids.empty();
  auto registration = launcher->runtime()->prepare_fixed_graph(
      operations, std::move(owners), std::move(tree_ids));
  return std::make_shared<gfx::FixedGraphRecording>(*this,
                                                    std::move(registration),
                                                    has_tree_dependencies);
}
}  // namespace taichi::lang
#else
namespace taichi::lang {
std::shared_ptr<gfx::FixedGraphRecording>
Program::create_vulkan_graph_recording(
    const std::vector<gfx::GraphRecordingSource> &,
    const std::unordered_map<std::string, aot::IValue> &) {
  TI_ERROR("Prepared Vulkan Graph is unavailable in this build");
}
}  // namespace taichi::lang
#endif
