#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <variant>

#include "taichi/aot/graph_data.h"
#include "taichi/rhi/device.h"

namespace taichi::lang::gfx {

// Internal JIT materializer contract. Validation and recording happen once at
// binding publication; replay contains neither a provider call nor this object.
class TI_DLL_EXPORT ExternalGraphCommand {
 public:
  virtual ~ExternalGraphCommand() = default;
  virtual std::vector<aot::Arg> arguments() const = 0;
  virtual void validate(
      Program &program,
      const std::unordered_map<std::string, aot::IValue> &args) const = 0;
  virtual void record(Device *device, CommandList *commands) const = 0;
  // Fixed image uses resolved during cold binding validation. The enclosing
  // recorder owns layout transitions and closes the replay layout cycle.
  virtual std::vector<std::pair<DeviceAllocation, ImageLayout>> image_uses() const {
    return {};
  }
  virtual bool supports_inline_recording() const {
    return false;
  }
  virtual bool requires_graphics_queue() const {
    return false;
  }
  // Cold invalidation dependencies, never traversed during replay.
  virtual std::vector<std::uint64_t> graphics_pipeline_dependencies() const {
    return {};
  }
  virtual std::vector<std::uint64_t> ray_resource_dependencies() const {
    return {};
  }
  virtual std::optional<std::vector<DeviceAllocation>> buffer_uses() const {
    return std::nullopt;
  }
  // Static dense storage owners, resolved during cold command preparation.
  // The enclosing recorder retains roots and retires on tree destruction.
  virtual std::vector<SNodeTreeDependency> snode_tree_dependencies() const {
    return {};
  }
};

struct GraphRecordingSource {
  std::variant<aot::CompiledGraph *, std::shared_ptr<ExternalGraphCommand>>
      value;
};

class GraphReplayRegistration;
class TI_DLL_EXPORT FixedGraphRecording {
 public:
  FixedGraphRecording(Program &program,
                      std::unique_ptr<GraphReplayRegistration> registration,
                      bool has_snode_tree_dependencies,
                      bool uses_graphics_queue = false,
                      bool independent_graphics = false);
  ~FixedGraphRecording();
  static bool supports_snode_tree_dependencies(Program &program,
                                              const aot::CompiledGraph &graph);
  static bool supports_graphics_queue(Program &program);
  void run();
  void close();
  std::uint64_t argument_bytes() const;
  bool uses_secondary_commands() const;
  bool uses_graphics_queue() const {
    return uses_graphics_queue_;
  }
  bool uses_independent_graphics() const {
    return independent_graphics_;
  }

 private:
  Program *program_;
  bool has_snode_tree_dependencies_{false};
  bool uses_graphics_queue_{false};
  bool independent_graphics_{false};
  mutable std::mutex mutex_;
  std::unique_ptr<GraphReplayRegistration> registration_;
};

// Conversion schema only. This object is never used as an executable or AOT
// artifact; native commands remain separate from serialized Graph dispatches.
TI_DLL_EXPORT aot::CompiledGraph graph_recording_argument_schema(
    const std::vector<GraphRecordingSource> &sources);

}  // namespace taichi::lang::gfx
