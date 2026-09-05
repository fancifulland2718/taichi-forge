#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "taichi/aot/graph_data.h"

namespace taichi::lang::cuda {

class GraphBindingExecutor;

// Private native materialization objects, not serialized Graph/AOT payloads.
// Only prepare() uploads arguments. Replay switches a single executable
// between prepared source graphs without writing their argument images again.
class TI_DLL_EXPORT GraphBindingFrame {
 public:
  ~GraphBindingFrame();
  GraphBindingFrame(const GraphBindingFrame &) = delete;
  GraphBindingFrame &operator=(const GraphBindingFrame &) = delete;

 private:
  friend class GraphBindingExecutor;
  struct State;
  GraphBindingFrame();
  std::unique_ptr<State> state_;
};

class TI_DLL_EXPORT GraphBindingExecutor {
 public:
  GraphBindingExecutor(const aot::CompiledGraph &graph,
                       const CompileConfig &config,
                       Program &program_owner);
  ~GraphBindingExecutor();
  GraphBindingExecutor(const GraphBindingExecutor &) = delete;
  GraphBindingExecutor &operator=(const GraphBindingExecutor &) = delete;

  static bool available();

  std::shared_ptr<GraphBindingFrame> prepare(
      const std::unordered_map<std::string, aot::IValue> &args);
  const aot::CompiledGraph &graph() const;
  void run(const std::shared_ptr<GraphBindingFrame> &frame);
  void close();
  std::unordered_map<std::string, std::uint64_t> snapshot();

 private:
  struct State;
  std::shared_ptr<State> state_;
};

}  // namespace taichi::lang::cuda
