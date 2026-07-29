#pragma once

#include <optional>
#include <string>
#include <vector>

#include "taichi/ir/type.h"
#include "taichi/aot/graph_data.h"

namespace taichi::lang {
class Kernel;
class GraphBuilder;

class Node {
 public:
  Node() = default;
  virtual ~Node() = default;
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;
  Node(Node &&) = default;
  Node &operator=(Node &&) = default;

  virtual void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) = 0;
};

class Dispatch : public Node {
 public:
  explicit Dispatch(Kernel *kernel, const std::vector<aot::Arg> &args)
      : kernel_(kernel), symbolic_args_(args) {
  }

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override;

  aot::CompiledDispatch compile_dispatch() const;

  Kernel *kernel() const {
    return kernel_;
  }

  const std::vector<aot::Arg> &symbolic_args() const {
    return symbolic_args_;
  }

 private:
  mutable bool serialized_{false};
  Kernel *kernel_{nullptr};
  std::vector<aot::Arg> symbolic_args_;
};

class Sequential : public Node {
 public:
  explicit Sequential(GraphBuilder *graph) : owning_graph_(graph) {
  }

  void append(Node *node);

  void dispatch(Kernel *kernel, const std::vector<aot::Arg> &args);

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override;

 private:
  std::vector<Node *> sequence_;
  GraphBuilder *owning_graph_{nullptr};
};

class GraphBuilder {
 public:
  explicit GraphBuilder();
  ~GraphBuilder();

  // TODO: compile() can take in Arch argument
  std::unique_ptr<aot::CompiledGraph> compile();

  Node *new_dispatch_node(Kernel *kernel, const std::vector<aot::Arg> &args);

  Sequential *new_sequential_node();

  void dispatch(Kernel *kernel, const std::vector<aot::Arg> &args);

  Sequential *seq() const;

  void enable_two_map_composer() {
    enable_two_map_composer_ = true;
  }

  bool two_map_composer_enabled() const {
    return enable_two_map_composer_;
  }

  std::optional<aot::CompiledDispatch> try_compose_two_maps(
      const Dispatch &first,
      const aot::CompiledDispatch &first_compiled,
      const Dispatch &second,
      const aot::CompiledDispatch &second_compiled);

 private:
  std::unique_ptr<Sequential> seq_{nullptr};
  std::unordered_map<std::string, aot::Arg> all_args_;
  std::vector<std::unique_ptr<Node>> all_nodes_;
  std::vector<std::unique_ptr<Kernel>> composed_kernels_;
  bool enable_two_map_composer_{false};
};

}  // namespace taichi::lang
