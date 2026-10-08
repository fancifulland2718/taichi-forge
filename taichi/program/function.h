#pragma once

#include <unordered_set>
#include "taichi/program/callable.h"
#include "taichi/program/function_key.h"

namespace taichi::lang {

class Program;
class Stmt;

class Function : public Callable {
 public:
  enum class IRStage : int {
    None = 0,
    AST = 1,
    InitialIR = 2,
    BeforeLowerAccess = 3,
    OptimizedIR = 4
  };

  FunctionKey func_key;

  Function(Program *program, const FunctionKey &func_key);

  // Set the function body to a frontend Python function which generates the C++
  // AST.
  void set_function_body(const std::function<void()> &func);

  // Set the function body to a CHI IR.
  void set_function_body(std::unique_ptr<IRNode> func_body);

  // A kernel-owned, scalar inline template. Lower once, clone at each call
  // before autodiff/offloading; this never creates a device function call.
  // The callback reports whether the ordinary inline return is an lvalue.
  void set_inline_body(const std::function<bool()> &func);

  bool is_inline_template() const {
    return is_inline_template_;
  }

  bool inline_returns_lvalue() const {
    return inline_returns_lvalue_;
  }

  [[nodiscard]] std::string get_name() const override;

  const std::optional<std::string> &try_get_ast_serialization_data() const {
    return ast_serialization_data_;
  }

  void set_ir_stage(IRStage type) {
    ir_stage_ = type;
  }

  IRStage ir_stage() const {
    return ir_stage_;
  }

  std::unordered_set<Stmt *> store_dests;

 private:
  bool is_inline_template_{false};
  bool inline_returns_lvalue_{false};
  IRStage ir_stage_{IRStage::None};
  std::optional<std::string> ast_serialization_data_;  // For generating AST-Key
};

}  // namespace taichi::lang
