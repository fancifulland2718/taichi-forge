#include "taichi/program/function.h"
#include "taichi/program/program.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/analysis/gather_snode_tree_dependencies.h"

namespace taichi::lang {

Function::Function(Program *program, const FunctionKey &func_key)
    : func_key(func_key) {
  this->program = program;
  arch = program->compile_config().arch;
}

void Function::set_function_body(const std::function<void()> &func) {
  argument_initialization_size = 0;
  context = std::make_unique<FrontendContext>(program->compile_config().arch,
                                              /*is_kernel_=*/false);
  ir = context->get_root();
  ir_stage_ = IRStage::AST;

  TI_ASSERT(ir->is<Block>());
  ir->as<Block>()->set_parent_callable(this);

  func();
  finalize_params();
  finalize_rets();

  // Inline templates also participate in in-memory kernel identity. Without
  // their body, two specializations with different captures can hash alike
  // even when disk caching is disabled.
  if (is_inline_template_ || program->compile_config().offline_cache) {
    std::ostringstream oss;
    gen_offline_cache_key(ir.get(), &oss, &source_snode_tree_dependencies_);
    ast_serialization_data_ = oss.str();
  } else {
    source_snode_tree_dependencies_ =
        irpass::analysis::gather_frontend_snode_tree_dependencies(*ir);
  }
}

void Function::set_function_body(std::unique_ptr<IRNode> func_body) {
  argument_initialization_size = 0;
  ir = std::move(func_body);

  TI_ASSERT(ir->is<Block>());
  ir->as<Block>()->set_parent_callable(this);

  ir_stage_ = IRStage::InitialIR;
  source_snode_tree_dependencies_ =
      irpass::analysis::gather_source_snode_tree_dependencies(*ir);
}

void Function::retire_definition() {
  ir.reset();
  context.reset();
  argument_initialization_size = 0;
  ast_serialization_data_.reset();
  store_dests.clear();
}

std::string Function::get_name() const {
  return func_key.get_full_name();
}

void Function::set_inline_body(const std::function<bool()> &func) {
  is_inline_template_ = true;
  set_function_body([&] { inline_returns_lvalue_ = func(); });
  TI_ASSERT(rets.size() == 1 && rets[0].dt->is<PrimitiveType>());
  irpass::frontend_type_check(ir.get());
  irpass::lower_ast(ir.get());
  irpass::type_check(ir.get(), program->compile_config());
  irpass::analysis::verify(ir.get());
  ir_stage_ = IRStage::InitialIR;
}

}  // namespace taichi::lang
