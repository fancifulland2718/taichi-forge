#pragma once

#include <optional>
#include <string>

#include "taichi/util/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/callable.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/aot/graph_data.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {

class Program;

class TI_DLL_EXPORT Kernel : public Callable {
 public:
  std::vector<SNode *> no_activate;

  bool is_accessor{false};

  Kernel(Program &program,
         const std::function<void()> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         const std::function<void(Kernel *)> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         std::unique_ptr<IRNode> &&ir,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  bool ir_is_ast() const {
    return ir_is_ast_;
  }

  LaunchContextBuilder make_launch_context();

  template <typename T>
  T fetch_ret(DataType dt, int i);

  [[nodiscard]] std::string get_name() const override;

  void set_kernel_key_for_cache(const std::string &kernel_key) const {
    kernel_key_ = kernel_key;
  }

  const std::string &get_cached_kernel_key() const {
    return kernel_key_;
  }

  // P-Compile-6: per-kernel compile_tier override.
  // When set, takes precedence over CompileConfig::compile_tier for this
  // kernel only. Empty optional = use program-level compile_tier (default).
  // Valid values: "fast", "balanced", "full". Invalid values are rejected at
  // the Python boundary; C++ side stores the string verbatim.
  void set_compile_tier_override(const std::string &tier) {
    compile_tier_override_ = tier;
  }

  void clear_compile_tier_override() {
    compile_tier_override_.reset();
  }

  const std::optional<std::string> &get_compile_tier_override() const {
    return compile_tier_override_;
  }

 private:
  void init(Program &program,
            const std::function<void()> &func,
            const std::string &name = "",
            AutodiffMode autodiff_mode = AutodiffMode::kNone);

  // True if |ir| is a frontend AST. False if it's already offloaded to CHI IR.
  bool ir_is_ast_{false};
  mutable std::string kernel_key_;
  std::optional<std::string> compile_tier_override_;
};

}  // namespace taichi::lang
