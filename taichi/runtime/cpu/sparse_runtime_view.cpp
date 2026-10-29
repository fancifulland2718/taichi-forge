#include "taichi/runtime/cpu/sparse_runtime_view.h"

#include <algorithm>
#include <limits>
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/util/str.h"

namespace taichi::lang::cpu {
struct SparseRuntimeView::State {
  std::shared_ptr<SparseTreeAdmission::Loan> loan;
  std::vector<LlvmSparseRuntimeViewBinding> bindings;
  std::vector<std::uint64_t> storage, temporaries, results;
  std::vector<void *> directory;
  JITModule *module{nullptr};
  std::function<void(LLVMRuntime *)> query_error;
  std::function<void(LLVMRuntime *, LlvmSparseRuntimeDiagnostic *)> snapshot;
  LLVMRuntime *source{nullptr}, *view{nullptr};
  bool bound{false};

  void retire() {
    if (!bound)
      return;
    LlvmSparseRuntimeViewStatus status;
    module->call("runtime_sparse_view_release_mapped", view, bindings.data(),
                 int(bindings.size()), &status);
    // Returning managers is a preflighted all-or-nothing transaction. Never
    // free their runtime directory if an unexpected retirement error occurs.
    if (status.error != LlvmSparseRuntimeViewError::success)
      throw std::runtime_error("CPU SNode view could not return its managers");
    bound = false;
  }
};

SparseRuntimeView::SparseRuntimeView(
    LlvmRuntimeExecutor &executor,
    const std::vector<LlvmSparseRuntimeViewBinding> &bindings)
    : state_(std::make_unique<State>()) {
  if (!arch_is_cpu(executor.get_config().arch) ||
      bindings.size() > std::size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("CPU SNode view requires a live CPU executor");
  executor.validate_snode_tree_instance_bindings(bindings);
  auto &s = *state_;
  s.bindings = bindings;
  s.source = executor.get_llvm_runtime();
  s.module = executor.get_runtime_jit_module();
  s.query_error = s.module->get_function<LLVMRuntime *>(
      "runtime_retrieve_and_reset_error_code");
  s.snapshot = s.module->get_function<LLVMRuntime *, LlvmSparseRuntimeDiagnostic *>(
      "runtime_sparse_view_snapshot");
  std::vector<LlvmSparseRuntimeViewTree> trees;
  for (const auto &binding : bindings)
    trees.push_back({binding.source_tree_id, 0, binding.generation});
  std::sort(trees.begin(), trees.end(),
            [](const auto &a, const auto &b) { return a.tree_id < b.tree_id; });
  s.loan = executor.reserve_cpu_sparse_trees(trees);
  std::uint64_t bytes = 0;
  s.module->call("runtime_sparse_view_get_size", &bytes);
  if (!bytes || bytes % sizeof(std::uint64_t))
    throw std::invalid_argument("Invalid CPU sparse runtime size");
  s.storage.resize(bytes / sizeof(std::uint64_t));
  s.temporaries.resize(taichi_global_tmp_buffer_size / sizeof(std::uint64_t));
  s.results.resize(taichi_result_buffer_entries);
  s.directory.resize(bindings.empty() ? 0 : bindings.back().tree_id + 1);
  s.view = reinterpret_cast<LLVMRuntime *>(s.storage.data());
  LlvmSparseRuntimeViewStatus status;
  const auto check = [&] {
    if (status.error != LlvmSparseRuntimeViewError::success)
      throw std::invalid_argument(fmt::format(
          "CPU SNode view rejected tree {} (status {})", status.tree_id,
          static_cast<int>(status.error)));
  };
  s.module->call("runtime_sparse_view_initialize", s.source, s.view,
                 s.directory.data(), int(s.directory.size()),
                 s.temporaries.data(), s.results.data(), &status);
  check();
  s.module->call("runtime_sparse_view_bind_mapped", s.view, s.bindings.data(),
                 int(s.bindings.size()), &status);
  check();
  s.bound = true;
}

SparseRuntimeView::~SparseRuntimeView() {
  try {
    state_->retire();
  } catch (...) {
    // Match the runtime's failed-retirement policy: keep loans and directory
    // alive instead of leaving managers pointing into freed storage.
    state_.release();
  }
}

LLVMRuntime *SparseRuntimeView::bind(LLVMRuntime *source,
                                    const std::vector<int> &trees) const {
  const auto &s = *state_;
  if (!s.bound || source != s.source)
    throw std::invalid_argument("CPU SNode view belongs to another runtime");
  for (int tree : trees)
    if (!std::binary_search(
            s.bindings.begin(), s.bindings.end(),
            LlvmSparseRuntimeViewBinding{tree, 0, 0, 0},
            [](const auto &a, const auto &b) { return a.tree_id < b.tree_id; }))
      throw std::invalid_argument("CPU SNode view omits a selected code tree");
  return s.view;
}

void SparseRuntimeView::check_error() const {
  auto &s = *state_;
  s.query_error(s.view);
  const auto code = s.results[taichi_result_buffer_error_id];
  if (!code)
    return;
  LlvmSparseRuntimeDiagnostic snapshot;
  s.snapshot(s.view, &snapshot);
  snapshot.code = code;
  if (snapshot.code != 1)
    throw std::runtime_error("Unknown CPU SNode runtime diagnostic code");
  const auto end = std::find(std::begin(snapshot.message),
                             std::end(snapshot.message), '\0');
  throw TaichiAssertionError(format_error_message(
      std::string(std::begin(snapshot.message), end), [&](int index) {
        if (index < 0 || index >= taichi_error_message_max_num_arguments)
          throw std::runtime_error("Invalid CPU SNode diagnostic argument");
        return snapshot.arguments[index];
      }));
}
}  // namespace taichi::lang::cpu
