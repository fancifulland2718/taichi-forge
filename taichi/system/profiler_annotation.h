#pragma once

#include <string>
#include <cstdint>

namespace taichi::lang {

// Thread-local label used by an ordinary kernel invocation. The returned
// previous value makes the Python context manager naturally nestable.
std::string push_dispatch_label(std::string label);
void restore_dispatch_label(std::string label);
const std::string *current_dispatch_label() noexcept;

void validate_dispatch_label(const std::string &label);

// Keep the historical task name first so existing profiler aggregation by
// kernel-name prefix remains valid while Nsight/NVTX users gain a stable task
// identity and per-dispatch label.
std::string make_labeled_task_name(const std::string &task_name,
                                   const std::string &task_id,
                                   const std::string &dispatch_label);

// Explicit, same-thread ranges. NVTX 3 is header-only and a no-op without an
// attached tool. Unlabeled dispatch/replay never enters this annotation path.
void push_external_profiler_range(const std::string &name,
                                  uint32_t category,
                                  uint64_t payload);
void pop_external_profiler_range();

// Existing labeled tasks use the same domain as explicit Graph search ranges.
class ScopedExternalProfilerAnnotation {
 public:
  explicit ScopedExternalProfilerAnnotation(const std::string &name);
  ~ScopedExternalProfilerAnnotation();

  ScopedExternalProfilerAnnotation(
      const ScopedExternalProfilerAnnotation &) = delete;
  ScopedExternalProfilerAnnotation &operator=(
      const ScopedExternalProfilerAnnotation &) = delete;

};

// CPU profiling starts inside generated LLVM code with a static task name.
// A labeled launch installs this host-side override around exactly one task.
class ScopedKernelProfilerName {
 public:
  explicit ScopedKernelProfilerName(const std::string &name) noexcept;
  ~ScopedKernelProfilerName();

  ScopedKernelProfilerName(const ScopedKernelProfilerName &) = delete;
  ScopedKernelProfilerName &operator=(const ScopedKernelProfilerName &) =
      delete;

  static const std::string *current() noexcept;

 private:
  const std::string *previous_{nullptr};
};

}  // namespace taichi::lang
