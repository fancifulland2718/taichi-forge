#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "taichi/common/core.h"
#include "taichi/common/logging.h"
#include "taichi/rhi/arch.h"
#include "taichi/rhi/public_device.h"

namespace taichi::lang {

// Type-erased owner for resources that must outlive a backend submission.
// Program provides the concrete batch; RuntimeCompletion only controls when
// that batch may be released.
class TI_DLL_EXPORT RuntimeCompletionResources {
 public:
  virtual ~RuntimeCompletionResources() = default;
  virtual std::size_t retained_resource_count(
      std::uint32_t kind) const noexcept = 0;
};

// Internal, backend-neutral completion token. Metadata is stored inline so a
// synchronously completed/no-work token reuses one process-wide state without
// allocating. Pending states are shared because Program and an eventual
// opt-in SubmissionTicket must observe the same first error and resource
// retirement.
class TI_DLL_EXPORT RuntimeCompletion {
 public:
  RuntimeCompletion() = default;

  static RuntimeCompletion completed(Arch backend,
                                     std::uint64_t program_domain,
                                     std::uint64_t sequence) noexcept;
  static RuntimeCompletion from_stream_semaphore(
      Arch backend,
      std::uint64_t program_domain,
      std::uint64_t sequence,
      StreamSemaphore semaphore);
  static RuntimeCompletion from_cuda_stream(std::uint64_t program_domain,
                                            std::uint64_t sequence,
                                            void *stream);

  bool valid() const noexcept;
  bool done() const;
  void wait() const;

  Arch backend() const noexcept {
    return backend_;
  }
  std::uint64_t program_domain() const noexcept {
    return program_domain_;
  }
  std::uint64_t sequence() const noexcept {
    return sequence_;
  }
  bool has_backend_work() const noexcept;
  std::size_t retained_resource_count(std::uint32_t kind) const noexcept;
  std::string first_error_message() const;

  // Internal Program hooks. Attaching is allowed exactly once while pending.
  void attach_resources(
      std::shared_ptr<RuntimeCompletionResources> resources) const;
  void mark_completed() const noexcept;
  void invalidate(const std::string &reason) const noexcept;

 private:
  struct State;

  RuntimeCompletion(Arch backend,
                    std::uint64_t program_domain,
                    std::uint64_t sequence,
                    std::shared_ptr<State> state) noexcept;

  Arch backend_{Arch::x64};
  std::uint64_t program_domain_{0};
  std::uint64_t sequence_{0};
  std::shared_ptr<State> state_;
};

}  // namespace taichi::lang
