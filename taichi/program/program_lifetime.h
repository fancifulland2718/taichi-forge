#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>

namespace taichi::lang {
class Program;

class ProgramLifetimeToken {
 public:
  explicit ProgramLifetimeToken(Program *program) : program_(program) {
  }

  // A delayed owner may wait for a Graph or call a producer's release hook.
  // Pin the Program across that work without holding the lifetime mutex.
  class Use {
   public:
    Use(const std::weak_ptr<ProgramLifetimeToken> &lifetime, Program *program)
        : token_(lifetime.lock()) {
      if (token_) {
        std::lock_guard<std::mutex> lock(token_->mutex_);
        if (program && token_->program_ == program) {
          ++token_->users_;
          admitted_ = true;
        }
      }
    }

    Use(const Use &) = delete;
    Use &operator=(const Use &) = delete;

    ~Use() {
      if (admitted_) {
        std::lock_guard<std::mutex> lock(token_->mutex_);
        if (--token_->users_ == 0 && !token_->program_)
          token_->drained_.notify_one();
      }
    }

    explicit operator bool() const noexcept {
      return admitted_;
    }

   private:
    std::shared_ptr<ProgramLifetimeToken> token_;
    bool admitted_{false};
  };

  void invalidate() {
    std::unique_lock<std::mutex> lock(mutex_);
    program_ = nullptr;
    drained_.wait(lock, [&] { return users_ == 0; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable drained_;
  Program *program_;
  std::size_t users_{0};
};
}  // namespace taichi::lang
