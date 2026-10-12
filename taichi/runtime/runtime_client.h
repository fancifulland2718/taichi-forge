#pragma once

#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

namespace taichi::lang {

// An opt-in native client of an existing backend owner. Each client serializes
// retirement against its own submissions, then releases recordings/code/data
// while that backend is still alive. Retirement must not invoke Python.
class RuntimeClient {
 public:
  virtual ~RuntimeClient() = default;
  virtual void retire() = 0;
};

// Cold lifecycle registration only: no device, submission mutex, global state
// or changes to Program's existing submission path. The owner calls retire()
// before backend teardown. Clients own their submission/retirement exclusion.
class RuntimeClientRegistry {
 public:
  void attach(const std::shared_ptr<RuntimeClient> &client);
  void retire();
  bool closed() const;

 private:
  mutable std::mutex mutex_;
  enum class State { open, retiring, closed };
  State state_{State::open};
  std::thread::id retiring_thread_;
  std::condition_variable retired_;
  std::vector<std::weak_ptr<RuntimeClient>> clients_;
};

}  // namespace taichi::lang
