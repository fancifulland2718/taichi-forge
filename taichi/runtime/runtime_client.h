#pragma once

#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <functional>
#include "taichi/program/runtime_resource_registry.h"

namespace taichi::lang {
class RuntimeCompletion;

// An opt-in native client of an existing backend owner. Each client serializes
// retirement against its own submissions, then releases recordings/code/data
// while that backend is still alive. Retirement must not invoke Python.
class RuntimeClient {
 public:
  virtual ~RuntimeClient() = default;
  virtual void retire() = 0;
  // Domain retirement already proves this exact prefix. Borrowing clients
  // may use it for a device-side return dependency before releasing access.
  // Ordinary clients preserve the existing teardown path.
  virtual void retire_after(const RuntimeCompletion &) { retire(); }
};

// Cold lifecycle registration only: no device, submission mutex, global state
// or changes to Program's existing submission path. The owner calls retire()
// before backend teardown. Clients own their submission/retirement exclusion.
class RuntimeClientRegistry {
 public:
  void attach(const std::shared_ptr<RuntimeClient> &client);
  // Borrowers whose wrapper can disappear during owner teardown stay alive
  // until explicit close/detach or the owner's retirement callback completes.
  void retain(const std::shared_ptr<RuntimeClient> &client);
  // Cold, immutable dependency registration. These clients also retire before
  // any referenced SNode root or its executable bindings are destroyed.
  void retain(const std::shared_ptr<RuntimeClient> &client,
              std::vector<int> snode_tree_ids);
  // Exact external-storage generations, qualified by their owning Program at
  // registration. A retired/reused slot cannot revoke an unrelated borrower.
  void retain_storage(const std::shared_ptr<RuntimeClient> &client,
                      std::vector<RuntimeResourceHandle> storage,
                      std::shared_ptr<void> storage_owner = {});
  void detach(RuntimeClient *client);
  void retire_snode_tree(int tree_id);
  void retire_storage(RuntimeResourceHandle storage);
  void retire();
  bool closed() const;

 private:
  mutable std::mutex mutex_;
  enum class State { open, retiring_subset, retiring, closed };
  State state_{State::open};
  std::thread::id retiring_thread_;
  std::condition_variable retired_;
  struct Entry {
    RuntimeClient *identity;
    // Release storage after the retained client and its native destructor.
    std::shared_ptr<void> storage_owner;
    std::weak_ptr<RuntimeClient> weak;
    std::shared_ptr<RuntimeClient> retained;
    std::vector<int> snode_tree_ids;
    std::vector<RuntimeResourceHandle> storage;
    bool selected{false};
  };
  std::vector<Entry> clients_;
  void attach_impl(const std::shared_ptr<RuntimeClient> &client,
                   bool retain,
                   std::vector<int> snode_tree_ids = {},
                   std::vector<RuntimeResourceHandle> storage = {},
                   std::shared_ptr<void> storage_owner = {});
  void retire_selected(const std::function<bool(const Entry &)> &select);
};

}  // namespace taichi::lang
