#include "taichi/runtime/runtime_client.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <tuple>

namespace taichi::lang {
void RuntimeClientRegistry::attach(
    const std::shared_ptr<RuntimeClient> &client) {
  attach_impl(client, false);
}

void RuntimeClientRegistry::retain(
    const std::shared_ptr<RuntimeClient> &client) {
  attach_impl(client, true);
}

void RuntimeClientRegistry::retain(const std::shared_ptr<RuntimeClient> &client,
                                   std::vector<int> snode_tree_ids) {
  attach_impl(client, true, std::move(snode_tree_ids));
}

void RuntimeClientRegistry::retain_storage(
    const std::shared_ptr<RuntimeClient> &client,
    std::vector<RuntimeResourceHandle> storage,
    std::shared_ptr<void> storage_owner) {
  if (storage.empty())
    throw std::invalid_argument("Runtime storage client requires dependencies");
  attach_impl(client, true, {}, std::move(storage), std::move(storage_owner));
}

void RuntimeClientRegistry::attach_impl(
    const std::shared_ptr<RuntimeClient> &client,
    bool retain,
    std::vector<int> snode_tree_ids,
    std::vector<RuntimeResourceHandle> storage,
    std::shared_ptr<void> storage_owner) {
  if (!client)
    throw std::invalid_argument("Cannot attach an empty runtime client");
  std::sort(snode_tree_ids.begin(), snode_tree_ids.end());
  if (!snode_tree_ids.empty() && snode_tree_ids.front() < 0)
    throw std::invalid_argument("Runtime client has an invalid SNode tree ID");
  snode_tree_ids.erase(
      std::unique(snode_tree_ids.begin(), snode_tree_ids.end()),
      snode_tree_ids.end());
  for (const auto &handle : storage)
    if (!handle)
      throw std::invalid_argument("Runtime client has invalid storage");
  std::sort(storage.begin(), storage.end(), [](const auto &a, const auto &b) {
    return std::tie(a.domain, a.kind, a.index, a.generation) <
           std::tie(b.domain, b.kind, b.index, b.generation);
  });
  storage.erase(std::unique(storage.begin(), storage.end()), storage.end());
  std::lock_guard<std::mutex> guard(mutex_);
  if (state_ != State::open)
    throw std::runtime_error("Runtime owner has retired");
  clients_.erase(
      std::remove_if(clients_.begin(), clients_.end(),
                     [](const auto &entry) { return entry.weak.expired(); }),
      clients_.end());
  for (const auto &entry : clients_)
    if (entry.identity == client.get())
      throw std::invalid_argument("Runtime client is already attached");
  clients_.push_back({client.get(), std::move(storage_owner), client,
                      retain ? client : nullptr, std::move(snode_tree_ids),
                      std::move(storage), false});
}

void RuntimeClientRegistry::detach(RuntimeClient *client) {
  Entry released{};
  {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = std::find_if(
        clients_.begin(), clients_.end(),
        [client](const auto &entry) { return entry.identity == client; });
    if (found != clients_.end()) {
      // A selected client stays retained until its retirement succeeds. A
      // callback's attempted detach must not discard a failed drain closure.
      if (state_ == State::retiring_subset && found->selected)
        return;
      released = std::move(*found);
      clients_.erase(found);
    }
  }
  // A last owner may run arbitrary native cleanup; never release it under the
  // registry mutex. Entries already moved into retire() remain retained there.
}

void RuntimeClientRegistry::retire_snode_tree(int tree_id) {
  if (tree_id < 0)
    throw std::invalid_argument("Cannot retire a negative SNode tree ID");
  retire_selected([&](const auto &entry) {
    return std::binary_search(entry.snode_tree_ids.begin(),
                              entry.snode_tree_ids.end(), tree_id);
  });
}

void RuntimeClientRegistry::retire_storage(RuntimeResourceHandle storage) {
  if (!storage)
    throw std::invalid_argument("Cannot retire invalid runtime storage");
  retire_selected([&](const auto &entry) {
    return std::find(entry.storage.begin(), entry.storage.end(), storage) !=
           entry.storage.end();
  });
}

void RuntimeClientRegistry::retire_selected(
    const std::function<bool(const Entry &)> &select) {
  struct RetainedClient {
    std::shared_ptr<void> storage_owner;
    std::shared_ptr<RuntimeClient> client;
  };
  std::vector<RetainedClient> clients;
  {
    std::unique_lock<std::mutex> guard(mutex_);
    if ((state_ == State::retiring_subset || state_ == State::retiring) &&
        retiring_thread_ == std::this_thread::get_id())
      throw std::logic_error("Reentrant resource client retirement");
    retired_.wait(guard, [&] {
      return state_ != State::retiring_subset && state_ != State::retiring;
    });
    if (state_ == State::closed)
      throw std::runtime_error("Runtime owner has retired");
    // Allocate/snapshot before closing admission. Failure leaves the registry
    // unchanged, and callbacks run with no registry mutex held.
    clients.reserve(clients_.size());
    for (const auto &entry : clients_) {
      if (select(entry)) {
        if (auto client = entry.weak.lock())
          clients.push_back({entry.storage_owner, std::move(client)});
      }
    }
    if (clients.empty())
      return;
    for (auto &entry : clients_)
      entry.selected = select(entry);
    state_ = State::retiring_subset;
    retiring_thread_ = std::this_thread::get_id();
  }
  std::exception_ptr failure;
  for (const auto &selected : clients) {
    const auto &client = selected.client;
    try {
      client->retire();
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
      continue;  // Retain failed clients for retry or whole-owner retirement.
    }
    Entry released{};
    {
      std::lock_guard<std::mutex> guard(mutex_);
      const auto found = std::find_if(
          clients_.begin(), clients_.end(),
          [&](const auto &entry) { return entry.identity == client.get(); });
      if (found != clients_.end()) {
        released = std::move(*found);
        clients_.erase(found);
      }
    }
  }
  // Keep all callback/destructor work inside the retirement transaction, so
  // concurrent owner retirement cannot free the backend while it is in use.
  clients.clear();
  {
    std::lock_guard<std::mutex> guard(mutex_);
    state_ = State::open;
    for (auto &entry : clients_)
      entry.selected = false;
    retiring_thread_ = {};
  }
  retired_.notify_all();
  if (failure)
    std::rethrow_exception(failure);
}

void RuntimeClientRegistry::retire() {
  std::vector<Entry> clients;
  {
    std::unique_lock<std::mutex> guard(mutex_);
    if (state_ == State::retiring_subset) {
      if (retiring_thread_ == std::this_thread::get_id())
        throw std::logic_error(
            "Cannot retire the runtime owner from a resource retirement "
            "callback");
      retired_.wait(guard, [&] { return state_ != State::retiring_subset; });
    }
    if (state_ == State::retiring) {
      // A callback may inspect/close its own registry. A different owner
      // thread must not tear down the backend while callbacks still run.
      if (retiring_thread_ == std::this_thread::get_id())
        return;
      retired_.wait(guard, [&] { return state_ == State::closed; });
    }
    if (state_ == State::closed)
      return;
    state_ = State::retiring;
    retiring_thread_ = std::this_thread::get_id();
    clients.swap(clients_);
  }
  // Never hold the registry lock across backend waits or client cleanup.
  // A failed client cannot prevent the remaining clients from retiring.
  std::exception_ptr failure;
  for (const auto &entry : clients) {
    if (auto client = entry.weak.lock()) {
      try {
        client->retire();
      } catch (...) {
        if (!failure)
          failure = std::current_exception();
      }
    }
  }
  clients.clear();
  {
    std::lock_guard<std::mutex> guard(mutex_);
    state_ = State::closed;
    retiring_thread_ = {};
  }
  retired_.notify_all();
  if (failure)
    std::rethrow_exception(failure);
}

bool RuntimeClientRegistry::closed() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return state_ != State::open;
}
}  // namespace taichi::lang
