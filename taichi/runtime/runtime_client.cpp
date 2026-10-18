#include "taichi/runtime/runtime_client.h"

#include <algorithm>
#include <exception>
#include <stdexcept>

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

void RuntimeClientRegistry::attach_impl(
    const std::shared_ptr<RuntimeClient> &client,
    bool retain,
    std::vector<int> snode_tree_ids) {
  if (!client)
    throw std::invalid_argument("Cannot attach an empty runtime client");
  std::sort(snode_tree_ids.begin(), snode_tree_ids.end());
  if (!snode_tree_ids.empty() && snode_tree_ids.front() < 0)
    throw std::invalid_argument("Runtime client has an invalid SNode tree ID");
  snode_tree_ids.erase(
      std::unique(snode_tree_ids.begin(), snode_tree_ids.end()),
      snode_tree_ids.end());
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
  clients_.push_back({client.get(), client, retain ? client : nullptr,
                      std::move(snode_tree_ids)});
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
      if (state_ == State::retiring_tree &&
          std::binary_search(found->snode_tree_ids.begin(),
                             found->snode_tree_ids.end(), retiring_tree_id_))
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
  std::vector<std::shared_ptr<RuntimeClient>> clients;
  {
    std::unique_lock<std::mutex> guard(mutex_);
    if ((state_ == State::retiring_tree || state_ == State::retiring) &&
        retiring_thread_ == std::this_thread::get_id())
      throw std::logic_error("Reentrant SNode client retirement");
    retired_.wait(guard, [&] {
      return state_ != State::retiring_tree && state_ != State::retiring;
    });
    if (state_ == State::closed)
      throw std::runtime_error("Runtime owner has retired");
    // Allocate/snapshot before closing admission. Failure leaves the registry
    // unchanged, and callbacks run with no registry mutex held.
    clients.reserve(clients_.size());
    for (const auto &entry : clients_) {
      if (std::binary_search(entry.snode_tree_ids.begin(),
                             entry.snode_tree_ids.end(), tree_id)) {
        if (auto client = entry.weak.lock())
          clients.emplace_back(std::move(client));
      }
    }
    if (clients.empty())
      return;
    state_ = State::retiring_tree;
    retiring_tree_id_ = tree_id;
    retiring_thread_ = std::this_thread::get_id();
  }
  std::exception_ptr failure;
  for (const auto &client : clients) {
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
    retiring_tree_id_ = -1;
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
    if (state_ == State::retiring_tree) {
      if (retiring_thread_ == std::this_thread::get_id())
        throw std::logic_error(
            "Cannot retire the runtime owner from an SNode retirement "
            "callback");
      retired_.wait(guard, [&] { return state_ != State::retiring_tree; });
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
