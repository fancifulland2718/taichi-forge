#include "taichi/runtime/runtime_client.h"

#include <algorithm>
#include <exception>
#include <stdexcept>

namespace taichi::lang {
void RuntimeClientRegistry::attach(
    const std::shared_ptr<RuntimeClient> &client) {
  if (!client)
    throw std::invalid_argument("Cannot attach an empty runtime client");
  std::lock_guard<std::mutex> guard(mutex_);
  if (state_ != State::open)
    throw std::runtime_error("Runtime owner has retired");
  clients_.erase(
      std::remove_if(clients_.begin(), clients_.end(),
                     [](const auto &entry) { return entry.expired(); }),
      clients_.end());
  for (const auto &entry : clients_)
    if (entry.lock() == client)
      throw std::invalid_argument("Runtime client is already attached");
  clients_.push_back(client);
}

void RuntimeClientRegistry::retire() {
  std::vector<std::weak_ptr<RuntimeClient>> clients;
  {
    std::unique_lock<std::mutex> guard(mutex_);
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
    if (auto client = entry.lock()) {
      try {
        client->retire();
      } catch (...) {
        if (!failure)
          failure = std::current_exception();
      }
    }
  }
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
