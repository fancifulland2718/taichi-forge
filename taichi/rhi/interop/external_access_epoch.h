#pragma once

#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "taichi/rhi/interop/external_sync.h"

namespace taichi::lang {

// Immutable cold binding for a repeatedly submitted complete recipe. Domain
// identity validation/deduplication is done once; each accepted submission
// keeps only a stack count and performs no allocation or ownership scan.
class RHI_DLL_EXPORT PreparedExternalAccess final {
 public:
  explicit PreparedExternalAccess(
      std::vector<std::shared_ptr<ExternalSynchronizationDomain>> domains) {
    domains_.reserve(domains.size());
    for (auto &domain : domains) {
      if (!domain || domain->identity() == 0)
        throw std::invalid_argument("Invalid external synchronization domain");
      bool duplicate = false;
      for (const auto &existing : domains_) {
        if (existing->identity() != domain->identity())
          continue;
        if (existing != domain)
          throw std::invalid_argument(
              "External synchronization identity is not unique");
        duplicate = true;
        break;
      }
      if (!duplicate) {
        if (domain->has_consumer_ready_query())
          ready_domains_.push_back(domain);
        domains_.push_back(std::move(domain));
      }
    }
  }

  class Epoch final {
   public:
    Epoch(const PreparedExternalAccess &plan, ExternalStreamDomain stream)
        : plan_(plan), stream_(stream) {
      if (!stream_.valid())
        throw std::invalid_argument(
            "External access requires a live stream domain");
      try {
        for (const auto &domain : plan_.domains_) {
          domain->acquire_for_consumer(stream_);
          ++acquired_;
        }
      } catch (...) {
        release_noexcept();
        throw;
      }
    }
    Epoch(const Epoch &) = delete;
    Epoch &operator=(const Epoch &) = delete;
    ~Epoch() {
      release_noexcept();
    }
    void release() {
      std::exception_ptr error;
      while (acquired_) {
        try {
          plan_.domains_[--acquired_]->release_from_consumer(stream_);
        } catch (...) {
          if (!error)
            error = std::current_exception();
        }
      }
      if (error)
        std::rethrow_exception(error);
    }

   private:
    void release_noexcept() noexcept {
      try {
        release();
      } catch (...) {
      }
    }
    const PreparedExternalAccess &plan_;
    ExternalStreamDomain stream_;
    std::size_t acquired_{0};
  };

  Epoch acquire(ExternalStreamDomain stream) const {
    return Epoch(*this, stream);
  }
  bool ready() const {
    for (const auto &domain : ready_domains_)
      if (!domain->consumer_ready())
        return false;
    return true;
  }
  std::size_t size() const noexcept {
    return domains_.size();
  }

 private:
  std::vector<std::shared_ptr<ExternalSynchronizationDomain>> domains_;
  std::vector<std::shared_ptr<ExternalSynchronizationDomain>> ready_domains_;
};

// One coarse ownership interval shared by all views from the same external
// synchronization domain. Construction acquires every unique domain before
// consumer work is submitted. release() signals them in reverse order and
// propagates the first failure after attempting every remaining release.
class RHI_DLL_EXPORT ExternalAccessEpoch final {
 public:
  ExternalAccessEpoch() = default;

  ExternalAccessEpoch(
      const std::vector<std::shared_ptr<ExternalSynchronizationDomain>>
          &domains,
      ExternalStreamDomain stream)
      : stream_(stream) {
    if (domains.empty()) {
      return;
    }
    if (!stream_.valid()) {
      throw std::invalid_argument(
          "external access epoch requires a valid consumer stream");
    }

    domains_.reserve(domains.size());
    for (const auto &domain : domains) {
      if (!domain || domain->identity() == 0) {
        release_noexcept();
        throw std::invalid_argument(
            "external access epoch received an invalid synchronization "
            "domain");
      }
      bool duplicate = false;
      for (const auto &existing : domains_) {
        if (existing->identity() != domain->identity()) {
          continue;
        }
        if (existing.get() != domain.get()) {
          release_noexcept();
          throw std::invalid_argument(
              "external synchronization identity is not unique");
        }
        duplicate = true;
        break;
      }
      if (duplicate) {
        continue;
      }
      try {
        domain->acquire_for_consumer(stream_);
        domains_.push_back(domain);
        active_ = true;
      } catch (...) {
        release_noexcept();
        throw;
      }
    }
  }

  ExternalAccessEpoch(const ExternalAccessEpoch &) = delete;
  ExternalAccessEpoch &operator=(const ExternalAccessEpoch &) = delete;

  ExternalAccessEpoch(ExternalAccessEpoch &&other) noexcept
      : stream_(other.stream_),
        domains_(std::move(other.domains_)),
        active_(std::exchange(other.active_, false)) {
    other.stream_ = {};
  }

  ExternalAccessEpoch &operator=(ExternalAccessEpoch &&other) {
    if (this == &other) {
      return *this;
    }
    release();
    stream_ = other.stream_;
    domains_ = std::move(other.domains_);
    active_ = std::exchange(other.active_, false);
    other.stream_ = {};
    return *this;
  }

  ~ExternalAccessEpoch() {
    release_noexcept();
  }

  bool active() const noexcept {
    return active_;
  }

  std::size_t domain_count() const noexcept {
    return domains_.size();
  }

  void release() {
    if (!active_) {
      return;
    }
    active_ = false;
    std::exception_ptr first_error;
    for (auto iter = domains_.rbegin(); iter != domains_.rend(); ++iter) {
      try {
        (*iter)->release_from_consumer(stream_);
      } catch (...) {
        if (!first_error) {
          first_error = std::current_exception();
        }
      }
    }
    domains_.clear();
    stream_ = {};
    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }

 private:
  void release_noexcept() noexcept {
    try {
      release();
    } catch (...) {
    }
  }

  ExternalStreamDomain stream_;
  std::vector<std::shared_ptr<ExternalSynchronizationDomain>> domains_;
  bool active_{false};
};

}  // namespace taichi::lang
