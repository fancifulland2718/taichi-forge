#pragma once

// Shared-read/exclusive admission for owner-qualified runtime resources.
// Backend adapters publish their exact reusable producer boundary and retain
// loans through command retirement; this state machine does not record GPU work.
#include "taichi/program/runtime_resource_registry.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace taichi::lang {

// Adapters supply the complete effects of a recording. Restoring an entry
// state at its end does not turn an intervening write into a shared read.
enum class SharedResourceUse { read, exclusive };

struct SharedResourceAdmissionName {
  static constexpr const char *value = "resource";
};

// Boundary is an existing completion, not a new synchronization primitive.
// Name only qualifies cold diagnostics; it adds no object or submission state.
template <class Boundary, class Name = SharedResourceAdmissionName>
class SharedResourceAdmission {
  friend struct SharedResourceAdmissionTestAccess;

 public:
  struct Use {
    RuntimeResourceHandle resource;
    SharedResourceUse access;
  };

  // The adapter pins the existing owner. The table itself must not own that
  // lease: the owner destructor may be responsible for unregistering its key.
  struct PinnedUse {
    Use use;
    std::shared_ptr<void> owner;
  };

 private:
  struct HandleHash {
    std::size_t operator()(RuntimeResourceHandle h) const noexcept {
      std::size_t result = std::hash<std::uint64_t>{}(h.domain);
      for (auto word : {h.kind, h.index, h.generation})
        result ^= std::hash<std::uint32_t>{}(word) + 0x9e3779b9u +
                  (result << 6) + (result >> 2);
      return result;
    }
  };
  struct PermitState;
  struct Entry {
    RuntimeResourceHandle resource;
    std::size_t readers{0};
    bool exclusive{false};
    bool ready{false};
    std::weak_ptr<const Boundary> boundary;
    struct Edge {
      std::weak_ptr<PermitState> permit;
      SharedResourceUse access;
    };
    std::vector<Edge> recordings;
  };
  struct BoundUse {
    std::shared_ptr<Entry> entry;
    SharedResourceUse access;
    std::shared_ptr<void> owner;
    std::shared_ptr<const Boundary> boundary;
  };
  struct PermitState {
    std::size_t blocked_resources{0};
    std::size_t removed_resources{0};
  };
  struct Control {
    explicit Control(std::shared_ptr<std::recursive_mutex> gate)
        : gate(std::move(gate)) {
    }
    std::shared_ptr<std::recursive_mutex> gate;
    bool closed{false};
    std::size_t submitting{0};
    std::unordered_map<RuntimeResourceHandle, std::shared_ptr<Entry>, HandleHash>
        resources;
  };
  struct Submitting {
    explicit Submitting(Control &control) : control(control) {
      ++control.submitting;
    }
    ~Submitting() {
      --control.submitting;
    }
    Control &control;
  };

  static bool blocked(const Entry &entry, SharedResourceUse use) noexcept {
    return entry.exclusive ||
           (entry.readers && (use == SharedResourceUse::exclusive || !entry.ready));
  }

  // Allocation-free once the reservation is published. Weak edges avoid a
  // cycle between resource state and immutable recording owners.
  template <class Mutation>
  static void mutate(Entry &entry, Mutation mutation) noexcept {
    const bool was_read_blocked = blocked(entry, SharedResourceUse::read);
    const bool was_write_blocked = blocked(entry, SharedResourceUse::exclusive);
    mutation();
    auto next = entry.recordings.begin();
    for (auto it = entry.recordings.begin(); it != entry.recordings.end(); ++it) {
      if (auto permit = it->permit.lock()) {
        const bool before = it->access == SharedResourceUse::exclusive
                                ? was_write_blocked
                                : was_read_blocked;
        const bool after = blocked(entry, it->access);
        if (before && !after)
          --permit->blocked_resources;
        else if (!before && after)
          ++permit->blocked_resources;
        *next++ = *it;
      }
    }
    entry.recordings.erase(next, entry.recordings.end());
  }

  static std::string diagnostic(const char *reason) {
    return std::string(Name::value) + " loan: " + reason;
  }

  static void require_open(const Control &control) {
    if (control.closed)
      throw std::invalid_argument(
          diagnostic("Shared resource owner is closed"));
  }

  static std::shared_ptr<Entry> find(const Control &control,
                                     RuntimeResourceHandle resource) {
    const auto found = control.resources.find(resource);
    if (!resource || found == control.resources.end())
      throw std::invalid_argument(
          diagnostic("Shared resource identity is stale or foreign"));
    return found->second;
  }

  static std::vector<BoundUse> bind(const Control &control,
                                    const std::vector<PinnedUse> &uses) {
    std::vector<BoundUse> result;
    result.reserve(uses.size());
    for (const auto &use : uses) {
      if (!use.owner)
        throw std::invalid_argument(
            diagnostic("Shared resource has no lifetime owner"));
      auto entry = find(control, use.use.resource);
      auto duplicate =
          std::find_if(result.begin(), result.end(),
                       [&](const auto &item) { return item.entry == entry; });
      if (duplicate == result.end())
        result.push_back({std::move(entry), use.use.access, use.owner, {}});
      else if (use.use.access == SharedResourceUse::exclusive)
        duplicate->access = SharedResourceUse::exclusive;
    }
    return result;
  }

 public:
  // Opaque recorder callbacks may reenter host APIs. Keep the existing host
  // gate and submission depth for the entire append transaction so a callback
  // cannot establish a resource loan between its preflight and command append.
  class OrdinaryScope {
   public:
    OrdinaryScope(const OrdinaryScope &) = delete;
    OrdinaryScope &operator=(const OrdinaryScope &) = delete;

   private:
    friend class SharedResourceAdmission;
    explicit OrdinaryScope(std::shared_ptr<Control> control)
        : control_(std::move(control)),
          lock_(*control_->gate),
          submitting_(*control_) {
      require_open(*control_);
    }
    std::shared_ptr<Control> control_;
    std::unique_lock<std::recursive_mutex> lock_;
    Submitting submitting_;
  };

  OrdinaryScope ordinary_scope() {
    return OrdinaryScope(control_);
  }

  class Permit : public std::enable_shared_from_this<Permit> {
   public:
    // A whole Graph may have preparation uploads, ordinary fallbacks and
    // native replay exits. Keep one admission transaction around all of them.
    // Moving the scope transfers its submission depth; the scope also retains
    // the permit's allocation owners until after the host gate is released.
    class SubmissionScope {
     public:
      SubmissionScope(const SubmissionScope &) = delete;
      SubmissionScope &operator=(const SubmissionScope &) = delete;
      SubmissionScope(SubmissionScope &&) noexcept = default;
      SubmissionScope &operator=(SubmissionScope &&) = delete;
      ~SubmissionScope() {
        if (owner_)
          --owner_->control_->submitting;
      }

     private:
      friend class Permit;
      explicit SubmissionScope(std::shared_ptr<const Permit> owner)
          : owner_(std::move(owner)), lock_(*owner_->control_->gate) {
        owner_->require_allowed();
        ++owner_->control_->submitting;
      }
      std::shared_ptr<const Permit> owner_;
      std::unique_lock<std::recursive_mutex> lock_;
    };

    SubmissionScope submission_scope() const {
      return SubmissionScope(this->shared_from_this());
    }

    // The same host gate must cover check + command append/submit. A separate
    // allowed() test followed by an unlocked submit has a reservation race.
    template <class Submit>
    decltype(auto) submit(Submit &&submit) const {
      std::lock_guard<std::recursive_mutex> lock(*control_->gate);
      require_allowed();
      // Constant work with respect to resource count; no resource/owner scan.
      Submitting scope(*control_);
      return std::forward<Submit>(submit)();
    }

   private:
    friend class SharedResourceAdmission;
    void require_allowed() const {
      require_open(*control_);
      if (state_->removed_resources || state_->blocked_resources)
        throw std::invalid_argument(
            diagnostic("Prepared recording conflicts with a resource loan"));
    }
    Permit(std::shared_ptr<Control> control,
           std::shared_ptr<PermitState> state,
           std::vector<BoundUse> uses)
        : control_(std::move(control)),
          state_(std::move(state)),
          uses_(std::move(uses)) {
    }
    std::shared_ptr<Control> control_;
    std::shared_ptr<PermitState> state_;
    // Reverse-edge weak_ptr::lock() must not temporarily become the last owner
    // of a resource lease while holding the host gate. Keep owners in Permit,
    // never in the weakly observed PermitState.
    std::vector<BoundUse> uses_;
  };

  class Loan {
   public:
    Loan(const Loan &) = delete;
    Loan &operator=(const Loan &) = delete;
    ~Loan() {
      if (active_) {
        std::lock_guard<std::recursive_mutex> lock(*control_->gate);
        for (const auto &use : uses_)
          mutate(*use.entry, [&] {
            if (use.access == SharedResourceUse::exclusive)
              use.entry->exclusive = false;
            else
              --use.entry->readers;
            if (!use.entry->exclusive && !use.entry->readers) {
              use.entry->ready = false;
              use.entry->boundary.reset();
            }
          });
      }
      // uses_ releases lifetime owners AFTER the gate above is released.
      // Domain command retirement, not close() of a facade, drops this object.
    }

    const std::vector<RuntimeResourceHandle> &needs_boundary() const noexcept {
      return needs_boundary_;
    }

    bool ready() const {
      std::lock_guard<std::recursive_mutex> lock(*control_->gate);
      return ready_ && !control_->closed;
    }

    // Only the backend adapter may publish a submitted/flushed producer
    // prefix establishing its declared boundary. Consumers must
    // wait all dependencies on their own domain queue before the first use.
    // Exclusive owners restore this boundary before their exact retirement;
    // publishing it never grants another reader access during the loan.
    // There is no host wait here and no new semaphore/completion type.
    void publish_boundary(std::shared_ptr<const Boundary> boundary) {
      std::lock_guard<std::recursive_mutex> lock(*control_->gate);
      require_open(*control_);
      if (ready_)
        return;
      if (!boundary)
        throw std::invalid_argument(
            diagnostic("Shared resource boundary dependency is missing"));
      for (auto &use : uses_)
        if (!use.entry->ready) {
          use.boundary = boundary;
          use.entry->boundary = boundary;
          mutate(*use.entry, [&] { use.entry->ready = true; });
        }
      ready_ = true;
    }

    std::vector<std::shared_ptr<const Boundary>> dependencies() const {
      std::lock_guard<std::recursive_mutex> lock(*control_->gate);
      require_open(*control_);
      if (!ready_)
        throw std::invalid_argument(
            diagnostic("Shared resource boundary is not published"));
      std::vector<std::shared_ptr<const Boundary>> result;
      for (const auto &use : uses_)
        if (std::find(result.begin(), result.end(), use.boundary) ==
            result.end())
          result.push_back(use.boundary);
      return result;
    }

   private:
    friend class SharedResourceAdmission;
    Loan(std::shared_ptr<Control> control,
         std::vector<BoundUse> uses,
         std::vector<RuntimeResourceHandle> needs_boundary)
        : control_(std::move(control)),
          uses_(std::move(uses)),
          needs_boundary_(std::move(needs_boundary)),
          ready_(needs_boundary_.empty()) {
    }
    std::shared_ptr<Control> control_;
    std::vector<BoundUse> uses_;
    const std::vector<RuntimeResourceHandle> needs_boundary_;
    bool ready_{false};
    bool active_{false};
  };

  explicit SharedResourceAdmission(std::shared_ptr<std::recursive_mutex> gate)
      : control_(std::make_shared<Control>(std::move(gate))) {
    if (!control_->gate)
      throw std::invalid_argument(
          diagnostic("Shared resource admission requires the host gate"));
  }
  ~SharedResourceAdmission() {
    close();
  }
  SharedResourceAdmission(const SharedResourceAdmission &) = delete;
  SharedResourceAdmission &operator=(const SharedResourceAdmission &) =
      delete;

  void register_resource(RuntimeResourceHandle resource) {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    require_open(*control_);
    if (!resource || control_->resources.count(resource))
      throw std::invalid_argument(
          diagnostic("Invalid or duplicate shared resource identity"));
    auto entry = std::make_shared<Entry>();
    entry->resource = resource;
    control_->resources.emplace(resource, std::move(entry));
  }

  // Physical untrack, not deletion of a public facade. Existing resource
  // leases remain valid after the view retires and keep this mapping alive.
  void remove_resource(RuntimeResourceHandle resource) {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    auto entry = find(*control_, resource);
    if (entry->readers || entry->exclusive)
      throw std::invalid_argument(
          diagnostic("Cannot untrack a resource with active loans"));
    for (const auto &edge : entry->recordings)
      if (auto permit = edge.permit.lock())
        ++permit->removed_resources;
    control_->resources.erase(resource);
  }

  std::shared_ptr<Permit> prepare(const std::vector<PinnedUse> &uses) {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    require_open(*control_);
    auto state = std::make_shared<PermitState>();
    auto bound = bind(*control_, uses);
    auto permit =
        std::shared_ptr<Permit>(new Permit(control_, state, std::move(bound)));
    // Complete every potentially throwing allocation before adding any edge.
    for (const auto &use : permit->uses_) {
      auto &edges = use.entry->recordings;
      if (edges.size() == edges.capacity()) {
        // Ordinary recordings may churn without ever creating a resource loan.
        // Prune cold, without lock() temporarily retaining any observed owner.
        edges.erase(std::remove_if(
                        edges.begin(), edges.end(),
                        [](const auto &edge) { return edge.permit.expired(); }),
                    edges.end());
        if (edges.size() == edges.capacity()) {
          if (edges.size() == edges.max_size())
            throw std::length_error(
                diagnostic("Shared resource recording index is exhausted"));
          const auto growth = std::min(edges.max_size() - edges.size(),
                                       std::max<std::size_t>(1, edges.size()));
          edges.reserve(edges.size() + growth);
        }
      }
    }
    for (const auto &use : permit->uses_) {
      if (blocked(*use.entry, use.access))
        ++state->blocked_resources;
      use.entry->recordings.push_back({state, use.access});
    }
    return permit;
  }

  std::shared_ptr<Loan> reserve(const std::vector<PinnedUse> &resources) {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    require_open(*control_);
    if (control_->submitting)
      throw std::invalid_argument(
          diagnostic("Cannot reserve a resource loan inside a submission callback"));
    auto uses = bind(*control_, resources);
    if (uses.empty())
      throw std::invalid_argument(diagnostic("Loan has no resources"));
    std::vector<RuntimeResourceHandle> needs_boundary;
    needs_boundary.reserve(uses.size());
    for (auto &use : uses) {
      if (use.entry->exclusive ||
          (use.access == SharedResourceUse::exclusive && use.entry->readers))
        throw std::invalid_argument(
            diagnostic("Shared resource has a conflicting active loan"));
      if (use.entry->readers && !use.entry->ready)
        throw std::invalid_argument(
            diagnostic("Shared resource boundary is still being prepared"));
      if (!use.entry->readers)
        needs_boundary.push_back(use.entry->resource);
      else {
        use.boundary = use.entry->boundary.lock();
        if (!use.boundary)
          throw std::logic_error(
              diagnostic("Live read loan lost its boundary dependency"));
      }
      if (use.entry->readers == (std::numeric_limits<std::size_t>::max)())
        throw std::overflow_error(
            diagnostic("Too many shared resource readers"));
    }
    auto loan = std::shared_ptr<Loan>(
        new Loan(control_, std::move(uses), std::move(needs_boundary)));
    loan->active_ = true;
    for (const auto &use : loan->uses_)
      mutate(*use.entry, [&] {
        if (use.access == SharedResourceUse::exclusive)
          use.entry->exclusive = true;
        else
          ++use.entry->readers;
      });
    return loan;
  }

  template <class Submit>
  decltype(auto) ordinary(const std::vector<Use> &uses, Submit &&submit) {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    require_open(*control_);
    // Full preflight, before recording any resource transition or submission.
    for (const auto &use : uses)
      if (blocked(*find(*control_, use.resource), use.access))
        throw std::invalid_argument(
            diagnostic("Ordinary resource use conflicts with a resource loan"));
    Submitting scope(*control_);
    return std::forward<Submit>(submit)();
  }

  // The runtime already holds this same gate across its complete API call.
  // Avoid allocating a temporary vector for its common single-resource check.
  void require_use(Use use) const {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    require_open(*control_);
    if (blocked(*find(*control_, use.resource), use.access))
      throw std::invalid_argument(
          diagnostic("Ordinary resource use conflicts with a resource loan"));
  }

  // Cold synchronization/diagnostics must not read mutable state currently
  // owned by a domain. This query does not grant access or replace a permit.
  bool has_loan(RuntimeResourceHandle resource) const {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    require_open(*control_);
    const auto entry = find(*control_, resource);
    return entry->exclusive || entry->readers;
  }

  void close() noexcept {
    std::lock_guard<std::recursive_mutex> lock(*control_->gate);
    control_->closed = true;
  }

 private:
  std::shared_ptr<Control> control_;
};

}  // namespace taichi::lang
