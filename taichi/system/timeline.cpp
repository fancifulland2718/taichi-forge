#include "taichi/system/timeline.h"
#include "taichi/util/environ_config.h"

#include <algorithm>

namespace taichi {

Timelines::Timelines() {
  const int configured = lang::get_environ_config(
      "TI_TIMELINE_MAX_EVENTS", static_cast<int>(kDefaultEventCapacity));
  event_capacity_.store(
      std::clamp<std::size_t>(static_cast<std::size_t>(std::max(1, configured)),
                              1, kMaximumEventCapacity),
      std::memory_order_relaxed);
}

bool Timelines::try_reserve_event() {
  const auto capacity = event_capacity_.load(std::memory_order_relaxed);
  auto recorded = recorded_events_.load(std::memory_order_relaxed);
  while (recorded < capacity) {
    if (recorded_events_.compare_exchange_weak(
            recorded, recorded + 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return true;
    }
  }
  dropped_events_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

std::size_t Timelines::event_capacity() const {
  return event_capacity_.load(std::memory_order_relaxed);
}

std::uint64_t Timelines::recorded_event_count() const {
  return recorded_events_.load(std::memory_order_relaxed);
}

std::string TimelineEvent::to_json() {
  std::string json{"{"};
  json += fmt::format("\"cat\":\"taichi\",");
  json += fmt::format("\"pid\":0,");
  json += fmt::format("\"tid\":\"{}\",", tid);
  json += fmt::format("\"ph\":\"{}\",", begin ? "B" : "E");
  json += fmt::format("\"name\":\"{}\",", name);
  json += fmt::format("\"ts\":\"{}\"", uint64(time * 1000000));
  json += "}";
  return json;
}

Timeline::Timeline() : tid_("unnamed") {
  Timelines::get_instance().insert_timeline(this);
}

Timeline &Timeline::get_this_thread_instance() {
  thread_local Timeline instance;
  return instance;
}

Timeline::~Timeline() {
  Timelines::get_instance().insert_events(fetch_events());
  Timelines::get_instance().remove_timeline(this);
}

void Timeline::clear() {
  std::lock_guard<std::mutex> _(mut_);
  events_.clear();
}
void Timeline::insert_event(const TimelineEvent &e) {
  auto &timelines = Timelines::get_instance();
  if (!timelines.get_enabled() || !timelines.try_reserve_event()) {
    return;
  }
  std::lock_guard<std::mutex> _(mut_);
  events_.push_back(e);
}

std::vector<TimelineEvent> Timeline::fetch_events() {
  std::lock_guard<std::mutex> _(mut_);
  std::vector<TimelineEvent> fetched;
  std::swap(fetched, events_);
  return fetched;
}

Timeline::Guard::Guard(const std::string &name) : name_(name) {
  auto &timeline = Timeline::get_this_thread_instance();
  timeline.insert_event({name, true, Time::get_time(), timeline.tid_});
}

Timeline::Guard::~Guard() {
  auto &timeline = Timeline::get_this_thread_instance();
  timeline.insert_event({name_, false, Time::get_time(), timeline.tid_});
}

void Timelines::insert_events(const std::vector<TimelineEvent> &events) {
  std::lock_guard<std::mutex> _(mut_);
  insert_events_without_locking(events);
}

void Timelines::insert_events_without_locking(
    const std::vector<TimelineEvent> &events) {
  events_.insert(events_.end(), events.begin(), events.end());
}

Timelines &taichi::Timelines::get_instance() {
  static auto instance = new Timelines();
  return *instance;
}

void Timelines::clear() {
  std::lock_guard<std::mutex> _(mut_);
  events_.clear();
  for (auto timeline : timelines_) {
    timeline->clear();
  }
  recorded_events_.store(0, std::memory_order_relaxed);
  dropped_events_.store(0, std::memory_order_relaxed);
}

void Timelines::save(const std::string &filename) {
  std::lock_guard<std::mutex> _(mut_);
  std::sort(timelines_.begin(), timelines_.end(), [](Timeline *a, Timeline *b) {
    return a->get_name() < b->get_name();
  });
  for (auto timeline : timelines_) {
    insert_events_without_locking(timeline->fetch_events());
  }
  if (!ends_with(filename, ".json")) {
    TI_WARN("Timeline filename {} should end with '.json'.", filename);
  }
  const auto dropped = dropped_events_.load(std::memory_order_relaxed);
  if (dropped != 0) {
    TI_WARN("Timeline reached its {}-event memory budget; {} later events "
            "were dropped.",
            event_capacity(), dropped);
  }
  std::ofstream fout(filename);
  fout << "[";
  bool first = true;
  for (auto &e : events_) {
    if (first) {
      first = false;
    } else {
      fout << ",";
    }
    fout << e.to_json() << std::endl;
  }
  fout << "]";
}

void Timelines::insert_timeline(Timeline *timeline) {
  std::lock_guard<std::mutex> _(mut_);
  timelines_.push_back(timeline);
}

void Timelines::remove_timeline(Timeline *timeline) {
  std::lock_guard<std::mutex> _(mut_);
  trash(std::remove(timelines_.begin(), timelines_.end(), timeline));
}

bool Timelines::get_enabled() const {
  return enabled_.load(std::memory_order_relaxed);
}

void Timelines::set_enabled(bool enabled) {
  enabled_.store(enabled, std::memory_order_relaxed);
}

std::uint64_t Timelines::dropped_event_count() const {
  return dropped_events_.load(std::memory_order_relaxed);
}

void Timelines::set_event_capacity_for_testing(std::size_t capacity) {
  TI_ASSERT(capacity >= 1 && capacity <= kMaximumEventCapacity);
  clear();
  event_capacity_.store(capacity, std::memory_order_relaxed);
}

}  // namespace taichi
