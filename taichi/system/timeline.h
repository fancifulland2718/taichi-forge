#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>

#include "taichi/common/core.h"
#include "taichi/system/timer.h"

namespace taichi {

struct TimelineEvent {
  std::string name;
  bool begin;
  float64 time;
  std::string tid;

  std::string to_json();
};

class Timeline {
 public:
  Timeline();

  ~Timeline();

  static Timeline &get_this_thread_instance();

  void set_name(const std::string &tid) {
    tid_ = tid;
  }

  std::string get_name() {
    return tid_;
  }

  void clear();

  void insert_event(const TimelineEvent &e);

  std::vector<TimelineEvent> fetch_events();

  class Guard {
   public:
    explicit Guard(const std::string &name);

    ~Guard();

   private:
    std::string name_;
  };

 private:
  std::string tid_;
  std::mutex mut_;
  std::vector<TimelineEvent> events_;
};

// A timeline system for multi-threaded applications
class Timelines {
 public:
  static constexpr std::size_t kDefaultEventCapacity = 65536;
  static constexpr std::size_t kMaximumEventCapacity = 1048576;

  Timelines();

  static Timelines &get_instance();

  bool try_reserve_event();

  void insert_events(const std::vector<TimelineEvent> &events);

  void insert_events_without_locking(const std::vector<TimelineEvent> &events);

  void insert_timeline(Timeline *timeline);

  void remove_timeline(Timeline *timeline);

  void clear();

  void save(const std::string &filename);

  bool get_enabled() const;

  void set_enabled(bool enabled);

  std::size_t event_capacity() const;

  std::uint64_t recorded_event_count() const;

  std::uint64_t dropped_event_count() const;

  void set_event_capacity_for_testing(std::size_t capacity);

 private:
  std::mutex mut_;
  std::vector<TimelineEvent> events_;
  std::vector<Timeline *> timelines_;
  std::atomic<std::size_t> event_capacity_{kDefaultEventCapacity};
  std::atomic<std::uint64_t> recorded_events_{0};
  std::atomic<std::uint64_t> dropped_events_{0};
  std::atomic<bool> enabled_{false};
};

#define TI_TIMELINE(name) \
  taichi::Timeline::Guard _timeline_guard_##__LINE__(name);

#define TI_AUTO_TIMELINE TI_TIMELINE(__FUNCTION__)

}  // namespace taichi
