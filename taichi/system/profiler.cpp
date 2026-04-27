#include "taichi/system/profiler.h"
#include "spdlog/fmt/bundled/color.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace taichi {

// A profiler's records form a tree structure
struct ProfilerRecordNode {
  std::vector<std::unique_ptr<ProfilerRecordNode>> childs;
  ProfilerRecordNode *parent;
  std::string name;
  float64 total_time;
  // Time per element
  bool account_tpe;
  uint64 total_elements;
  int64 num_samples;

  ProfilerRecordNode(const std::string &name, ProfilerRecordNode *parent) {
    this->name = name;
    this->parent = parent;
    this->total_time = 0.0_f64;
    this->num_samples = 0ll;
    this->total_elements = 0ll;
    this->account_tpe = false;
  }

  void insert_sample(float64 sample) {
    num_samples += 1;
    total_time += sample;
  }

  void insert_sample(float64 sample, uint64 elements) {
    account_tpe = true;
    num_samples += 1;
    total_time += sample;
    total_elements += elements;
  }

  float64 get_averaged() const {
    return total_time / (float64)std::max(num_samples, int64(1));
  }

  float64 get_averaged_tpe() const {
    TI_ASSERT(account_tpe);
    return total_time / (float64)total_elements;
  }

  ProfilerRecordNode *get_child(const std::string &name) {
    for (auto &ch : childs) {
      if (ch->name == name) {
        return ch.get();
      }
    }
    childs.push_back(std::make_unique<ProfilerRecordNode>(name, this));
    return childs.back().get();
  }
};

class ProfilerRecords {
 public:
  std::unique_ptr<ProfilerRecordNode> root;
  ProfilerRecordNode *current_node;
  int current_depth;
  bool enabled;

  explicit ProfilerRecords(const std::string &name) {
    root = std::make_unique<ProfilerRecordNode>(
        fmt::format("[Profiler {}]", name), nullptr);
    current_node = root.get();
    current_depth = 0;  // depth(root) = 0
    enabled = true;
  }

  void clear() {
    root->childs.clear();
    current_node = root.get();
    current_depth = 0;
    enabled = true;
  }

  void print(ProfilerRecordNode *node, int depth);

  void print() {
    fmt::print(fg(fmt::color::cyan), std::string(80, '>') + "\n");
    print(root.get(), 0);
    fmt::print(fg(fmt::color::cyan), std::string(80, '>') + "\n");
  }

  void insert_sample(float64 time) {
    if (!enabled)
      return;
    current_node->insert_sample(time);
  }

  void insert_sample(float64 time, uint64 tpe) {
    if (!enabled)
      return;
    current_node->insert_sample(time, tpe);
  }

  void push(const std::string name) {
    if (!enabled)
      return;
    current_node = current_node->get_child(name);
    current_depth += 1;
  }

  void pop() {
    if (!enabled)
      return;
    current_node = current_node->parent;
    current_depth -= 1;
  }

  static ProfilerRecords &get_this_thread_instance() {
    // Use a raw pointer so that it lives together with the process
    static thread_local ProfilerRecords *profiler_records = nullptr;
    if (profiler_records == nullptr) {
      profiler_records = Profiling::get_instance().get_this_thread_profiler();
    }
    return *profiler_records;
  }
};

void ProfilerRecords::print(ProfilerRecordNode *node, int depth) {
  auto make_indent = [depth](int additional) {
    for (int i = 0; i < depth + additional; i++) {
      fmt::print("    ");
    }
  };
  using TimeScale = std::pair<real, std::string>;

  auto get_time_scale = [&](real t) -> TimeScale {
    if (t < 1e-6) {
      return std::make_pair(1e9_f, "ns");
    } else if (t < 1e-3) {
      return std::make_pair(1e6_f, "us");
    } else if (t < 1) {
      return std::make_pair(1e3_f, "ms");
    } else if (t < 60) {
      return std::make_pair(1_f, " s");
    } else if (t < 3600) {
      return std::make_pair(1.0_f / 60_f, " m");
    } else {
      return std::make_pair(1.0_f / 3600_f, "h");
    }
  };

  auto get_readable_time_with_scale = [&](real t, TimeScale scale) {
    return fmt::format("{:7.3f} {}", t * scale.first, scale.second);
  };

  auto get_readable_time = [&](real t) {
    auto scale = get_time_scale(t);
    return get_readable_time_with_scale(t, scale);
  };

  float64 total_time = node->total_time;
  fmt::color level_color;
  if (depth == 0)
    level_color = fmt::color::red;
  else if (depth == 1)
    level_color = fmt::color::light_green;
  else if (depth == 2)
    level_color = fmt::color::yellow;
  else if (depth == 3)
    level_color = fmt::color::light_blue;
  else if (depth >= 4)
    level_color = fmt::color::magenta;
  if (depth == 0) {
    // Root node only
    make_indent(0);
    fmt::print(fg(level_color), "{}\n", node->name.c_str());
  }
  if (total_time < 1e-6f) {
    for (auto &ch : node->childs) {
      make_indent(1);
      auto child_time = ch->total_time;
      auto bulk_statistics =
          fmt::format("{} {}", get_readable_time(child_time), ch->name);
      fmt::print(fg(level_color), "{:40}", bulk_statistics);
      fmt::print(fg(fmt::color::cyan), " [{} x {}]\n", ch->num_samples,
                 get_readable_time_with_scale(
                     ch->get_averaged(), get_time_scale(ch->get_averaged())));
      print(ch.get(), depth + 1);
    }
  } else {
    TimeScale scale = get_time_scale(total_time);
    float64 unaccounted = total_time;
    for (auto &ch : node->childs) {
      make_indent(1);
      auto child_time = ch->total_time;
      std::string bulk_statistics = fmt::format(
          "{} {:5.2f}%  {}", get_readable_time_with_scale(child_time, scale),
          child_time * 100.0 / total_time, ch->name);
      fmt::print(fg(level_color), "{:40}", bulk_statistics);
      fmt::print(fg(fmt::color::cyan), " [{} x {}]\n", ch->num_samples,
                 get_readable_time_with_scale(
                     ch->get_averaged(), get_time_scale(ch->get_averaged())));
      if (ch->account_tpe) {
        make_indent(1);
        fmt::print("                     [TPE] {}\n",
                   get_readable_time(ch->total_time));
      }
      print(ch.get(), depth + 1);
      unaccounted -= child_time;
    }
    if (!node->childs.empty() && (unaccounted > total_time * 0.005)) {
      make_indent(1);
      fmt::print(fg(level_color), "{} {:5.2f}%  {}\n",
                 get_readable_time_with_scale(unaccounted, scale),
                 unaccounted * 100.0 / total_time, "[unaccounted]");
    }
  }
}

ScopedProfiler::ScopedProfiler(std::string name, uint64 elements) {
  start_time_ = Time::get_time();
  this->name_ = name;
  this->elements_ = elements;
  stopped_ = false;
  ProfilerRecords::get_this_thread_instance().push(name);
}

void ScopedProfiler::stop() {
  TI_ASSERT_INFO(!stopped_, "Profiler already stopped.");
  float64 elapsed = Time::get_time() - start_time_;
  if ((int64)elements_ != -1) {
    ProfilerRecords::get_this_thread_instance().insert_sample(elapsed,
                                                              elements_);
  } else {
    ProfilerRecords::get_this_thread_instance().insert_sample(elapsed);
  }
  ProfilerRecords::get_this_thread_instance().pop();

  // Phase 0': also record as a flat trace event when tracing is enabled.
  // Cheap predicate (single atomic load) when disabled.
  if (Profiling::is_tracing_enabled()) {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    uint64 tid = std::hash<std::string>{}(ss.str());
    Profiling::get_instance().record_trace_event(
        TraceEvent{name_, tid, start_time_ * 1e6, elapsed * 1e6});
  }
}

void ScopedProfiler::disable() {
  ProfilerRecords::get_this_thread_instance().enabled = false;
}

void ScopedProfiler::enable() {
  ProfilerRecords::get_this_thread_instance().enabled = true;
}

ScopedProfiler::~ScopedProfiler() {
  if (!stopped_) {
    stop();
  }
}

Profiling &Profiling::get_instance() {
  static auto prof = new Profiling;
  return *prof;
}

ProfilerRecords *Profiling::get_this_thread_profiler() {
  std::lock_guard<std::mutex> _(mut_);
  auto id = std::this_thread::get_id();
  std::stringstream ss;
  ss << id;
  if (profilers_.find(id) == profilers_.end()) {
    // Note: thread id may be reused
    profilers_[id] = new ProfilerRecords(fmt::format("thread {}", ss.str()));
  }
  return profilers_[id];
}

void Profiling::print_profile_info() {
  std::lock_guard<std::mutex> _(mut_);
  for (auto p : profilers_) {
    p.second->print();
  }
}

void Profiling::clear_profile_info() {
  std::lock_guard<std::mutex> _(mut_);
  for (auto p : profilers_) {
    p.second->clear();
  }
  std::lock_guard<std::mutex> _t(trace_mut_);
  trace_events_.clear();
}

// ---------------------------------------------------------------------------
// Phase 0': CSV + Chrome trace export
// ---------------------------------------------------------------------------

// P-Compile-7: TU-static runtime override of the env-driven tracing gate.
// -1 = use env cache (default), 0 = forced off, 1 = forced on.
static std::atomic<int> g_tracing_runtime_override{-1};

void Profiling::set_tracing_runtime_override(bool enabled) {
  g_tracing_runtime_override.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void Profiling::clear_tracing_runtime_override() {
  g_tracing_runtime_override.store(-1, std::memory_order_relaxed);
}

bool Profiling::is_tracing_enabled() {
  // Runtime override (set by `ti.compile_profile()` ctx mgr) takes
  // precedence over the env-cached value.
  int ov = g_tracing_runtime_override.load(std::memory_order_relaxed);
  if (ov >= 0) {
    return ov != 0;
  }

  // Evaluated once per process. Flip by setting TI_COMPILE_PROFILE to a
  // non-empty string before importing Taichi.
  static std::atomic<int> cached{-1};
  int v = cached.load(std::memory_order_relaxed);
  if (v >= 0) {
    return v != 0;
  }
  const char *env = std::getenv("TI_COMPILE_PROFILE");
  int enabled = (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0)
                    ? 1
                    : 0;
  cached.store(enabled, std::memory_order_relaxed);
  if (enabled) {
    // Best-effort auto-flush on process exit. Users can still call the
    // export_* methods explicitly.
    static std::atomic<bool> registered{false};
    bool expected = false;
    if (registered.compare_exchange_strong(expected, true)) {
      std::atexit([]() {
        const char *out = std::getenv("TI_COMPILE_PROFILE");
        if (!out || out[0] == '\0')
          return;
        std::string base = out;
        // If user set =1/=on we derive a default file name.
        if (base == "1" || base == "on" || base == "ON" || base == "true") {
          base = "taichi_compile_profile";
        }
        Profiling::get_instance().export_csv(base + ".csv");
        Profiling::get_instance().export_chrome_trace(base + ".json");
      });
    }
  }
  return enabled != 0;
}

void Profiling::record_trace_event(TraceEvent &&ev) {
  std::lock_guard<std::mutex> _(trace_mut_);
  trace_events_.push_back(std::move(ev));
}

namespace {

// Recursive CSV walker. Path is '/'-joined ancestor names.
void write_csv_node(std::ostream &os,
                    const std::string &thread_label,
                    const std::string &parent_path,
                    const ProfilerRecordNode *node) {
  std::string path = parent_path.empty()
                         ? node->name
                         : parent_path + "/" + node->name;
  // Skip the synthetic root (it has no samples) but still recurse.
  if (node->num_samples > 0) {
    double total = node->total_time;
    double avg = node->get_averaged();
    double tpe = node->account_tpe ? node->get_averaged_tpe() : 0.0;
    // CSV: thread,path,calls,total_s,avg_s,tpe_s
    os << '"' << thread_label << "\",\"" << path << "\","
       << node->num_samples << "," << total << "," << avg << "," << tpe
       << "\n";
  }
  for (const auto &ch : node->childs) {
    write_csv_node(os, thread_label, path, ch.get());
  }
}

// Basic JSON string escape (Chrome trace event names).
std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

}  // namespace

bool Profiling::export_csv(const std::string &path) {
  std::ofstream os(path);
  if (!os) {
    return false;
  }
  os << "thread,path,calls,total_s,avg_s,tpe_s\n";
  std::lock_guard<std::mutex> _(mut_);
  for (auto &p : profilers_) {
    std::stringstream tid_ss;
    tid_ss << p.first;
    write_csv_node(os, tid_ss.str(), "", p.second->root.get());
  }
  return static_cast<bool>(os);
}

bool Profiling::export_chrome_trace(const std::string &path) {
  std::ofstream os(path);
  if (!os) {
    return false;
  }
  // Chrome Trace "JSON Array Format": bare array of events.
  os << "[\n";
  std::lock_guard<std::mutex> _(trace_mut_);
  bool first = true;
  for (const auto &ev : trace_events_) {
    if (!first) {
      os << ",\n";
    }
    first = false;
    // 'X' complete event: requires ts + dur.
    os << "{\"name\":\"" << json_escape(ev.name)
       << "\",\"ph\":\"X\",\"pid\":1"
       << ",\"tid\":" << ev.tid
       << ",\"ts\":" << ev.ts_us
       << ",\"dur\":" << ev.dur_us
       << ",\"cat\":\"taichi.compile\"}";
  }
  os << "\n]\n";
  return static_cast<bool>(os);
}

}  // namespace taichi
