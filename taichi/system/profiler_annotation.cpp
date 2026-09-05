#include "taichi/system/profiler_annotation.h"

#include "taichi/common/logging.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <nvtx3/nvToolsExt.h>

namespace taichi::lang {
namespace {

thread_local std::string dispatch_label;
thread_local const std::string *profiler_name_override{nullptr};

nvtxDomainHandle_t profiler_domain() {
  // NVTX owns domain lifetime until process exit. Initialize only on an
  // explicitly annotated operation, never on import or ordinary replay.
  static nvtxDomainHandle_t domain = [] {
    auto value = nvtxDomainCreateA("taichi_forge");
    const char *categories[] = {"task", "search", "stage", "recipe",
                                "materialization", "trial", "user"};
    for (uint32_t index = 0; index < 7; ++index) {
      nvtxDomainNameCategoryA(value, index + 1, categories[index]);
    }
    return value;
  }();
  return domain;
}

std::string escape_trace_component(const std::string &value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value) {
    if (c == '\\' || c == '|') {
      result.push_back('\\');
    }
    result.push_back(c);
  }
  return result;
}

}  // namespace

void validate_dispatch_label(const std::string &label) {
  constexpr std::size_t kMaximumDispatchLabelBytes = 128;
  TI_ERROR_IF(label.size() > kMaximumDispatchLabelBytes,
              "Dispatch labels may contain at most {} UTF-8 bytes, got {}",
              kMaximumDispatchLabelBytes, label.size());
  for (unsigned char c : label) {
    TI_ERROR_IF(c == 0 || c == '\n' || c == '\r',
                "Dispatch labels cannot contain NUL or line breaks");
  }
}

std::string push_dispatch_label(std::string label) {
  validate_dispatch_label(label);
  std::string previous = std::move(dispatch_label);
  dispatch_label = std::move(label);
  return previous;
}

void restore_dispatch_label(std::string label) {
  validate_dispatch_label(label);
  dispatch_label = std::move(label);
}

const std::string *current_dispatch_label() noexcept {
  return dispatch_label.empty() ? nullptr : &dispatch_label;
}

std::string make_labeled_task_name(const std::string &task_name,
                                   const std::string &task_id,
                                   const std::string &label) {
  TI_ASSERT(!label.empty());
  return task_name + " | tf.task=" +
         escape_trace_component(task_id.empty() ? "unavailable" : task_id) +
         " label=" + escape_trace_component(label);
}

ScopedExternalProfilerAnnotation::ScopedExternalProfilerAnnotation(
    const std::string &name) {
  push_external_profiler_range(name, 1, 0);
}

ScopedExternalProfilerAnnotation::~ScopedExternalProfilerAnnotation() {
  pop_external_profiler_range();
}

void push_external_profiler_range(const std::string &name,
                                  uint32_t category,
                                  uint64_t payload) {
  nvtxEventAttributes_t event{};
  event.version = NVTX_VERSION;
  event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event.category = category;
  event.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event.message.ascii = name.c_str();
  event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  event.payload.ullValue = payload;
  nvtxDomainRangePushEx(profiler_domain(), &event);
}

void pop_external_profiler_range() {
  nvtxDomainRangePop(profiler_domain());
}

ScopedKernelProfilerName::ScopedKernelProfilerName(
    const std::string &name) noexcept
    : previous_(profiler_name_override) {
  profiler_name_override = &name;
}

ScopedKernelProfilerName::~ScopedKernelProfilerName() {
  profiler_name_override = previous_;
}

const std::string *ScopedKernelProfilerName::current() noexcept {
  return profiler_name_override;
}

}  // namespace taichi::lang
