#include "taichi/system/profiler_annotation.h"

#include <memory>
#include <mutex>

#include "taichi/common/dynamic_loader.h"
#include "taichi/common/logging.h"

namespace taichi::lang {
namespace {

thread_local std::string dispatch_label;
thread_local const std::string *profiler_name_override{nullptr};

using NvtxRangePush = int (*)(const char *);
using NvtxRangePop = int (*)();

struct NvtxFunctions {
  std::unique_ptr<DynamicLoader> loader;
  NvtxRangePush range_push{nullptr};
  NvtxRangePop range_pop{nullptr};
};

NvtxFunctions &nvtx_functions() {
  static NvtxFunctions functions;
  static std::once_flag once;
  std::call_once(once, [&] {
#if defined(_WIN32)
    const char *candidates[] = {"nvToolsExt64_1.dll"};
#elif defined(__APPLE__)
    const char *candidates[] = {"libnvToolsExt.dylib"};
#else
    const char *candidates[] = {"libnvToolsExt.so.1", "libnvToolsExt.so"};
#endif
    for (const char *candidate : candidates) {
      auto loader = std::make_unique<DynamicLoader>(candidate);
      if (!loader->loaded()) {
        continue;
      }
      auto push = reinterpret_cast<NvtxRangePush>(
          loader->load_function_optional("nvtxRangePushA"));
      auto pop = reinterpret_cast<NvtxRangePop>(
          loader->load_function_optional("nvtxRangePop"));
      if (push != nullptr && pop != nullptr) {
        functions.loader = std::move(loader);
        functions.range_push = push;
        functions.range_pop = pop;
        break;
      }
    }
  });
  return functions;
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
  auto &nvtx = nvtx_functions();
  if (nvtx.range_push != nullptr) {
    nvtx.range_push(name.c_str());
    active_ = true;
  }
}

ScopedExternalProfilerAnnotation::~ScopedExternalProfilerAnnotation() {
  if (active_) {
    nvtx_functions().range_pop();
  }
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
