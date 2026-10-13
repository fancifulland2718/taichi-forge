#include "taichi/aot/execution_plan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include "taichi/aot/execution_plan_json.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang::aot {
using namespace execution_json;

ExecutionPlan ExecutionPlan::parse(const std::string &json) {
  try {
    const auto root = liong::json::parse_complete(json);
    fields(root, {"version", "operations", "resources", "gradients",
                  "initialization"});
    const auto version = integer(root["version"]);
    if (version != 1 && version != 2)
      throw std::invalid_argument("Unsupported execution plan version");
    const bool has_initialization = root.obj.inner.count("initialization");
    if ((version == 2) != has_initialization)
      throw std::invalid_argument(
          "Initialization requires execution plan version 2");
    ExecutionPlan plan;
    plan.resources = ExecutionResources::parse(root);
    std::set<std::string> arguments;
    auto parse_steps = [&](const JsonValue &steps,
                           std::vector<Operation> &operations) {
      if (!steps.is_arr() || steps.arr.inner.empty() ||
          steps.arr.inner.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(
            "Execution plan needs a nonempty operation array");
      for (const auto &step : steps.arr.inner) {
        const auto kind = name(step["kind"]);
        if (kind == "kernel") {
          fields(step, {"kind", "entry", "arguments", "label"});
          Kernel kernel{name(step["entry"]), {}, {}};
          if (auto found = step.obj.inner.find("label");
              found != step.obj.inner.end()) {
            kernel.label = name(found->second);
            if (!is_valid_dispatch_label(kernel.label))
              throw std::invalid_argument(
                  "Dispatch labels cannot contain NUL or line breaks");
          }
          if (!step["arguments"].is_arr() ||
              step["arguments"].arr.inner.size() >
                  std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument(
                "Kernel arguments must be an array of names");
          for (const auto &value : step["arguments"].arr.inner) {
            kernel.arguments.push_back(name(value));
            arguments.insert(kernel.arguments.back());
          }
          operations.emplace_back(std::move(kernel));
        } else if (kind == "copy") {
          fields(step, {"kind", "destination", "source", "destination_offset",
                        "source_offset", "size_bytes"});
          Copy copy{name(step["destination"]), name(step["source"]),
                    optional_integer(step, "destination_offset").value_or(0),
                    optional_integer(step, "source_offset").value_or(0),
                    optional_integer(step, "size_bytes")};
          arguments.insert(copy.destination);
          arguments.insert(copy.source);
          operations.emplace_back(std::move(copy));
        } else if (kind == "native") {
          auto native = NativeOperationDescription::parse(step);
          for (const auto &[port, slot] : native.bindings)
            arguments.insert(slot);
          operations.emplace_back(std::move(native));
        } else if (kind == "sort") {
          fields(step, {"kind", "keys", "values"});
          Sort sort{name(step["keys"]), std::nullopt};
          arguments.insert(sort.keys);
          if (auto found = step.obj.inner.find("values");
              found != step.obj.inner.end()) {
            sort.values = name(found->second);
            arguments.insert(*sort.values);
          }
          operations.emplace_back(std::move(sort));
        } else if (kind == "compact") {
          fields(step,
                 {"kind", "values", "flags", "output", "count", "columns"});
          Compact compact{{}, name(step["flags"]), name(step["count"])};
          auto add_column = [&](const JsonValue &column) {
            compact.columns.push_back(
                {name(column["values"]), name(column["output"])});
            arguments.insert(compact.columns.back().values);
            arguments.insert(compact.columns.back().output);
          };
          if (auto found = step.obj.inner.find("columns");
              found != step.obj.inner.end()) {
            if (step.obj.inner.count("values") ||
                step.obj.inner.count("output") || !found->second.is_arr() ||
                found->second.arr.inner.empty())
              throw std::invalid_argument(
                  "Compact columns must be nonempty and cannot use scalar "
                  "shorthand");
            for (const auto &column : found->second.arr.inner) {
              fields(column, {"values", "output"});
              add_column(column);
            }
          } else {
            add_column(step);
          }
          arguments.insert(compact.flags);
          arguments.insert(compact.count);
          operations.emplace_back(std::move(compact));
        } else if (kind == "fill32") {
          fields(step,
                 {"kind", "destination", "value", "offset", "size_bytes"});
          auto value = integer(step["value"]);
          if (value > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument(
                "fill32 value exceeds a 32-bit pattern");
          Fill fill{name(step["destination"]),
                    static_cast<std::uint32_t>(value),
                    optional_integer(step, "offset").value_or(0),
                    optional_integer(step, "size_bytes")};
          arguments.insert(fill.destination);
          operations.emplace_back(std::move(fill));
        } else {
          auto native = native_operation_registry().lower_legacy(step);
          if (!native)
            throw std::invalid_argument(
                "Unsupported execution plan operation: " + kind);
          for (const auto &[port, slot] : native->bindings)
            arguments.insert(slot);
          operations.emplace_back(std::move(*native));
        }
      }
    };
    parse_steps(root["operations"], plan.operations);
    plan.replay_bindings = plan.resources.runtime_bindings(arguments);
    if (has_initialization)
      parse_steps(root["initialization"], plan.initialization);
    plan.resources.analyze(arguments);
    plan.arguments.assign(arguments.begin(), arguments.end());
    return plan;
  } catch (const std::exception &error) {
    throw std::invalid_argument(std::string("Invalid execution plan: ") +
                                error.what());
  }
}
}  // namespace taichi::lang::aot
