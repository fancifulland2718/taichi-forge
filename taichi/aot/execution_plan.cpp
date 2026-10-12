#include "taichi/aot/execution_plan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include "taichi/aot/execution_plan_json.h"

namespace taichi::lang::aot {
using namespace execution_json;

ExecutionPlan ExecutionPlan::parse(const std::string &json) {
  try {
    const auto root = liong::json::parse_complete(json);
    fields(root, {"version", "operations", "resources", "gradients"});
    if (integer(root["version"]) != 1)
      throw std::invalid_argument("Unsupported execution plan version");
    const auto &steps = root["operations"];
    if (!steps.is_arr() || steps.arr.inner.empty() ||
        steps.arr.inner.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "Execution plan needs a nonempty operation array");
    ExecutionPlan plan;
    plan.resources = ExecutionResources::parse(root);
    std::set<std::string> arguments;
    for (const auto &step : steps.arr.inner) {
      const auto kind = name(step["kind"]);
      if (kind == "kernel") {
        fields(step, {"kind", "entry", "arguments"});
        Kernel kernel{name(step["entry"]), {}};
        if (!step["arguments"].is_arr() ||
            step["arguments"].arr.inner.size() >
                std::numeric_limits<std::uint32_t>::max())
          throw std::invalid_argument(
              "Kernel arguments must be an array of names");
        for (const auto &value : step["arguments"].arr.inner) {
          kernel.arguments.push_back(name(value));
          arguments.insert(kernel.arguments.back());
        }
        plan.operations.emplace_back(std::move(kernel));
      } else if (kind == "copy") {
        fields(step, {"kind", "destination", "source", "destination_offset",
                      "source_offset", "size_bytes"});
        Copy copy{name(step["destination"]), name(step["source"]),
                  optional_integer(step, "destination_offset").value_or(0),
                  optional_integer(step, "source_offset").value_or(0),
                  optional_integer(step, "size_bytes")};
        arguments.insert(copy.destination);
        arguments.insert(copy.source);
        plan.operations.emplace_back(std::move(copy));
      } else if (kind == "fft_c2c_f32") {
        fields(step, {"kind", "input", "output", "dimensions", "batch_count",
                      "direction"});
        Fft fft{name(step["input"]), name(step["output"]), {}};
        const auto &dimensions = step["dimensions"];
        if (!dimensions.is_arr() || dimensions.arr.inner.empty())
          throw std::invalid_argument(
              "FFT dimensions must be a nonempty array");
        for (const auto &axis : dimensions.arr.inner) {
          auto extent = integer(axis);
          if (!extent)
            throw std::invalid_argument("FFT dimensions must be positive");
          fft.dimensions.push_back(extent);
        }
        fft.batch_count = optional_integer(step, "batch_count").value_or(1);
        if (!fft.batch_count)
          throw std::invalid_argument("FFT batch count must be positive");
        const auto direction = name(step["direction"]);
        if (direction != "forward" && direction != "inverse")
          throw std::invalid_argument(
              "FFT direction must be forward or inverse");
        fft.inverse = direction == "inverse";
        arguments.insert(fft.input);
        arguments.insert(fft.output);
        plan.operations.emplace_back(std::move(fft));
      } else if (kind == "sort") {
        fields(step, {"kind", "keys", "values"});
        Sort sort{name(step["keys"]), std::nullopt};
        arguments.insert(sort.keys);
        if (auto found = step.obj.inner.find("values");
            found != step.obj.inner.end()) {
          sort.values = name(found->second);
          arguments.insert(*sort.values);
        }
        plan.operations.emplace_back(std::move(sort));
      } else if (kind == "compact") {
        fields(step, {"kind", "values", "flags", "output", "count", "columns"});
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
        plan.operations.emplace_back(std::move(compact));
      } else if (kind == "fill32") {
        fields(step, {"kind", "destination", "value", "offset", "size_bytes"});
        auto value = integer(step["value"]);
        if (value > std::numeric_limits<std::uint32_t>::max())
          throw std::invalid_argument("fill32 value exceeds a 32-bit pattern");
        Fill fill{name(step["destination"]), static_cast<std::uint32_t>(value),
                  optional_integer(step, "offset").value_or(0),
                  optional_integer(step, "size_bytes")};
        arguments.insert(fill.destination);
        plan.operations.emplace_back(std::move(fill));
      } else {
        throw std::invalid_argument("Unsupported execution plan operation: " +
                                    kind);
      }
    }
    plan.resources.analyze(arguments);
    plan.arguments.assign(arguments.begin(), arguments.end());
    return plan;
  } catch (const std::exception &error) {
    throw std::invalid_argument(std::string("Invalid execution plan: ") +
                                error.what());
  }
}
}  // namespace taichi::lang::aot
