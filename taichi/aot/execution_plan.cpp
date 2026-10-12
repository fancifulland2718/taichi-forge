#include "taichi/aot/execution_plan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include "taichi/common/json.h"

namespace taichi::lang::aot {
namespace {
using liong::json::JsonValue;

void fields(const JsonValue &object,
            std::initializer_list<const char *> allowed) {
  if (!object.is_obj())
    throw std::invalid_argument("Execution plan entries must be objects");
  for (const auto &[name, value] : object.obj.inner) {
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end())
      throw std::invalid_argument("Unknown execution plan field: " + name);
  }
}

std::string name(const JsonValue &value) {
  if (!value.is_str() || value.str.empty() ||
      value.str.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "Execution plan names must be nonempty strings without NUL");
  return value.str;
}

std::uint64_t integer(const JsonValue &value) {
  if (value.ty != liong::json::L_JSON_INT || value.num_int < 0)
    throw std::invalid_argument(
        "Execution plan sizes and offsets must be nonnegative integers");
  return static_cast<std::uint64_t>(value.num_int);
}

std::optional<std::uint64_t> optional_integer(const JsonValue &object,
                                              const char *key) {
  auto found = object.obj.inner.find(key);
  return found == object.obj.inner.end()
             ? std::nullopt
             : std::optional(integer(found->second));
}
}  // namespace

ExecutionPlan ExecutionPlan::parse(const std::string &json) {
  try {
    const auto root = liong::json::parse_complete(json);
    fields(root, {"version", "operations"});
    if (integer(root["version"]) != 1)
      throw std::invalid_argument("Unsupported execution plan version");
    const auto &steps = root["operations"];
    if (!steps.is_arr() || steps.arr.inner.empty() ||
        steps.arr.inner.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "Execution plan needs a nonempty operation array");
    ExecutionPlan plan;
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
    plan.arguments.assign(arguments.begin(), arguments.end());
    return plan;
  } catch (const std::exception &error) {
    throw std::invalid_argument(std::string("Invalid execution plan: ") +
                                error.what());
  }
}
}  // namespace taichi::lang::aot
