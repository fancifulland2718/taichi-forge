#include "taichi/aot/execution_plan.h"

#include <algorithm>
#include <functional>
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
                  "initialization", "constraints"});
    const auto version = integer(root["version"]);
    if (version < 1 || version > 8)
      throw std::invalid_argument("Unsupported execution plan version");
    const bool has_initialization = root.obj.inner.count("initialization");
    if (version < 3 && ((version == 2) != has_initialization))
      throw std::invalid_argument(
          "Initialization requires execution plan version 2");
    ExecutionPlan plan;
    plan.resources = ExecutionResources::parse(root);
    std::set<std::string> arguments;
    std::function<void(const JsonValue &, std::vector<Operation> &, bool)>
        parse_steps;
    parse_steps = [&](const JsonValue &steps,
                      std::vector<Operation> &operations, bool allow_empty) {
      if (!steps.is_arr() || (!allow_empty && steps.arr.inner.empty()) ||
          steps.arr.inner.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(
            "Execution plan needs a nonempty operation array");
      for (const auto &step : steps.arr.inner) {
        const auto kind = name(step["kind"]);
        if (kind == "while") {
          if (version < 3)
            throw std::invalid_argument(
                "Structured control requires execution plan version 3");
          fields(step, {"kind", "predicate", "max_iterations", "condition",
                        "body", "continue_while_nonzero", "lowering"});
          auto loop = std::make_shared<While>();
          loop->predicate = name(step["predicate"]);
          loop->max_iterations = integer(step["max_iterations"]);
          loop->continue_while_nonzero =
              !step.obj.inner.count("continue_while_nonzero") ||
              optional_bool(step, "continue_while_nonzero");
          arguments.insert(loop->predicate);
          parse_steps(step["condition"], loop->condition, true);
          parse_steps(step["body"], loop->body, false);
          if (const auto found = step.obj.inner.find("lowering");
              found != step.obj.inner.end()) {
            if (version < 4)
              throw std::invalid_argument(
                  "Selected control requires execution plan version 4");
            const auto &lowering = found->second;
            const auto kind = name(lowering["kind"]);
            const bool nested = kind == "vulkan_nested";
            fields(lowering,
                   nested
                       ? std::initializer_list<const char *>{"kind", "version",
                                                             "counter",
                                                             "status", "inners"}
                       : std::initializer_list<const char *>{
                             "kind", "version", "counter", "status", "chunks"});
            if ((kind != "vulkan_structured" && !nested) ||
                integer(lowering["version"]) != 1 ||
                !loop->continue_while_nonzero)
              throw std::invalid_argument(
                  "Unsupported selected control recipe");
            loop->vulkan.emplace();
            auto &control = *loop->vulkan;
            control.counter = name(lowering["counter"]);
            arguments.insert(control.counter);
            if (lowering.obj.inner.count("status")) {
              control.status = name(lowering["status"]);
              arguments.insert(*control.status);
            }
            if (nested) {
              if (version < 5)
                throw std::invalid_argument(
                    "Nested Vulkan control requires execution plan version 5");
              control.nested.emplace();
              if (!lowering["inners"].is_arr())
                throw std::invalid_argument(
                    "Vulkan nested controls must be an array");
              const auto u32 = [&](const JsonValue &value) {
                const auto number = integer(value);
                if (number > std::numeric_limits<std::uint32_t>::max())
                  throw std::invalid_argument(
                      "Vulkan nested descriptor exceeds uint32");
                return static_cast<std::uint32_t>(number);
              };
              for (const auto &value : lowering["inners"].arr.inner) {
                fields(value,
                       {"predicate", "counter", "status", "condition_begin",
                        "body_begin", "end", "max_iterations", "chunk_size"});
                VulkanNestedControlRecipe<std::string>::Inner inner;
                inner.predicate = name(value["predicate"]);
                inner.counter = name(value["counter"]);
                arguments.insert(inner.predicate);
                arguments.insert(inner.counter);
                if (value.obj.inner.count("status")) {
                  inner.status = name(value["status"]);
                  arguments.insert(*inner.status);
                }
                const auto boundary = [&](const char *key) {
                  const auto index = integer(value[key]);
                  if (index > std::numeric_limits<std::size_t>::max())
                    throw std::invalid_argument(
                        "Vulkan nested boundary exceeds address space");
                  return static_cast<std::size_t>(index);
                };
                inner.condition_begin = boundary("condition_begin");
                inner.body_begin = boundary("body_begin");
                inner.end = boundary("end");
                inner.max_iterations = u32(value["max_iterations"]);
                inner.chunk_size = u32(value["chunk_size"]);
                control.nested->inners.push_back(std::move(inner));
              }
              control.nested->validate(
                  loop->condition.size(),
                  loop->condition.size() + loop->body.size(),
                  loop->max_iterations);
            } else {
              if (!lowering["chunks"].is_arr())
                throw std::invalid_argument(
                    "Vulkan control chunks must be an array");
              for (const auto &chunk : lowering["chunks"].arr.inner) {
                fields(chunk, {"iterations", "strategy"});
                const auto count = integer(chunk["iterations"]);
                if (count > std::numeric_limits<std::uint32_t>::max())
                  throw std::invalid_argument(
                      "Vulkan chunk descriptor exceeds uint32 iterations");
                const auto strategy = name(chunk["strategy"]);
                const std::vector<std::string> names{
                    "compact", "chained", "conditional", "coarse_conditional"};
                const auto selected =
                    std::find(names.begin(), names.end(), strategy);
                if (selected == names.end())
                  throw std::invalid_argument(
                      "Unknown selected Vulkan control strategy");
                control.recipe.chunks.push_back(
                    {static_cast<std::uint32_t>(count),
                     static_cast<VulkanControlStrategy>(
                         1 + std::distance(names.begin(), selected))});
              }
              control.recipe.validate(loop->max_iterations);
            }
            const auto only_kernels = [](const auto &region) {
              return std::all_of(region.begin(), region.end(),
                                 [](const auto &op) {
                                   return std::holds_alternative<Kernel>(op);
                                 });
            };
            if (!only_kernels(loop->condition) || !only_kernels(loop->body))
              throw std::invalid_argument(
                  "Vulkan structured control requires flat kernel regions");
          }
          operations.emplace_back(std::move(loop));
        } else if (kind == "kernel") {
          fields(step, {"kind", "entry", "arguments", "label", "launch"});
          Kernel kernel{name(step["entry"]), {}, {}};
          if (auto found = step.obj.inner.find("launch");
              found != step.obj.inner.end()) {
            if (version < 6)
              throw std::invalid_argument(
                  "Selected kernel launch requires execution plan version 6");
            const auto &launch = found->second;
            const auto kind = name(launch["kind"]);
            if (kind == "vulkan_indirect") {
              fields(launch, {"kind", "packet"});
              kernel.indirect_packet = name(launch["packet"]);
              arguments.insert(*kernel.indirect_packet);
            } else if (kind == "cpu_exact" || kind == "cuda_logical_exact" ||
                       kind == "cuda_adaptive") {
              const bool adaptive = kind == "cuda_adaptive";
              if (version < (adaptive ? 8 : 7))
                throw std::invalid_argument(
                    "Bounded launch requires execution plan version 7, or 8 for adaptive");
              if (adaptive)
                fields(launch, {"kind", "extent", "capacity", "block_dim", "update_policy"});
              else
                fields(launch, {"kind", "extent", "capacity"});
              const auto capacity = integer(launch["capacity"]);
              if (!capacity || capacity > std::numeric_limits<std::int32_t>::max())
                throw std::invalid_argument(
                    "Bounded capacity must fit a positive i32");
              kernel.bounded_range = Kernel::BoundedRange{
                  kind != "cpu_exact", name(launch["extent"]),
                  static_cast<std::uint32_t>(capacity)};
              if (adaptive) {
                const auto block = integer(launch["block_dim"]);
                if (!block || block > std::numeric_limits<std::uint32_t>::max())
                  throw std::invalid_argument("Adaptive block dimension must fit a positive u32");
                const auto policy = name(launch["update_policy"]);
                if (policy != "per_node" && policy != "grouped_stateful")
                  throw std::invalid_argument("Unsupported adaptive update policy");
                kernel.bounded_range->block_dim = static_cast<std::uint32_t>(block);
                kernel.bounded_range->grouped_update = policy == "grouped_stateful";
              }
              arguments.insert(kernel.bounded_range->extent);
            } else {
              throw std::invalid_argument("Unsupported kernel launch recipe");
            }
          }
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
    parse_steps(root["operations"], plan.operations, false);
    plan.replay_bindings = plan.resources.runtime_bindings(arguments);
    if (has_initialization)
      parse_steps(root["initialization"], plan.initialization, false);
    if (const auto found = root.obj.inner.find("constraints");
        found != root.obj.inner.end()) {
      if (!found->second.is_arr())
        throw std::invalid_argument("Execution constraints must be an array");
      const auto used = plan.resources.runtime_bindings(arguments);
      auto binding = [&](const JsonValue &value) {
        auto result = name(value);
        if (!used.count(result))
          throw std::invalid_argument("Constraint references unused binding: " +
                                      result);
        return result;
      };
      auto shape = [&](const JsonValue &value, bool positive) {
        if (!value.is_arr())
          throw std::invalid_argument("Constraint shape must be an array");
        std::vector<std::uint64_t> result;
        for (const auto &axis : value.arr.inner) {
          const auto dimension = integer(axis);
          if (positive && !dimension)
            throw std::invalid_argument("Element extents must be positive");
          result.push_back(dimension);
        }
        return result;
      };
      for (const auto &constraint : found->second.arr.inner) {
        const auto kind = name(constraint["kind"]);
        if (kind == "array_layout") {
          fields(constraint, {"kind", "array", "minimum_shape", "exact_shape",
                              "element_shape", "record_bytes", "alignment",
                              "dtype"});
          ArrayLayout layout;
          layout.array = binding(constraint["array"]);
          if (constraint.obj.inner.count("dtype")) {
            layout.required.dtype = name(constraint["dtype"]);
            static const std::set<std::string> supported{
                "f16", "f32", "f64", "i8", "i16", "i32", "i64",
                "u8", "u16", "u32", "u64"};
            if (!supported.count(layout.required.dtype))
              throw std::invalid_argument("Unsupported array layout dtype");
          }
          layout.required.exact_shape =
              constraint.obj.inner.count("exact_shape");
          if (layout.required.exact_shape ==
              (constraint.obj.inner.count("minimum_shape") != 0))
            throw std::invalid_argument(
                "Array layout requires either minimum_shape or exact_shape");
          layout.required.minimum_shape =
              shape(constraint[layout.required.exact_shape ? "exact_shape"
                                                           : "minimum_shape"],
                    false);
          if (constraint.obj.inner.count("element_shape"))
            layout.required.element_shape =
                shape(constraint["element_shape"], true);
          layout.required.record_bytes = integer(constraint["record_bytes"]);
          layout.required.alignment = integer(constraint["alignment"]);
          if (!layout.required.record_bytes || !layout.required.alignment)
            throw std::invalid_argument(
                "Record size and alignment must be positive");
          plan.constraints.emplace_back(std::move(layout));
        } else if (kind == "disjoint") {
          fields(constraint, {"kind", "left", "right", "allow_same_view"});
          plan.constraints.emplace_back(Disjoint{
              binding(constraint["left"]), binding(constraint["right"]),
              optional_bool(constraint, "allow_same_view")});
        } else {
          throw std::invalid_argument("Unsupported execution constraint: " +
                                      kind);
        }
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
