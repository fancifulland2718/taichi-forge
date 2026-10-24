#include "taichi/runtime/native_argpack_storage.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"

#include <limits>
#include <set>

namespace taichi::lang {
namespace {
struct ArgPackDescription {
  static constexpr const char *kind = "memory_argpack";
  static constexpr std::uint32_t version = 1;
  std::string entry;
  std::uint32_t argument;
  std::vector<std::vector<int>> paths;
  std::vector<std::shared_ptr<NativeHostBlobResource>> values;
};

std::shared_ptr<NativeResource> prepare_argpack(
    NativePreparationContext &context, const ArgPackDescription &description) {
  if (!context.resolve_argpack_layout)
    throw std::invalid_argument("Persistent ArgPack requires a compiled module layout resolver");
  const auto layout = context.resolve_argpack_layout(description.entry,
                                                      description.argument);
  if (!layout)
    throw std::invalid_argument("Persistent ArgPack layout resolver returned no layout");
  NativeArgPackStorage::Values values;
  for (std::size_t i = 0; i < description.paths.size(); ++i) {
    const auto &blob = description.values.at(i);
    if (blob->arch() != context.arch || &blob->device() != &context.device)
      throw std::invalid_argument("ArgPack value blob belongs to another device");
    auto path = description.paths[i];
    path.insert(path.begin(), int(description.argument));
    values.emplace(std::move(path), blob->bytes());
  }
  // Storage validates the complete value inventory before allocating anything.
  // Lifted resources remain ordinary plan arguments with existing ownership.
  return std::make_shared<NativeArgPackStorage>(context.arch, context.device,
                                                layout, values, context.upload_owned_data);
}
}  // namespace

void register_argpack_resources(NativeOperationRegistry &registry) {
  registry.define<ArgPackDescription>(
      [](Device &, const ArgPackDescription &) {}, NativeObjectKind::resource);
  registry.register_schema<ArgPackDescription>(
      "memory", "argpack", {}, [](const liong::json::JsonValue &parameters) {
        using namespace operation_json;
        fields(parameters, {"entry", "argument", "value_count", "paths"});
        const auto argument = integer(parameters["argument"]);
        if (argument > std::numeric_limits<int>::max())
          throw std::invalid_argument("ArgPack argument exceeds the compiled parameter ABI");
        const auto &paths = parameters["paths"];
        if (!paths.is_arr() || paths.arr.inner.size() != integer(parameters["value_count"]))
          throw std::invalid_argument("ArgPack value paths do not match their binding inventory");
        ArgPackDescription value{name(parameters["entry"]),
                                  static_cast<std::uint32_t>(argument), {}, {}};
        std::set<std::vector<int>> unique;
        for (const auto &path : paths.arr.inner) {
          if (!path.is_arr() || path.arr.inner.empty())
            throw std::invalid_argument("ArgPack values require nonempty relative paths");
          std::vector<int> indices;
          for (const auto &part : path.arr.inner) {
            const auto index = integer(part);
            if (index > std::numeric_limits<int>::max())
              throw std::invalid_argument("ArgPack member exceeds the compiled parameter ABI");
            indices.push_back(static_cast<int>(index));
          }
          if (!unique.insert(indices).second)
            throw std::invalid_argument("Duplicate ArgPack value path");
          value.paths.push_back(std::move(indices));
        }
        return [value](const NativeOperationBindings &bindings) {
          auto resolved = value;
          for (std::size_t i = 0; i < value.paths.size(); ++i)
            resolved.values.push_back(bindings.at("value_" + std::to_string(i))
                                          .resource<NativeHostBlobResource>());
          return resolved;
        };
      }, {}, {}, {{"value_count", {"value"}}});
#ifdef TI_WITH_LLVM
  registry.register_context_factory<ArgPackDescription>(
      host_arch(), NativeOperationMode::cpu, prepare_argpack);
#endif
#ifdef TI_WITH_CUDA
  registry.register_context_factory<ArgPackDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_argpack);
#endif
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<ArgPackDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline, prepare_argpack);
#endif
}
}  // namespace taichi::lang
