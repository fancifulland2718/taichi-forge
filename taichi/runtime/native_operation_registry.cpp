#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"
#include <set>

namespace taichi::lang {
void NativeOperationRegistry::register_schema_impl(
    NativeOperationDescription::Key key,
    Key definition_key,
    std::type_index type,
    std::vector<std::string> ports,
    ErasedDecoder decode,
    std::string legacy_kind,
    std::vector<std::string> optional_ports) {
  using operation_json::name;
  name(liong::json::JsonValue(std::get<0>(key)));
  name(liong::json::JsonValue(std::get<1>(key)));
  definition(definition_key, type);
  std::set<std::string> unique;
  for (const auto &port : ports) {
    name(liong::json::JsonValue(port));
    if (!unique.insert(port).second)
      throw std::invalid_argument("Duplicate native schema binding: " + port);
  }
  for (const auto &port : optional_ports) {
    name(liong::json::JsonValue(port));
    if (!unique.insert(port).second)
      throw std::invalid_argument("Duplicate native schema binding: " + port);
  }
  if (schemas_.count(key) ||
      (!legacy_kind.empty() && legacy_schemas_.count(legacy_kind)))
    throw std::invalid_argument("Duplicate native operation schema");
  if (!legacy_kind.empty()) {
    name(liong::json::JsonValue(legacy_kind));
    if (unique.count("kind") ||
        definition(definition_key, type).kind != NativeObjectKind::operation)
      throw std::invalid_argument("Legacy native binding conflicts with kind");
  }
  auto inserted = schemas_.emplace(
      key, Schema{std::move(definition_key), type, std::move(ports),
                  std::move(decode), std::move(optional_ports)});
  try {
    if (!legacy_kind.empty())
      legacy_schemas_.emplace(std::move(legacy_kind), std::move(key));
  } catch (...) {
    schemas_.erase(inserted.first);
    throw;
  }
}

NativeOperationRegistry::DecodedOperation NativeOperationRegistry::decode(
    const NativeOperationDescription &description,
    NativeObjectKind object_kind) const {
  const auto found = schemas_.find(description.key());
  if (found == schemas_.end())
    throw std::invalid_argument(
        "Unregistered native operation schema: " + description.provider + "/" +
        description.operation);
  const auto &schema = found->second;
  if (definition(schema.definition, schema.type).kind != object_kind)
    throw std::invalid_argument(
        "Native operation/resource schema kind mismatch");
  for (const auto &port : schema.ports) {
    if (!description.bindings.count(port))
      throw std::invalid_argument("Missing native operation binding: " + port);
  }
  DecodedOperation result;
  result.kind_ = schema.definition.first;
  result.version_ = schema.definition.second;
  result.type_ = schema.type;
  result.object_kind_ = object_kind;
  for (const auto &[port, slot] : description.bindings) {
    if (std::find(schema.ports.begin(), schema.ports.end(), port) ==
            schema.ports.end() &&
        std::find(schema.optional_ports.begin(), schema.optional_ports.end(),
                  port) == schema.optional_ports.end())
      throw std::invalid_argument("Unknown native schema binding: " + port);
    operation_json::name(liong::json::JsonValue(slot));
    result.ports_.push_back(port);
  }
  try {
    result.bind_ = schema.decode(description.parameters);
  } catch (const std::exception &error) {
    throw std::invalid_argument(description.provider + "/" +
                                description.operation + ": " + error.what());
  }
  return result;
}

std::optional<NativeOperationDescription> NativeOperationRegistry::lower_legacy(
    const liong::json::JsonValue &description) const {
  const auto kind = operation_json::name(description["kind"]);
  const auto found = legacy_schemas_.find(kind);
  if (found == legacy_schemas_.end())
    return std::nullopt;
  NativeOperationDescription result;
  std::tie(result.provider, result.operation, result.version) = found->second;
  result.parameters = description;
  result.parameters.obj.inner.erase("kind");
  for (const auto &port : schemas_.at(found->second).ports) {
    result.bindings.emplace(port, operation_json::name(description[port]));
    result.parameters.obj.inner.erase(port);
  }
  for (const auto &port : schemas_.at(found->second).optional_ports) {
    if (const auto input = description.obj.inner.find(port);
        input != description.obj.inner.end()) {
      result.bindings.emplace(port, operation_json::name(input->second));
      result.parameters.obj.inner.erase(port);
    }
  }
  return result;
}

std::shared_ptr<NativeOperation> NativeOperationRegistry::prepare(
    Arch arch,
    Device &device,
    const DecodedOperation &operation,
    const NativeOperationBindings &bindings) const {
  NativePreparationContext context(arch, device);
  return prepare(context, operation, bindings);
}

std::shared_ptr<NativeOperation> NativeOperationRegistry::prepare(
    NativePreparationContext &context,
    const DecodedOperation &operation,
    const NativeOperationBindings &bindings) const {
  return std::dynamic_pointer_cast<NativeOperation>(
      prepare_bound(context, operation, bindings, NativeObjectKind::operation));
}

std::shared_ptr<NativeResource> NativeOperationRegistry::prepare_resource(
    Arch arch,
    Device &device,
    const DecodedOperation &resource,
    const NativeOperationBindings &bindings) const {
  NativePreparationContext context(arch, device);
  return prepare_resource(context, resource, bindings);
}

std::shared_ptr<NativeResource> NativeOperationRegistry::prepare_resource(
    NativePreparationContext &context,
    const DecodedOperation &resource,
    const NativeOperationBindings &bindings) const {
  return prepare_bound(context, resource, bindings, NativeObjectKind::resource);
}

std::shared_ptr<NativeResource> NativeOperationRegistry::prepare_bound(
    NativePreparationContext &context,
    const DecodedOperation &operation,
    const NativeOperationBindings &bindings,
    NativeObjectKind kind) const {
  if (operation.object_kind_ != kind)
    throw std::invalid_argument(
        "Native operation/resource binding kind mismatch");
  if (!operation.bind_ || bindings.size() != operation.ports_.size())
    throw std::invalid_argument(
        "Native operation storage set does not match its schema");
  for (const auto &port : operation.ports_)
    if (!bindings.count(port))
      throw std::invalid_argument("Missing native operation storage: " + port);
  const auto value = operation.bind_(bindings);
  return prepare_impl(operation.kind_, operation.version_, operation.type_,
                      context, value.get(), kind);
}

void NativeOperationRegistry::define_impl(const std::string &kind,
                                          std::uint32_t version,
                                          std::type_index type,
                                          ErasedValidator validate,
                                          NativeObjectKind object_kind) {
  if (kind.empty() || kind.find('\0') != std::string::npos || !version)
    throw std::invalid_argument("Native operation requires a kind and version");
  if (!definitions_
           .emplace(Key{kind, version},
                    Definition{type, std::move(validate), {}, object_kind})
           .second)
    throw std::invalid_argument("Duplicate native operation definition: " +
                                kind);
}

const NativeOperationRegistry::Definition &NativeOperationRegistry::definition(
    const Key &key,
    std::type_index type) const {
  const auto found = definitions_.find(key);
  if (found == definitions_.end())
    throw std::invalid_argument("Unregistered native operation definition: " +
                                key.first);
  if (found->second.type != type)
    throw std::invalid_argument("Native operation descriptor type mismatch: " +
                                key.first);
  return found->second;
}

void NativeOperationRegistry::register_impl(const std::string &kind,
                                            std::uint32_t version,
                                            std::type_index type,
                                            Arch arch,
                                            NativeOperationMode mode,
                                            ErasedFactory factory,
                                            ContextProbe probe) {
  const Key key{kind, version};
  definition(key, type);
  if (!((mode == NativeOperationMode::cpu && arch_is_cpu(arch)) ||
        (mode == NativeOperationMode::cuda_capture && arch == Arch::cuda) ||
        (mode == NativeOperationMode::vulkan_inline && arch == Arch::vulkan)))
    throw std::invalid_argument(
        "Native operation composition mode/backend mismatch");
  if (!definitions_.at(key)
           .backends
           .emplace(arch, Backend{{kind, version, arch, mode},
                                  std::move(factory),
                                  std::move(probe)})
           .second)
    throw std::invalid_argument("Duplicate native operation backend: " + kind);
}

const NativeOperationCapability *NativeOperationRegistry::find(
    const std::string &kind,
    std::uint32_t version,
    Arch arch) const {
  const auto found = definitions_.find({kind, version});
  if (found == definitions_.end() ||
      found->second.kind != NativeObjectKind::operation)
    return nullptr;
  const auto backend = found->second.backends.find(arch);
  return backend == found->second.backends.end() ? nullptr
                                                 : &backend->second.capability;
}

std::vector<NativeOperationCapability> NativeOperationRegistry::capabilities()
    const {
  std::vector<NativeOperationCapability> result;
  for (const auto &[key, definition] : definitions_)
    if (definition.kind == NativeObjectKind::operation)
      for (const auto &[arch, backend] : definition.backends)
        result.push_back(backend.capability);
  return result;
}

NativeOperationAvailability NativeOperationRegistry::probe(
    const std::string &kind,
    std::uint32_t version,
    Arch arch,
    NativeObjectKind object_kind) const {
  NativeProviderSession providers;
  return probe(kind, version, arch, providers, object_kind);
}

NativeOperationAvailability NativeOperationRegistry::probe(
    const std::string &kind,
    std::uint32_t version,
    Arch arch,
    NativeProviderSession &providers,
    NativeObjectKind object_kind) const {
  const auto found = definitions_.find({kind, version});
  if (found == definitions_.end() || found->second.kind != object_kind)
    return {};
  const auto backend = found->second.backends.find(arch);
  if (backend == found->second.backends.end())
    return {};
  return backend->second.probe ? backend->second.probe(providers)
                               : NativeOperationAvailability{true, 0};
}

std::shared_ptr<NativeResource> NativeOperationRegistry::prepare_impl(
    const std::string &kind,
    std::uint32_t version,
    std::type_index type,
    NativePreparationContext &context,
    const void *value,
    NativeObjectKind object_kind) const {
  const auto arch = context.arch;
  auto &device = context.device;
  const auto &entry = definition({kind, version}, type);
  if (entry.kind != object_kind)
    throw std::invalid_argument(
        "Native operation/resource factory kind mismatch");
  const auto backend = entry.backends.find(arch);
  if (backend == entry.backends.end())
    throw std::invalid_argument("No native operation factory for " + kind +
                                " v" + std::to_string(version) + " on " +
                                arch_name(arch));
  entry.validate(device, value);
  if (backend->second.probe &&
      !backend->second.probe(context.providers).available)
    throw std::invalid_argument("Native operation provider unavailable: " +
                                kind);
  auto result = backend->second.factory(context, value);
  if (!result || result->arch() != arch || &result->device() != &device)
    throw std::invalid_argument(
        "Native operation factory returned an invalid owner: " + kind);
  if (object_kind == NativeObjectKind::operation &&
      !std::dynamic_pointer_cast<NativeOperation>(result))
    throw std::invalid_argument("Native factory did not return an operation: " +
                                kind);
  if (object_kind == NativeObjectKind::resource &&
      std::dynamic_pointer_cast<NativeOperation>(result))
    throw std::invalid_argument(
        "Native resource factory returned an operation: " + kind);
  return result;
}

const NativeOperationRegistry &native_operation_registry() {
  static const auto registry = [] {
    NativeOperationRegistry result;
    register_compact_operation(result);
    register_sort_operation(result);
    register_fft_operation(result);
    register_blas_operations(result);
    return result;
  }();
  return registry;
}
}  // namespace taichi::lang
