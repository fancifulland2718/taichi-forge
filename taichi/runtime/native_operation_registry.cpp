#include "taichi/runtime/native_operation_registry.h"

namespace taichi::lang {
void NativeOperationRegistry::define_impl(const std::string &kind,
                                          std::uint32_t version,
                                          std::type_index type,
                                          ErasedValidator validate) {
  if (kind.empty() || kind.find('\0') != std::string::npos || !version)
    throw std::invalid_argument("Native operation requires a kind and version");
  if (!definitions_
           .emplace(Key{kind, version},
                    Definition{type, std::move(validate), {}})
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
                                            Probe probe) {
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
  if (found == definitions_.end())
    return nullptr;
  const auto backend = found->second.backends.find(arch);
  return backend == found->second.backends.end() ? nullptr
                                                 : &backend->second.capability;
}

std::vector<NativeOperationCapability> NativeOperationRegistry::capabilities()
    const {
  std::vector<NativeOperationCapability> result;
  for (const auto &[key, definition] : definitions_)
    for (const auto &[arch, backend] : definition.backends)
      result.push_back(backend.capability);
  return result;
}

NativeOperationAvailability NativeOperationRegistry::probe(
    const std::string &kind,
    std::uint32_t version,
    Arch arch) const {
  const auto found = definitions_.find({kind, version});
  if (found == definitions_.end())
    return {};
  const auto backend = found->second.backends.find(arch);
  if (backend == found->second.backends.end())
    return {};
  return backend->second.probe ? backend->second.probe()
                               : NativeOperationAvailability{true, 0};
}

std::shared_ptr<NativeOperation> NativeOperationRegistry::prepare_impl(
    const std::string &kind,
    std::uint32_t version,
    std::type_index type,
    Arch arch,
    Device &device,
    const void *value) const {
  const auto &entry = definition({kind, version}, type);
  const auto backend = entry.backends.find(arch);
  if (backend == entry.backends.end())
    throw std::invalid_argument("No native operation factory for " + kind +
                                " v" + std::to_string(version) + " on " +
                                arch_name(arch));
  entry.validate(device, value);
  if (backend->second.probe && !backend->second.probe().available)
    throw std::invalid_argument("Native operation provider unavailable: " +
                                kind);
  auto result = backend->second.factory(device, value);
  if (!result || result->arch() != arch || &result->device() != &device)
    throw std::invalid_argument(
        "Native operation factory returned an invalid owner: " + kind);
  return result;
}

const NativeOperationRegistry &native_operation_registry() {
  static const auto registry = [] {
    NativeOperationRegistry result;
    register_compact_operation(result);
    register_sort_operation(result);
    register_fft_operation(result);
    return result;
  }();
  return registry;
}
}  // namespace taichi::lang
