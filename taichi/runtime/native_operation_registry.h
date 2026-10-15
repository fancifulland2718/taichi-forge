#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <typeindex>
#include <utility>
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_operation_description.h"
#include "taichi/runtime/native_provider.h"

namespace taichi::lang {
class RuntimeFaultDomain;

struct NativeOperationCapability {
  std::string kind;
  std::uint32_t version;
  Arch arch;
  NativeOperationMode mode;
};

struct NativeOperationAvailability {
  bool available{false};
  // Zero denotes an unversioned built-in or an unavailable dependency.
  std::uint32_t library_version{0};
};

struct NativePreparationContext {
  NativePreparationContext(
      Arch arch,
      Device &device,
      NativeProviderLocations locations = {},
      std::shared_ptr<RuntimeFaultDomain> fault_domain = {})
      : arch(arch),
        device(device),
        providers(std::move(locations)),
        fault_domain(std::move(fault_domain)) {
  }
  Arch arch;
  Device &device;
  NativeProviderSession providers;
  std::shared_ptr<RuntimeFaultDomain> fault_domain;
};

// Build privately, then publish as const. No static registration side effects,
// replacement, implicit device probing or warm-path lookup. Descriptors and
// factories are an internal C++ contract, not an ABI for independently built
// plugins. A factory prepares without submitting work; its result owns scratch
// and any provider/code leases. FixedExecution retains it until recordings
// retire.
class NativeOperationRegistry {
 public:
  // Cold, finite groups named prefix_0 ... prefix_(count-1). Count is supplied
  // by the description; the actual binding inventory bounds expansion, not an
  // arbitrary library/application limit. All members of every group are
  // required.
  struct BindingGroup {
    std::string count_parameter;
    std::vector<std::string> prefixes;
  };
  using Probe = std::function<NativeOperationAvailability()>;
  template <typename Descriptor>
  using Validator = std::function<void(Device &, const Descriptor &)>;
  template <typename Descriptor>
  using Factory = std::function<
      std::shared_ptr<NativeResource>(Arch, Device &, const Descriptor &)>;
  using ContextProbe =
      std::function<NativeOperationAvailability(NativeProviderSession &)>;
  template <typename Descriptor>
  using ContextFactory =
      std::function<std::shared_ptr<NativeResource>(NativePreparationContext &,
                                                    const Descriptor &)>;
  template <typename Descriptor>
  using Binder = std::function<Descriptor(const NativeOperationBindings &)>;
  template <typename Descriptor>
  using Decoder =
      std::function<Binder<Descriptor>(const liong::json::JsonValue &)>;

  // Decoding owns immutable parameters and resolves a registered typed binder.
  // It never probes a dependency, allocates device memory or creates a plan.
  class DecodedOperation {
   public:
    const std::string &kind() const {
      return kind_;
    }
    std::uint32_t version() const {
      return version_;
    }

   private:
    friend class NativeOperationRegistry;
    std::string kind_;
    std::uint32_t version_{0};
    std::type_index type_{typeid(void)};
    NativeObjectKind object_kind_{NativeObjectKind::operation};
    std::vector<std::string> ports_;
    std::function<std::shared_ptr<const void>(const NativeOperationBindings &)>
        bind_;
  };

  template <typename Descriptor>
  void register_schema(std::string provider,
                       std::string operation,
                       std::vector<std::string> ports,
                       Decoder<Descriptor> decode,
                       std::string legacy_kind = {},
                       std::vector<std::string> optional_ports = {},
                       std::optional<BindingGroup> binding_group = {}) {
    if (!decode)
      throw std::invalid_argument("Native schema requires a decoder");
    register_schema_impl(
        {std::move(provider), std::move(operation), Descriptor::version},
        {Descriptor::kind, Descriptor::version}, typeid(Descriptor),
        std::move(ports),
        [decode = std::move(decode)](const liong::json::JsonValue &parameters) {
          auto bind = decode(parameters);
          if (!bind)
            throw std::invalid_argument(
                "Native schema returned an empty binder");
          return
              [bind = std::move(bind)](const NativeOperationBindings &bindings)
                  -> std::shared_ptr<const void> {
                return std::make_shared<Descriptor>(bind(bindings));
              };
        },
        std::move(legacy_kind), std::move(optional_ports),
        std::move(binding_group));
  }

  DecodedOperation decode(
      const NativeOperationDescription &description,
      NativeObjectKind kind = NativeObjectKind::operation) const;
  std::optional<NativeOperationDescription> lower_legacy(
      const liong::json::JsonValue &description) const;
  std::shared_ptr<NativeOperation> prepare(
      Arch arch,
      Device &device,
      const DecodedOperation &operation,
      const NativeOperationBindings &bindings) const;
  std::shared_ptr<NativeResource> prepare_resource(
      Arch arch,
      Device &device,
      const DecodedOperation &resource,
      const NativeOperationBindings &bindings) const;
  std::shared_ptr<NativeOperation> prepare(
      NativePreparationContext &context,
      const DecodedOperation &operation,
      const NativeOperationBindings &bindings) const;
  std::shared_ptr<NativeResource> prepare_resource(
      NativePreparationContext &context,
      const DecodedOperation &resource,
      const NativeOperationBindings &bindings) const;

  template <typename Descriptor>
  void define(Validator<Descriptor> validate,
              NativeObjectKind kind = NativeObjectKind::operation) {
    if (!validate)
      throw std::invalid_argument("Native operation requires a validator");
    define_impl(
        Descriptor::kind, Descriptor::version, typeid(Descriptor),
        [validate = std::move(validate)](Device &device, const void *value) {
          validate(device, *static_cast<const Descriptor *>(value));
        },
        kind);
  }

  template <typename Descriptor>
  void register_factory(Arch arch,
                        NativeOperationMode mode,
                        Factory<Descriptor> factory,
                        Probe probe = {}) {
    if (!factory)
      throw std::invalid_argument("Native operation requires a factory");
    // Built-ins without deployment dependencies use the same factory path.
    register_context_factory<Descriptor>(
        arch, mode,
        [factory = std::move(factory)](NativePreparationContext &context,
                                       const Descriptor &value) {
          return factory(context.arch, context.device, value);
        },
        probe ? ContextProbe([probe = std::move(probe)](
                                 NativeProviderSession &) { return probe(); })
              : ContextProbe{});
  }

  template <typename Descriptor>
  void register_context_factory(Arch arch,
                                NativeOperationMode mode,
                                ContextFactory<Descriptor> factory,
                                ContextProbe probe = {}) {
    if (!factory)
      throw std::invalid_argument("Native operation requires a factory");
    register_impl(
        Descriptor::kind, Descriptor::version, typeid(Descriptor), arch, mode,
        [factory = std::move(factory)](NativePreparationContext &context,
                                       const void *value) {
          return factory(context, *static_cast<const Descriptor *>(value));
        },
        std::move(probe));
  }

  template <typename Descriptor>
  std::shared_ptr<NativeOperation> prepare(Arch arch,
                                           Device &device,
                                           const Descriptor &value) const {
    NativePreparationContext context(arch, device);
    return prepare(context, value);
  }

  template <typename Descriptor>
  std::shared_ptr<NativeOperation> prepare(NativePreparationContext &context,
                                           const Descriptor &value) const {
    return std::dynamic_pointer_cast<NativeOperation>(
        prepare_impl(Descriptor::kind, Descriptor::version, typeid(Descriptor),
                     context, &value, NativeObjectKind::operation));
  }

  const NativeOperationCapability *find(const std::string &kind,
                                        std::uint32_t version,
                                        Arch arch) const;
  std::vector<NativeOperationCapability> capabilities() const;
  // Explicit cold probe: may load a dependency, never creates a plan or work
  // buffer. Capability enumeration and prepared replay never invoke it.
  NativeOperationAvailability probe(
      const std::string &kind,
      std::uint32_t version,
      Arch arch,
      NativeObjectKind object_kind = NativeObjectKind::operation) const;
  NativeOperationAvailability probe(
      const std::string &kind,
      std::uint32_t version,
      Arch arch,
      NativeProviderSession &providers,
      NativeObjectKind object_kind = NativeObjectKind::operation) const;

 private:
  using Key = std::pair<std::string, std::uint32_t>;
  using ErasedValidator = std::function<void(Device &, const void *)>;
  using ErasedFactory =
      std::function<std::shared_ptr<NativeResource>(NativePreparationContext &,
                                                    const void *)>;
  struct Backend {
    NativeOperationCapability capability;
    ErasedFactory factory;
    ContextProbe probe;
  };
  struct Definition {
    std::type_index type;
    ErasedValidator validate;
    std::map<Arch, Backend> backends;
    NativeObjectKind kind;
  };
  std::map<Key, Definition> definitions_;
  using ErasedBinder = std::function<std::shared_ptr<const void>(
      const NativeOperationBindings &)>;
  using ErasedDecoder =
      std::function<ErasedBinder(const liong::json::JsonValue &)>;
  struct Schema {
    Key definition;
    std::type_index type;
    std::vector<std::string> ports;
    ErasedDecoder decode;
    std::vector<std::string> optional_ports;
    std::optional<BindingGroup> binding_group;
  };
  std::map<NativeOperationDescription::Key, Schema> schemas_;
  std::map<std::string, NativeOperationDescription::Key> legacy_schemas_;

  void register_schema_impl(NativeOperationDescription::Key key,
                            Key definition_key,
                            std::type_index type,
                            std::vector<std::string> ports,
                            ErasedDecoder decode,
                            std::string legacy_kind,
                            std::vector<std::string> optional_ports,
                            std::optional<BindingGroup> binding_group);

  void define_impl(const std::string &kind,
                   std::uint32_t version,
                   std::type_index type,
                   ErasedValidator validate,
                   NativeObjectKind object_kind);
  void register_impl(const std::string &kind,
                     std::uint32_t version,
                     std::type_index type,
                     Arch arch,
                     NativeOperationMode mode,
                     ErasedFactory factory,
                     ContextProbe probe);
  const Definition &definition(const Key &key, std::type_index type) const;
  std::shared_ptr<NativeResource> prepare_bound(
      NativePreparationContext &context,
      const DecodedOperation &object,
      const NativeOperationBindings &bindings,
      NativeObjectKind kind) const;
  std::shared_ptr<NativeResource> prepare_impl(
      const std::string &kind,
      std::uint32_t version,
      std::type_index type,
      NativePreparationContext &context,
      const void *value,
      NativeObjectKind object_kind) const;
};

const NativeOperationRegistry &native_operation_registry();

// Explicit entry points keep registration deterministic under static linking.
void register_compact_operation(NativeOperationRegistry &registry);
void register_sort_operation(NativeOperationRegistry &registry);
void register_fft_operation(NativeOperationRegistry &registry);
void register_blas_operations(NativeOperationRegistry &registry);
void register_cusparse_operations(NativeOperationRegistry &registry);
void register_cudss_operations(NativeOperationRegistry &registry);
void register_cutensor_operations(NativeOperationRegistry &registry);
void register_cusparselt_operations(NativeOperationRegistry &registry);
void register_amgx_operations(NativeOperationRegistry &registry);

}  // namespace taichi::lang
