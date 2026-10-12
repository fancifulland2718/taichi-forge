#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <typeindex>
#include <utility>
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Composition modes implemented by FixedExecution. Registration says which
// compiled implementation exists, not whether every device/descriptor is valid.
enum class NativeOperationMode : std::uint32_t {
  cpu = 1,
  cuda_capture = 2,
  vulkan_inline = 4,
};

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

// Build privately, then publish as const. No static registration side effects,
// replacement, implicit device probing or warm-path lookup. Descriptors and factories
// are an internal C++ contract, not an ABI for independently built plugins.
// A factory prepares without submitting work; its result owns scratch and any
// provider/code leases. FixedExecution retains it until recordings retire.
class NativeOperationRegistry {
 public:
  using Probe = std::function<NativeOperationAvailability()>;
  template <typename Descriptor>
  using Validator = std::function<void(Device &, const Descriptor &)>;
  template <typename Descriptor>
  using Factory = std::function<
      std::shared_ptr<NativeOperation>(Arch, Device &, const Descriptor &)>;
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
    std::vector<std::string> ports_;
    std::function<std::shared_ptr<const void>(const NativeOperationBindings &)>
        bind_;
  };

  template <typename Descriptor>
  void register_schema(std::string provider,
                       std::string operation,
                       std::vector<std::string> ports,
                       Decoder<Descriptor> decode,
                       std::string legacy_kind = {}) {
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
        std::move(legacy_kind));
  }

  DecodedOperation decode(const NativeOperationDescription &description) const;
  std::optional<NativeOperationDescription> lower_legacy(
      const liong::json::JsonValue &description) const;
  std::shared_ptr<NativeOperation> prepare(
      Arch arch,
      Device &device,
      const DecodedOperation &operation,
      const NativeOperationBindings &bindings) const;

  template <typename Descriptor>
  void define(Validator<Descriptor> validate) {
    if (!validate)
      throw std::invalid_argument("Native operation requires a validator");
    define_impl(
        Descriptor::kind, Descriptor::version, typeid(Descriptor),
        [validate = std::move(validate)](Device &device, const void *value) {
          validate(device, *static_cast<const Descriptor *>(value));
        });
  }

  template <typename Descriptor>
  void register_factory(Arch arch,
                        NativeOperationMode mode,
                        Factory<Descriptor> factory,
                        Probe probe = {}) {
    if (!factory)
      throw std::invalid_argument("Native operation requires a factory");
    register_impl(
        Descriptor::kind, Descriptor::version, typeid(Descriptor), arch, mode,
        [arch, factory = std::move(factory)](Device &device,
                                             const void *value) {
          return factory(arch, device, *static_cast<const Descriptor *>(value));
        },
        std::move(probe));
  }

  template <typename Descriptor>
  std::shared_ptr<NativeOperation> prepare(Arch arch,
                                           Device &device,
                                           const Descriptor &value) const {
    return prepare_impl(Descriptor::kind, Descriptor::version,
                        typeid(Descriptor), arch, device, &value);
  }

  const NativeOperationCapability *find(const std::string &kind,
                                        std::uint32_t version,
                                        Arch arch) const;
  std::vector<NativeOperationCapability> capabilities() const;
  // Explicit cold probe: may load a dependency, never creates a plan or work
  // buffer. Capability enumeration and prepared replay never invoke it.
  NativeOperationAvailability probe(const std::string &kind,
                                    std::uint32_t version,
                                    Arch arch) const;

 private:
  using Key = std::pair<std::string, std::uint32_t>;
  using ErasedValidator = std::function<void(Device &, const void *)>;
  using ErasedFactory =
      std::function<std::shared_ptr<NativeOperation>(Device &, const void *)>;
  struct Backend {
    NativeOperationCapability capability;
    ErasedFactory factory;
    Probe probe;
  };
  struct Definition {
    std::type_index type;
    ErasedValidator validate;
    std::map<Arch, Backend> backends;
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
  };
  std::map<NativeOperationDescription::Key, Schema> schemas_;
  std::map<std::string, NativeOperationDescription::Key> legacy_schemas_;

  void register_schema_impl(NativeOperationDescription::Key key,
                            Key definition_key,
                            std::type_index type,
                            std::vector<std::string> ports,
                            ErasedDecoder decode,
                            std::string legacy_kind);

  void define_impl(const std::string &kind,
                   std::uint32_t version,
                   std::type_index type,
                   ErasedValidator validate);
  void register_impl(const std::string &kind,
                     std::uint32_t version,
                     std::type_index type,
                     Arch arch,
                     NativeOperationMode mode,
                     ErasedFactory factory,
                     Probe probe);
  const Definition &definition(const Key &key, std::type_index type) const;
  std::shared_ptr<NativeOperation> prepare_impl(const std::string &kind,
                                                std::uint32_t version,
                                                std::type_index type,
                                                Arch arch,
                                                Device &device,
                                                const void *value) const;
};

const NativeOperationRegistry &native_operation_registry();

// Explicit entry points keep registration deterministic under static linking.
void register_compact_operation(NativeOperationRegistry &registry);
void register_sort_operation(NativeOperationRegistry &registry);
void register_fft_operation(NativeOperationRegistry &registry);
void register_blas_operations(NativeOperationRegistry &registry);

}  // namespace taichi::lang
