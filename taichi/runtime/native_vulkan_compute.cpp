#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/common/operation_json.h"
#include "picosha2.h"

#include <cstring>
#include <limits>
#include <set>
#ifdef TI_WITH_VULKAN
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_shader_bindings.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "external/SPIRV-Reflect/spirv_reflect.h"
#endif

namespace taichi::lang {
namespace {
using Json = liong::json::JsonValue;
using namespace operation_json;

uint32_t u32(const Json &value) {
  const auto n = integer(value);
  if (n > UINT32_MAX)
    throw std::invalid_argument("Compute integer exceeds uint32 ABI");
  return uint32_t(n);
}
std::string digest(const Json &value) {
  auto result = name(value);
  if (result.size() != 64 ||
      result.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument("Compute expected_shader requires SHA256");
  return result;
}
void same_device(Device &device, const NativeResource &resource) {
  if (resource.arch() != Arch::vulkan || &resource.device() != &device)
    throw std::invalid_argument(
        "Compute resource requires the same Vulkan device");
}
struct PipelineDescription {
  static constexpr const char *kind = "vulkan_compute_pipeline";
  static constexpr uint32_t version = 1;
  std::shared_ptr<NativeHostBlobResource> shader;
  std::string expected;
};
class PipelineAccess : public NativeResource {
 public:
  virtual const std::string &contract() const = 0;
#ifdef TI_WITH_VULKAN
  virtual vulkan::VulkanPipeline &pipeline() const = 0;
  virtual uint32_t minimum_bytes(uint32_t set, uint32_t binding) const = 0;
#endif
};
struct DispatchDescription {
  static constexpr const char *kind = "vulkan_compute_dispatch";
  static constexpr uint32_t version = 1;
  struct Buffer {
    uint32_t set, binding, array;
    bool storage;
  };
  std::shared_ptr<PipelineAccess> pipeline;
  std::string expected;
  std::array<uint32_t, 3> groups{};
  std::vector<Buffer> buffers;
  std::vector<NativeArrayBinding> arrays;
};

#ifdef TI_WITH_VULKAN
vulkan::VulkanDevice &owner_device(NativePreparationContext &c) {
  auto *device = dynamic_cast<vulkan::VulkanDevice *>(&c.device);
  if (!device || !c.gfx_runtime || c.gfx_runtime->get_ti_device() != device ||
      !c.fault_domain)
    throw std::invalid_argument(
        "Compute requires its owner's Vulkan GFX runtime");
  return *device;
}

class PreparedPipeline final : public PipelineAccess {
 public:
  PreparedPipeline(NativePreparationContext &c, const PipelineDescription &d)
      : device_(owner_device(c)),
        fault_(c.fault_domain),
        contract_(d.expected) {
    const auto &bytes = d.shader->bytes();
    if (bytes.size() < 20 || bytes.size() % 4)
      throw std::invalid_argument(
          "Compute shader requires complete SPIR-V words");
    if (picosha2::hash256_hex_string(bytes.begin(), bytes.end()) != contract_)
      throw std::invalid_argument("Compute shader contract mismatch");
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    if (words.front() != 0x07230203u)
      throw std::invalid_argument("Compute shader requires SPIR-V magic");
    spv_reflect::ShaderModule reflection(bytes.size(), words.data());
    if (reflection.GetResult() != SPV_REFLECT_RESULT_SUCCESS)
      throw std::invalid_argument("Cannot reflect compute shader");
    const auto &module = reflection.GetShaderModule();
    const auto *entry = spvReflectGetEntryPoint(&module, "main");
    if (!entry || entry->shader_stage != SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT)
      throw std::invalid_argument(
          "Compute shader requires a compute main entry");
    if (entry->used_push_constant_count)
      throw std::invalid_argument("Compute v1 has no push constants contract");
    const auto &limits = device_.get_vk_physical_device_props().limits;
    const uint32_t size[]{entry->local_size.x, entry->local_size.y,
                          entry->local_size.z};
    uint64_t invocations = 1;
    for (size_t i = 0; i < 3; ++i) {
      if (!size[i] || size[i] > limits.maxComputeWorkGroupSize[i] ||
          invocations > limits.maxComputeWorkGroupInvocations / size[i])
        throw std::invalid_argument(
            "Compute local size exceeds physical device limits");
      invocations *= size[i];
    }
    uint64_t storage = 0, uniform = 0;
    for (uint32_t i = 0; i < module.descriptor_binding_count; ++i) {
      const auto &binding = module.descriptor_bindings[i];
      if (binding.count != 1 ||
          (binding.descriptor_type !=
               SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
           binding.descriptor_type !=
               SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER))
        throw std::invalid_argument(
            "Compute v1 requires scalar buffer descriptors");
      if (binding.set >= limits.maxBoundDescriptorSets ||
          binding.binding > INT32_MAX)
        throw std::invalid_argument(
            "Compute descriptor exceeds device or command limits");
      storage +=
          binding.descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      uniform +=
          binding.descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      minimum_bytes_[{binding.set, binding.binding}] = binding.block.size;
    }
    if (storage > limits.maxPerStageDescriptorStorageBuffers ||
        storage > limits.maxDescriptorSetStorageBuffers ||
        uniform > limits.maxPerStageDescriptorUniformBuffers ||
        uniform > limits.maxDescriptorSetUniformBuffers ||
        storage + uniform > limits.maxPerStageResources)
      throw std::invalid_argument(
          "Compute descriptors exceed physical device limits");
    // Direct construction preserves cold reflection/driver errors. The caller
    // owns the fault domain and handles failed preparation transactionally.
    vulkan::VulkanPipeline::Params params;
    params.device = &device_;
    params.name = "aot_compute";
    vulkan::SpirvCodeView code;
    code.data = words.data();
    code.size = bytes.size();
    code.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    params.code.push_back(code);
    pipeline_ = std::make_unique<vulkan::VulkanPipeline>(params);
  }
  ~PreparedPipeline() override {
    if (!fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized)
      pipeline_.release();
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  bool retains_input(const std::string &) const override {
    return false;
  }
  const std::string &contract() const override {
    return contract_;
  }
  vulkan::VulkanPipeline &pipeline() const override {
    return *pipeline_;
  }
  uint32_t minimum_bytes(uint32_t set, uint32_t binding) const override {
    const auto found = minimum_bytes_.find({set, binding});
    if (found == minimum_bytes_.end())
      throw std::invalid_argument(
          "Compute descriptor is absent from shader reflection");
    return found->second;
  }

 private:
  vulkan::VulkanDevice &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::string contract_;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> minimum_bytes_;
  std::unique_ptr<vulkan::VulkanPipeline> pipeline_;
};

class PreparedDispatch final : public NativeOperation {
  struct Payload {
    std::shared_ptr<PipelineAccess> pipeline;
    PreparedVulkanShaderBindings bindings;
  };

 public:
  PreparedDispatch(NativePreparationContext &c, const DispatchDescription &d)
      : device_(owner_device(c)),
        fault_(c.fault_domain),
        groups_(d.groups),
        payload_(std::make_unique<Payload>()) {
    const auto &limits = device_.get_vk_physical_device_props().limits;
    for (size_t i = 0; i < 3; ++i)
      if (groups_[i] > limits.maxComputeWorkGroupCount[i])
        throw std::invalid_argument(
            "Compute dispatch exceeds physical device limits");
    payload_->pipeline = d.pipeline;
    VulkanShaderResourceSets sets;
    for (const auto &binding : d.buffers) {
      const auto &array = d.arrays.at(binding.array);
      buffers_.push_back(array.pointer);
      const auto alignment = binding.storage
                                 ? limits.minStorageBufferOffsetAlignment
                                 : limits.minUniformBufferOffsetAlignment;
      const auto maximum = binding.storage ? limits.maxStorageBufferRange
                                           : limits.maxUniformBufferRange;
      if (!array.bytes || array.bytes > maximum ||
          array.pointer.offset % alignment ||
          array.bytes < d.pipeline->minimum_bytes(binding.set, binding.binding))
        throw std::invalid_argument(
            "Compute buffer range/alignment violates shader or device limits");
      const auto usage =
          binding.storage ? AllocUsage::Storage : AllocUsage::Uniform;
      if (!int(device_.allocation_usage(array.pointer) & usage))
        throw std::invalid_argument("Compute buffer lacks the required usage");
      auto &set = sets[binding.set];
      if (!set)
        set = device_.create_resource_set_unique();
      if (!set)
        throw std::runtime_error("Compute descriptor set creation failed");
      if (binding.storage)
        set->rw_buffer(binding.binding, array.pointer, size_t(array.bytes));
      else
        set->buffer(binding.binding, array.pointer, size_t(array.bytes));
    }
    payload_->bindings =
        prepare_vulkan_shader_bindings(d.pipeline->pipeline(), std::move(sets));
  }
  ~PreparedDispatch() override {
    if (!fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized)
      payload_.release();
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    // The reflected compute schema accepts buffers only. Publish the complete
    // actual binding set so detached recording can validate its resource owner.
    return {{}, false, buffers_,
            std::vector<NativeVulkanRecordingRequirements::ImageUse>{}};
  }
  void record_vulkan(Device *, CommandList *commands) override {
    commands->memory_barrier();
    commands->bind_pipeline(&payload_->pipeline->pipeline());
    for (const auto &[index, set] : payload_->bindings)
      TI_ERROR_IF(commands->bind_shader_resources(set.get(), index) !=
                      RhiResult::success,
                  "Compute descriptor binding failed");
    TI_ERROR_IF(commands->dispatch(groups_[0], groups_[1], groups_[2]) !=
                    RhiResult::success,
                "Compute dispatch failed");
    commands->memory_barrier();
  }

 private:
  vulkan::VulkanDevice &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::array<uint32_t, 3> groups_;
  std::vector<DeviceAllocation> buffers_;
  std::unique_ptr<Payload> payload_;
};
#endif
}  // namespace

void register_vulkan_compute_operations(NativeOperationRegistry &registry) {
  registry.define<PipelineDescription>(
      [](Device &device, const PipelineDescription &d) {
        same_device(device, *d.shader);
      },
      NativeObjectKind::resource);
  registry.register_schema<PipelineDescription>(
      "vulkan_compute", "pipeline", {"shader"}, [](const Json &p) {
        fields(p, {"expected_shader"});
        const auto expected = digest(p["expected_shader"]);
        return [expected](const NativeOperationBindings &b) {
          return PipelineDescription{
              b.at("shader").resource<NativeHostBlobResource>(), expected};
        };
      });
  registry.define<DispatchDescription>([](Device &device,
                                          const DispatchDescription &d) {
    same_device(device, *d.pipeline);
    if (d.pipeline->contract() != d.expected)
      throw std::invalid_argument("Compute shader contract mismatch");
    for (const auto &array : d.arrays) {
      if (array.pointer.device != &device || array.bytes > SIZE_MAX ||
          array.pointer.offset > SIZE_MAX)
        throw std::invalid_argument(
            "Compute array requires a local size_t range");
      resolve_native_storage_range(Arch::vulkan, array.pointer, array.bytes);
    }
  });
  registry.register_schema<DispatchDescription>(
      "vulkan_compute", "dispatch", {"pipeline"},
      [](const Json &p) {
        fields(p, {"expected_shader", "groups", "buffers", "array_count"});
        DispatchDescription d;
        d.expected = digest(p["expected_shader"]);
        if (!p["groups"].is_arr() || p["groups"].arr.inner.size() != 3 ||
            !p["buffers"].is_arr())
          throw std::invalid_argument(
              "Compute requires three group counts and buffer descriptors");
        for (size_t i = 0; i < 3; ++i)
          d.groups[i] = u32(p["groups"].arr.inner[i]);
        const auto count = u32(p["array_count"]);
        std::set<std::pair<uint32_t, uint32_t>> used;
        for (const auto &v : p["buffers"].arr.inner) {
          fields(v, {"set", "binding", "array", "storage"});
          DispatchDescription::Buffer b{u32(v["set"]), u32(v["binding"]),
                                        u32(v["array"]), optional_bool(v, "storage")};
          if (b.set > INT32_MAX || b.binding > INT32_MAX || b.array >= count ||
              !used.emplace(b.set, b.binding).second)
            throw std::invalid_argument(
                "Compute descriptor is duplicated or outside its binding "
                "range");
          d.buffers.push_back(b);
        }
        return [d, count](const NativeOperationBindings &b) mutable {
          auto result = d;
          result.pipeline = b.at("pipeline").resource<PipelineAccess>();
          for (uint32_t i = 0; i < count; ++i)
            result.arrays.push_back(b.at("array_" + std::to_string(i)).array());
          return result;
        };
      },
      {}, {}, {{"array_count", {"array"}}});
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<PipelineDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &c, const PipelineDescription &d) {
        return std::make_shared<PreparedPipeline>(c, d);
      });
  registry.register_context_factory<DispatchDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &c, const DispatchDescription &d) {
        return std::make_shared<PreparedDispatch>(c, d);
      });
#endif
}
}  // namespace taichi::lang
