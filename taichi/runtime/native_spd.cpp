#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/native_image.h"
#include "taichi/runtime/vulkan_spd.h"
#include "taichi/common/operation_json.h"
#include "picosha2.h"
#include <cstring>
#ifdef TI_WITH_VULKAN
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {
using Json = liong::json::JsonValue;
struct SpdDescription {
  static constexpr const char *kind = "spd_plan";
  static constexpr uint32_t version = 1;
  std::shared_ptr<NativeSampledImage> source, output;
  std::shared_ptr<NativeHostBlobResource> shader;
  std::string reduction, expected;
};

std::string expected_contract(const Json &p) {
  const auto found = p.obj.inner.find("expected_plan");
  if (found == p.obj.inner.end())
    return {};
  const auto value = operation_json::name(found->second);
  if (value.size() != 64 ||
      value.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument("Expected SPD plan must be a SHA256 contract");
  return value;
}
void match_contract(const std::string &expected, const std::string &actual) {
  if (!expected.empty() && expected != actual)
    throw std::invalid_argument("SPD code or image layout contract mismatch");
}
void local(Device &device, const std::shared_ptr<NativeResource> &resource) {
  if (!resource || resource->arch() != Arch::vulkan ||
      &resource->device() != &device)
    throw std::invalid_argument(
        "SPD requires resources on the same Vulkan device");
}
std::string format(BufferFormat f) {
  switch (f) {
    case BufferFormat::r32f:
      return "r32f";
    case BufferFormat::rgba8:
      return "rgba8";
    case BufferFormat::rgba32f:
      return "rgba32f";
    default:
      throw std::invalid_argument("Unsupported SPD image format");
  }
}
// Length-delimited strings and little-endian u64 values, independent of JSON
// spelling, producer paths or live resource identities.
std::string contract(const SpdDescription &d) {
  picosha2::hash256_one_by_one hash;
  auto number = [&](uint64_t value) {
    uint8_t bytes[8];
    for (size_t i = 0; i < 8; ++i)
      bytes[i] = uint8_t(value >> (8 * i));
    hash.process(bytes, bytes + 8);
  };
  auto text = [&](const std::string &value) {
    number(value.size());
    hash.process(value.begin(), value.end());
  };
  text("forge.spd.plan.v1");
  text(d.reduction);
  for (const auto &image : {d.source, d.output}) {
    const auto &p = image->parameters();
    text(format(p.format));
    number(p.x);
    number(p.y);
    number(p.z);
    number(p.mip_levels);
  }
  const auto &bytes = d.shader->bytes();
  text(picosha2::hash256_hex_string(bytes.begin(), bytes.end()));
  hash.finish();
  return picosha2::get_hash_hex_string(hash);
}
void validate_plan(Device &device, const SpdDescription &d) {
  local(device, d.source);
  local(device, d.output);
  local(device, d.shader);
  const auto &bytes = d.shader->bytes();
  if (bytes.size() < 5 * sizeof(uint32_t) || bytes.size() % sizeof(uint32_t))
    throw std::invalid_argument("SPD requires complete SPIR-V words");
  uint32_t magic;
  std::memcpy(&magic, bytes.data(), sizeof(magic));
  if (magic != 0x07230203u)
    throw std::invalid_argument("SPD requires SPIR-V magic");
  match_contract(d.expected, contract(d));
}
class SpdPlanResource : public NativeResource {
 public:
  std::shared_ptr<NativeSampledImage> source, output;
  std::string identity;
  virtual void record(CommandList *) = 0;
};
struct SpdRun {
  static constexpr const char *kind = "spd_run";
  static constexpr uint32_t version = 1;
  std::shared_ptr<SpdPlanResource> plan;
  std::shared_ptr<NativeSampledImage> source, output;
  std::string expected;
};

#ifdef TI_WITH_VULKAN
class VulkanSpdPlan final : public SpdPlanResource {
 public:
  VulkanSpdPlan(NativePreparationContext &context, const SpdDescription &d)
      : device_(context.device),
        runtime_(*context.gfx_runtime),
        fault_(context.fault_domain) {
    source = d.source;
    output = d.output;
    identity = contract(d);
    std::vector<uint32_t> words(d.shader->bytes().size() / sizeof(uint32_t));
    std::memcpy(words.data(), d.shader->bytes().data(),
                d.shader->bytes().size());
    auto owners =
        std::make_shared<std::vector<std::shared_ptr<NativeSampledImage>>>(
            std::initializer_list<std::shared_ptr<NativeSampledImage>>{source,
                                                                       output});
    commands_ = prepare_vulkan_spd(
        static_cast<vulkan::VulkanDevice &>(device_), source->parameters(),
        source->gfx_sampled_allocation(), output->parameters(),
        output->gfx_sampled_allocation(), words, owners, false);
  }
  void initialize() {
    // Cold only. Retire the first counter clear before releasing construction
    // inputs or publishing a resource to another prepared execution.
    initialization_pending_ = true;
    runtime_.enqueue_compute_op_lambda(
        [initialize = commands_.initialize](Device *, CommandList *list) {
          initialize(list);
        },
        {});
    runtime_.synchronize();
    initialization_pending_ = false;
    commands_.initialize = {};
  }
  ~VulkanSpdPlan() override {
    if (initialization_pending_ && fault_->backend_calls_safe() &&
        fault_->state() != RuntimeLifecycleState::kFinalized) {
      try {
        runtime_.synchronize();
      } catch (...) {
      }
    }
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 4;
  }
  bool retains_input(const std::string &port) const override {
    return port != "shader";
  }
  std::optional<NativePreparationDependencies> preparation_dependencies()
      const override {
    if (initialization_pending_)
      return std::nullopt;
    // Counter clearing is either completed or an explicit initialization
    // command. Image dependencies retain their own preparation contracts.
    return NativePreparationDependencies{{}, {source.get(), output.get()}};
  }
  void record(CommandList *list) override {
    commands_.record_inline(list);
  }
  void record_initialization(CommandList *list) {
    commands_.initialize(list);
  }

 private:
  Device &device_;
  gfx::GfxRuntime &runtime_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  VulkanSpdCommands commands_;
  bool initialization_pending_{false};
};
class VulkanSpdInitialization final : public NativeOperation {
 public:
  std::optional<NativePreparationDependencies> preparation_dependencies()
      const override {
    return NativePreparationDependencies{{}, {plan_.get()}};
  }
  explicit VulkanSpdInitialization(std::shared_ptr<VulkanSpdPlan> plan)
      : plan_(std::move(plan)) {
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return plan_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    NativeVulkanRecordingRequirements requirements;
    requirements.buffers.emplace();
    requirements.image_uses.emplace();
    requirements.stateful_resources = {{plan_, true}};
    return requirements;
  }
  void record_vulkan(Device *, CommandList *list) override {
    plan_->record_initialization(list);
  }

 private:
  std::shared_ptr<VulkanSpdPlan> plan_;
};
class VulkanSpdRun final : public NativeOperation {
 public:
  std::optional<NativePreparationDependencies> preparation_dependencies()
      const override {
    return NativePreparationDependencies{{}, {plan_.get()}};
  }
  explicit VulkanSpdRun(std::shared_ptr<SpdPlanResource> plan)
      : plan_(std::move(plan)) {
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return plan_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void validate_recording(NativeRecordingState &state) const override {
    plan_->source->validate_sampling(state);
    // The plan always writes every allocated destination mip.
    state.initialized.insert(plan_->output.get());
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    NativeVulkanRecordingRequirements requirements;
    requirements.images = {{plan_->source->gfx_sampled_allocation(),
                            ImageLayout::shader_read_write},
                           {plan_->output->gfx_sampled_allocation(),
                            ImageLayout::shader_read_write}};
    requirements.buffers.emplace();
    // The source uses storage-image loads, so it cannot use the sampled-only
    // shared-image route. The counter is mutable even though every full run
    // restores it to zero; independent instances require their own plan.
    requirements.image_uses = {{plan_->source, false}, {plan_->output, false}};
    requirements.stateful_resources = {{plan_, true}};
    return requirements;
  }
  void record_vulkan(Device *, CommandList *list) override {
    plan_->record(list);
  }

 private:
  std::shared_ptr<SpdPlanResource> plan_;
};
#endif
}  // namespace

void register_spd_operations(NativeOperationRegistry &registry) {
  registry.define<SpdDescription>(validate_plan, NativeObjectKind::resource);
  registry.register_schema<SpdDescription>(
      "spd", "plan", {"source", "output", "shader"},
      [](const Json &p) -> NativeOperationRegistry::Binder<SpdDescription> {
        operation_json::fields(p, {"reduction", "expected_plan"});
        const auto reduction = operation_json::name(p["reduction"]);
        if (reduction != "mean" && reduction != "min" && reduction != "max")
          throw std::invalid_argument("SPD reduction must be mean, min or max");
        const auto expected = expected_contract(p);
        return [reduction, expected](const NativeOperationBindings &b) {
          return SpdDescription{
              b.at("source").resource<NativeSampledImage>(),
              b.at("output").resource<NativeSampledImage>(),
              b.at("shader").resource<NativeHostBlobResource>(), reduction,
              expected};
        };
      });
  registry.define<SpdRun>([](Device &device, const SpdRun &d) {
    local(device, d.plan);
    local(device, d.source);
    local(device, d.output);
    if (d.plan->source != d.source || d.plan->output != d.output)
      throw std::invalid_argument(
          "SPD run must bind the plan's fixed source and output images");
    match_contract(d.expected, d.plan->identity);
  });
  registry.register_schema<SpdRun>(
      "spd", "run", {"plan", "source", "output"},
      [](const Json &p) -> NativeOperationRegistry::Binder<SpdRun> {
        operation_json::fields(p, {"expected_plan"});
        const auto expected = expected_contract(p);
        return [expected](const NativeOperationBindings &b) {
          return SpdRun{b.at("plan").resource<SpdPlanResource>(),
                        b.at("source").resource<NativeSampledImage>(),
                        b.at("output").resource<NativeSampledImage>(),
                        expected};
        };
      });
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<SpdDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context, const SpdDescription &d) {
        if (!context.gfx_runtime || !context.fault_domain)
          throw std::invalid_argument(
              "SPD requires its owner's GFX runtime and retirement domain");
        auto plan = std::make_shared<VulkanSpdPlan>(context, d);
        if (context.defer_vulkan_initialization)
          context.defer_vulkan_initialization(
              std::make_shared<VulkanSpdInitialization>(plan));
        else
          plan->initialize();
        return plan;
      });
  registry.register_factory<SpdRun>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &, const SpdRun &d) {
        return std::make_shared<VulkanSpdRun>(d.plan);
      });
#endif
}
}  // namespace taichi::lang
