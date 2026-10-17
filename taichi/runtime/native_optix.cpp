#include "taichi/runtime/native_operation_registry.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include "taichi/common/operation_json.h"
#include "taichi/optix/forge_optix_runtime_provider.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_optix_access.h"
#include "taichi/runtime/native_image.h"
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/provider_completion_resource.h"
#endif

namespace taichi::lang {
namespace {
using native_optix::calls_safe;
using native_optix::check;
struct OptixContextDescription {
  static constexpr const char *kind = "optix_context";
  static constexpr uint32_t version = 1;
  bool validation{false};
};

using OptixContextResource = NativeOptixContext;

struct OptixSceneDescription {
  static constexpr const char *kind = "optix_triangle_scene";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixContextResource> context;
  NativeArrayBinding vertices, indices;
  uint32_t vertex_count{0}, triangle_count{0};
  bool allow_update{false};
};

struct OptixGasDescription : OptixSceneDescription {
  static constexpr const char *kind = "optix_triangle_gas";
};

class OptixSceneResource;
struct OptixInstancesDescription {
  static constexpr const char *kind = "optix_instance_scene";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixContextResource> context;
  std::vector<std::shared_ptr<OptixSceneResource>> gas;
  NativeArrayBinding transforms;
  std::vector<uint32_t> custom_indices, visibility_masks, opaque, sbt_offsets;
  bool allow_update{false};
};
struct OptixSbtInstancesDescription : OptixInstancesDescription {
  static constexpr uint32_t version = 2;
};

class OptixSceneResource : public NativeOptixScene {
 public:
  template <typename Description>
  explicit OptixSceneResource(Description description)
      : description_(std::move(description)) {
  }
  bool is_gas() const {
    return std::holds_alternative<OptixGasDescription>(description_);
  }
  bool is_instances() const {
    return std::holds_alternative<OptixInstancesDescription>(description_);
  }
  const OptixSceneDescription &geometry() const {
    if (is_gas())
      return std::get<OptixGasDescription>(description_);
    return std::get<OptixSceneDescription>(description_);
  }
  const OptixInstancesDescription &instances() const {
    return std::get<OptixInstancesDescription>(description_);
  }
  std::shared_ptr<OptixContextResource> context() const override {
    return std::visit([](const auto &d) { return d.context; }, description_);
  }
  bool allow_update() const {
    return std::visit([](const auto &d) { return d.allow_update; }, description_);
  }
  uint32_t instance_count() const {
    return is_instances() ? uint32_t(instances().gas.size()) : 1;
  }
  std::vector<uint32_t> primitive_counts() const {
    if (!is_instances())
      return {geometry().triangle_count};
    std::vector<uint32_t> result;
    result.reserve(instances().gas.size());
    for (const auto &gas : instances().gas)
      result.push_back(gas->geometry().triangle_count);
    return result;
  }
  uint64_t face_table_size() const {
    uint64_t total = 0;
    for (auto value : primitive_counts()) {
      if (value > std::numeric_limits<size_t>::max() / 4 - total)
        throw std::overflow_error("OptiX instance face table size overflows");
      total += value;
    }
    return total;
  }
  std::vector<const OptixSceneDescription *> geometry_inputs() const {
    if (!is_instances())
      return {&geometry()};
    std::vector<const OptixSceneDescription *> result;
    std::unordered_set<const OptixSceneResource *> seen;
    for (const auto &gas : instances().gas)
      if (seen.insert(gas.get()).second)
        result.push_back(&gas->geometry());
    return result;
  }

 private:
  const std::variant<OptixSceneDescription, OptixGasDescription,
                     OptixInstancesDescription> description_;
};

struct OptixBoundFaceRule {
  uint32_t mode{0};
  std::optional<NativeArrayBinding> primitives;
};

struct OptixBoundAlphaMask {
  NativeArrayBinding uvs;
  std::shared_ptr<NativeSampledImage> image;
  float cutoff{0.5f};
  uint32_t channel{0};
};

struct OptixQueryOperation {
  static constexpr const char *kind = "optix_query";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixSceneResource> scene;
  NativeArrayBinding rays, hits;
  std::optional<NativeArrayBinding> hit_indices, faces;
  uint32_t ray_count{0};
  bool occlusion{false}, any_hit{false}, front_only{false};
  std::vector<OptixBoundFaceRule> face_rules;
  std::vector<std::optional<OptixBoundAlphaMask>> alpha_masks;
};

// Version 1 freezes a flat table; version 2 binds live per-instance tables.
// Version 3 also binds managed alpha images. All retain the output ABI and
// ordered execution path.
struct OptixBoundFaceQueryOperation : OptixQueryOperation {
  static constexpr uint32_t version = 2;
};

struct OptixAlphaQueryOperation : OptixQueryOperation {
  static constexpr uint32_t version = 3;
};

struct OptixRefitOperation {
  static constexpr const char *kind = "optix_refit";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixSceneResource> scene;
  NativeArrayBinding vertices;
};

struct OptixInstanceRefitOperation {
  static constexpr const char *kind = "optix_instance_refit";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixSceneResource> scene;
  std::optional<NativeArrayBinding> transforms;
};

uint32_t count(const liong::json::JsonValue &p, const char *key) {
  const auto value = operation_json::integer(p[key]);
  if (!value || value > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("OptiX counts must fit the positive uint32 ABI");
  return uint32_t(value);
}

NativeStorageRange layout(Device &device,
                           const NativeArrayBinding &array,
                           uint64_t count,
                           uint32_t width,
                           bool integer,
                           bool transforms = false) {
  if (!count || array.pointer.device != &device ||
      (integer ? array.scalar_type != PrimitiveType::i32 &&
                     array.scalar_type != PrimitiveType::u32
               : array.scalar_type != PrimitiveType::f32) ||
      count > std::numeric_limits<size_t>::max() / width / 4 ||
      array.bytes != count * width * 4)
    throw std::invalid_argument("OptiX requires matching local compact arrays");
  NativeArrayLayoutRequirement required{{count}, {}, 4, 4, true};
  if (transforms && array.element_shape == std::vector<uint64_t>{3, 4}) {
    required.element_shape = {3, 4};
    required.record_bytes = 48;
  } else if (!transforms &&
             array.element_shape == std::vector<uint64_t>{width}) {
    required.element_shape = {width};
    required.record_bytes = uint64_t(width) * 4;
  } else if (transforms &&
             array.shape == std::vector<uint64_t>{count, 3, 4}) {
    required.minimum_shape = {count, 3, 4};
  } else if (width > 1 || array.shape.size() == 2) {
    required.minimum_shape.push_back(width);
  }
  // Match the JIT ray/transform record layouts; equal byte counts alone are
  // insufficient (flat, transposed and differently shaped tensors are rejected).
  validate_native_array_layout(array, required);
  const auto range = resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes);
  if (range.memory_offset % 4)
    throw std::invalid_argument("OptiX array address must be scalar-aligned");
  return range;
}

void validate_scene(Device &device, const OptixSceneDescription &d) {
  if (!d.context || &d.context->device() != &device ||
      d.context->arch() != Arch::cuda)
    throw std::invalid_argument("OptiX scene requires a local CUDA context");
  layout(device, d.vertices, d.vertex_count, 3, false);
  layout(device, d.indices, d.triangle_count, 3, true);
}

void validate_query(Device &device, const OptixQueryOperation &op) {
  if (!op.scene || &op.scene->device() != &device ||
      op.scene->arch() != Arch::cuda || op.scene->is_gas() ||
      bool(op.hit_indices) == op.occlusion || (op.front_only && op.faces) ||
      (!op.face_rules.empty() && (op.front_only || op.faces)) ||
      (op.occlusion && !op.any_hit))
    throw std::invalid_argument("OptiX query has incompatible ports or policy");
  std::vector<NativeStorageRange> reads{
      layout(device, op.rays, op.ray_count, 8, false)};
  std::vector<NativeStorageWrite> writes{{
      layout(device, op.hits, op.ray_count, op.occlusion ? 1 : 4, op.occlusion),
      {}}};
  if (op.hit_indices)
    writes.push_back({layout(device, *op.hit_indices, op.ray_count, 4, true), {}});
  if (op.faces)
    reads.push_back(layout(device, *op.faces,
                           op.scene->face_table_size(), 1, true));
  if (!op.face_rules.empty()) {
    if (op.face_rules.size() != op.scene->instance_count())
      throw std::invalid_argument("OptiX face rules must match the instance count");
    const auto counts = op.scene->primitive_counts();
    for (size_t i = 0; i < op.face_rules.size(); ++i) {
      const auto &rule = op.face_rules[i];
      if (rule.mode > 2 || bool(rule.primitives) != (rule.mode == 2))
        throw std::invalid_argument("OptiX face rule has incompatible table binding");
      if (rule.primitives)
        reads.push_back(layout(device, *rule.primitives, counts[i], 1, true));
    }
  }
  if (!op.alpha_masks.empty()) {
    if (op.alpha_masks.size() != op.scene->instance_count())
      throw std::invalid_argument("OptiX masks must match the instance count");
    for (size_t i = 0; i < op.alpha_masks.size(); ++i) {
      if (!op.alpha_masks[i]) continue;
      const auto &mask = *op.alpha_masks[i];
      if (!mask.image || &mask.image->device() != &device ||
          mask.image->arch() != Arch::cuda)
        throw std::invalid_argument("OptiX mask requires a local managed CUDA image");
      const auto &p = mask.image->parameters();
      if (p.dimension != ImageDimension::d2D || p.mip_levels != 1 ||
          (p.format != BufferFormat::r32f && p.format != BufferFormat::rgba32f) ||
          mask.channel >= (p.format == BufferFormat::r32f ? 1u : 4u) ||
          !std::isfinite(mask.cutoff) || mask.cutoff < 0 || mask.cutoff > 1)
        throw std::invalid_argument("OptiX mask requires a 2D r32f/rgba32f image and valid channel/cutoff");
      if (op.scene->is_instances() && op.scene->instances().opaque[i])
        throw std::invalid_argument("OptiX opaque instance cannot bind an alpha mask");
      const auto &geometry = op.scene->is_instances()
          ? op.scene->instances().gas[i]->geometry() : op.scene->geometry();
      reads.push_back(layout(device, mask.uvs, geometry.vertex_count, 2, false));
    }
  }
  // Scene inputs can remain public arguments; a query must not corrupt them
  // through a differently named output or an imported/subview alias.
  for (const auto *geometry : op.scene->geometry_inputs()) {
    reads.push_back(layout(device, geometry->vertices,
                           geometry->vertex_count, 3, false));
    reads.push_back(layout(device, geometry->indices,
                           geometry->triangle_count, 3, true));
  }
  if (op.scene->is_instances())
    reads.push_back(layout(device, op.scene->instances().transforms,
                           op.scene->instance_count(), 12, false, true));
  validate_native_disjoint_writes(reads, writes);
}

void validate_refit(Device &device, const OptixRefitOperation &op) {
  if (!op.scene || &op.scene->device() != &device ||
      op.scene->arch() != Arch::cuda || op.scene->is_instances() ||
      !op.scene->allow_update())
    throw std::invalid_argument("OptiX refit requires a local updateable scene");
  layout(device, op.vertices, op.scene->geometry().vertex_count, 3, false);
}

void validate_instances(Device &device, const OptixInstancesDescription &d) {
  if (!d.context || d.context->arch() != Arch::cuda ||
      &d.context->device() != &device || d.gas.empty() ||
      d.gas.size() > std::numeric_limits<uint32_t>::max() ||
      d.custom_indices.size() != d.gas.size() ||
      d.visibility_masks.size() != d.gas.size() || d.opaque.size() != d.gas.size() ||
      (!d.sbt_offsets.empty() && d.sbt_offsets.size() != d.gas.size()))
    throw std::invalid_argument("OptiX instance description is inconsistent");
  for (const auto &gas : d.gas)
    if (!gas || !gas->is_gas() || gas->context() != d.context ||
        &gas->device() != &device)
      throw std::invalid_argument("OptiX instances require same-context GAS resources");
  layout(device, d.transforms, d.gas.size(), 12, false, true);
}

void validate_instance_refit(Device &device, const OptixInstanceRefitOperation &op) {
  if (!op.scene || &op.scene->device() != &device ||
      op.scene->arch() != Arch::cuda || !op.scene->is_instances() ||
      !op.scene->allow_update())
    throw std::invalid_argument("OptiX instance refit requires an updateable IAS");
  if (op.transforms)
    layout(device, *op.transforms, op.scene->instance_count(), 12, false, true);
}

std::vector<uint32_t> instance_values(const liong::json::JsonValue &p,
                                       const char *key, uint32_t count,
                                       uint32_t maximum, uint32_t fallback,
                                       bool boolean = false) {
  const auto found = p.obj.inner.find(key);
  if (found == p.obj.inner.end())
    return std::vector<uint32_t>(count, fallback);
  if (!found->second.is_arr() || found->second.arr.inner.size() != count)
    throw std::invalid_argument("OptiX instance metadata must match instance_count");
  std::vector<uint32_t> result;
  result.reserve(count);
  for (const auto &value : found->second.arr.inner) {
    if (boolean && value.ty != liong::json::L_JSON_BOOLEAN)
      throw std::invalid_argument("OptiX opaque entries must be booleans");
    const auto word = boolean ? uint64_t(value.b) : operation_json::integer(value);
    if (word > maximum)
      throw std::invalid_argument("OptiX instance metadata exceeds the vendor bit width");
    result.push_back(uint32_t(word));
  }
  return result;
}

#ifdef TI_WITH_CUDA
const auto &definition = native_optix::provider_definition();

TiForgeOptixProviderApi execution_api(const NativeProviderRuntime &runtime) {
  const auto api = runtime.execution_api<TiForgeOptixExecutionApi>().provider;
  constexpr uint64_t required = TI_FORGE_OPTIX_FEATURE_TRIANGLE_GAS |
      TI_FORGE_OPTIX_FEATURE_SINGLE_INSTANCE_IAS | TI_FORGE_OPTIX_FEATURE_GAS_UPDATE |
      TI_FORGE_OPTIX_FEATURE_TYPED_HITS | TI_FORGE_OPTIX_FEATURE_ALPHA_MASK |
      TI_FORGE_OPTIX_FEATURE_RUNTIME_ORDERED_STREAM |
      TI_FORGE_OPTIX_FEATURE_COMPACT_OCCLUSION |
      TI_FORGE_OPTIX_FEATURE_SHARED_TRIANGLE_GAS |
      TI_FORGE_OPTIX_FEATURE_MULTI_INSTANCE_IAS |
      TI_FORGE_OPTIX_FEATURE_DEVICE_INSTANCE_TRANSFORM_UPDATE |
      TI_FORGE_OPTIX_FEATURE_INSTANCE_OPACITY |
      TI_FORGE_OPTIX_FEATURE_FACE_FILTER_TYPED |
      TI_FORGE_OPTIX_FEATURE_FACE_FILTER_OCCLUSION |
      TI_FORGE_OPTIX_FEATURE_FACE_FILTER_PER_PRIMITIVE;
  if (api.struct_size < sizeof(api) ||
      api.provider_abi_version != TI_FORGE_OPTIX_PROVIDER_ABI_VERSION ||
      (api.info.features & required) != required ||
      !api.create_context || !api.destroy_context || !api.create_triangle_scene ||
      !api.destroy_triangle_scene || !api.update_triangle_scene ||
      !api.get_scene_memory || !api.prepare_typed || !api.prepare_alpha ||
      !api.prepare_faces || !api.trace_typed || !api.trace_alpha ||
      !api.trace_faces || !api.trace_occlusion || !api.get_last_error ||
      !api.create_triangle_gas || !api.update_triangle_gas ||
      !api.get_triangle_gas_memory || !api.destroy_triangle_gas ||
      !api.create_instance_scene || !api.update_instance_scene ||
      !api.get_instance_scene_memory || !api.destroy_instance_scene ||
      !api.trace_instance_scene_typed || !api.trace_instance_alpha ||
      !api.trace_instance_faces || !api.trace_instance_occlusion)
    throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                              "OptiX domain execution table is incomplete");
  return api;
}

NativeOperationAvailability probe(NativeProviderSession &providers) {
  try {
    execution_api(*providers.runtime(definition));
    // The exact SDK ABI is available in the domain table. Do not report it as
    // a driver runtime release: that version is not exposed by this API.
    return {true, 0};
  } catch (const NativeProviderError &error) {
    if (error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE ||
        error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE)
      return {};
    throw;
  }
}

bool has_instance_sbt_api(const TiForgeOptixProviderApi &api) {
  return (api.info.features & TI_FORGE_OPTIX_FEATURE_INSTANCE_SBT_OFFSET) &&
         api.create_instance_scene_sbt && api.prepare_instance_sbt;
}

NativeOperationAvailability probe_instance_sbt(NativeProviderSession &providers) {
  if (!probe(providers).available) return {};
  return {has_instance_sbt_api(execution_api(*providers.runtime(definition))), 0};
}

uint64_t address(const NativeArrayBinding &array) {
  return resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes)
      .memory_offset;
}

// On a lost/finalized device, preserve the complete adapter/context lease.
// OptiX's common library lease alone cannot account for an abandoned context.
struct ContextLease {
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeOptixProviderApi api;
  TiForgeOptixFixedQueryApi queries{};
  TiForgeOptixContext handle{nullptr};
};

class CudaOptixContext final : public OptixContextResource {
 public:
  explicit CudaOptixContext(NativePreparationContext &context)
      : device_(context.device), fault(context.fault_domain) {
    lease = std::make_unique<ContextLease>();
    lease->runtime = context.providers.runtime(definition);
    lease->api = execution_api(*lease->runtime);
    if (lease->api.info.features & TI_FORGE_OPTIX_FEATURE_PREPARED_FIXED_QUERY) {
      const auto execution =
          lease->runtime->execution_api<TiForgeOptixExecutionApiV2>(2);
      lease->queries = execution.queries;
      if (lease->queries.struct_size != sizeof(lease->queries) ||
          lease->queries.revision != TI_FORGE_OPTIX_FIXED_QUERY_REVISION ||
          !lease->queries.prepare || !lease->queries.launch ||
          !lease->queries.destroy || !lease->queries.memory)
        throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                                  "OptiX fixed query table is incomplete");
    }
  }
  void initialize(bool validation) {
    TiForgeOptixContextDesc desc{};
    desc.struct_size = sizeof(desc);
    desc.cuda_context = reinterpret_cast<uint64_t>(
        CUDAContext::get_instance().get_context());
    desc.validation_mode = validation;
    desc.runtime_library_path = lease->runtime->info().library_path.c_str();
    check(lease->api, lease->api.create_context(&desc, &lease->handle),
          "create context");
    if (!lease->handle)
      throw std::runtime_error("OptiX returned a null context");
  }
  ~CudaOptixContext() override {
    if (!lease || !lease->handle)
      return;
    if (!calls_safe(fault)) {
      lease.release();
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      const auto result = lease->api.destroy_context(lease->handle);
      if (result != TI_FORGE_OPTIX_SUCCESS) {
        TI_WARN("OptiX context cleanup failed (result {}).", int(result));
        lease.release();
      }
    } catch (...) {
      lease.release();
    }
  }
  Arch arch() const override { return Arch::cuda; }
  Device &device() const override { return device_; }
  size_t workspace_bytes() const override { return sbt_bytes; }
  const TiForgeOptixProviderApi &provider_api() const override {
    return lease->api;
  }
  TiForgeOptixContext context_handle() const override { return lease->handle; }
  std::shared_ptr<RuntimeFaultDomain> fault_domain() const override { return fault; }
  std::shared_ptr<RuntimeFaultDomain> fault;
  std::unique_ptr<ContextLease> lease;
  size_t sbt_bytes{0};

 private:
  Device &device_;
};

class CudaOptixScene;
struct SceneLease {
  std::shared_ptr<CudaOptixContext> context;
  std::vector<std::shared_ptr<CudaOptixScene>> gas;
  DeviceAllocationUnique indices;
  TiForgeOptixTriangleScene handle{nullptr};
  TiForgeOptixTriangleSceneDesc desc{};
};

class CudaOptixScene final : public OptixSceneResource,
                              public CudaProviderCompletionResource {
 public:
  template <typename Description>
  explicit CudaOptixScene(Description description)
      : OptixSceneResource(std::move(description)),
        lease(std::make_unique<SceneLease>()) {
    lease->context =
        std::dynamic_pointer_cast<CudaOptixContext>(context());
    if (!lease->context)
      throw std::invalid_argument("OptiX requires its native context owner");
  }
  void initialize() {
    if (is_instances()) {
      initialize_instances();
      return;
    }
    const auto &d = geometry();
    auto &driver = CUDADriver::get_instance();
    // Cold topology validation and snapshot only. Later refits use these same
    // immutable indices; no per-frame host readback or topology probing.
    driver.context_synchronize();
    std::vector<uint32_t> indices(size_t(d.triangle_count) * 3);
    driver.memcpy_device_to_host(indices.data(),
                                 reinterpret_cast<void *>(address(d.indices)),
                                 d.indices.bytes);
    for (auto index : indices)
      if (index >= d.vertex_count)
        throw std::invalid_argument("OptiX triangle index exceeds vertex count");
    auto allocation = device().allocate_memory_unique({size_t(d.indices.bytes)});
    if (allocation.second != RhiResult::success)
      throw std::runtime_error("OptiX topology allocation failed");
    lease->indices = std::move(allocation.first);
    auto indices_address = reinterpret_cast<uint64_t>(
        static_cast<cuda::CudaDevice &>(device()).get_memory_addr(*lease->indices));
    driver.memcpy_host_to_device(reinterpret_cast<void *>(indices_address),
                                 indices.data(), d.indices.bytes);
    lease->desc = {sizeof(lease->desc), d.vertex_count, d.triangle_count,
                    d.allow_update, address(d.vertices), indices_address, 0};
    auto &api = lease->context->lease->api;
    auto create = is_gas() ? api.create_triangle_gas : api.create_triangle_scene;
    check(api, create(lease->context->lease->handle, &lease->desc, &lease->handle),
          "create triangle geometry");
    if (!lease->handle)
      throw std::runtime_error("OptiX returned a null scene");
    driver.stream_synchronize(nullptr);
    update_memory();
  }
  void initialize_instances() {
    const auto &d = instances();
    const auto &api = lease->context->lease->api;
    if (!d.sbt_offsets.empty() && !has_instance_sbt_api(api))
      throw std::invalid_argument("OptiX adapter lacks instance SBT offsets");
    auto &driver = CUDADriver::get_instance();
    driver.context_synchronize();
    std::vector<float> transforms(d.gas.size() * 12);
    driver.memcpy_device_to_host(transforms.data(),
                                 reinterpret_cast<void *>(address(d.transforms)),
                                 d.transforms.bytes);
    std::vector<TiForgeOptixInstanceDesc> descriptions(d.gas.size());
    lease->gas.reserve(d.gas.size());
    for (size_t i = 0; i < d.gas.size(); ++i) {
      auto gas = std::dynamic_pointer_cast<CudaOptixScene>(d.gas[i]);
      if (!gas || !gas->is_gas())
        throw std::invalid_argument("OptiX instance requires a native GAS owner");
      lease->gas.push_back(gas);
      auto &desc = descriptions[i];
      desc.struct_size = sizeof(desc);
      desc.reserved = d.opaque[i];
      desc.gas = gas->lease->handle;
      std::copy_n(transforms.data() + 12 * i, 12, desc.transform);
      desc.custom_index = d.custom_indices[i];
      desc.visibility_mask = d.visibility_masks[i];
    }
    TiForgeOptixInstanceSceneDesc desc{sizeof(desc), instance_count(),
                                      d.allow_update, 0, descriptions.data(), 0};
    // The shared adapter checks finite/invertible transforms and vendor ID
    // limits before constructing its IAS; no second transform policy here.
    // Its extended entry also queries the actual device SBT offset limit.
    check(api, d.sbt_offsets.empty()
                   ? api.create_instance_scene(lease->context->lease->handle,
                                                &desc, &lease->handle)
                   : api.create_instance_scene_sbt(lease->context->lease->handle,
                                                    &desc, d.sbt_offsets.data(),
                                                    &lease->handle),
          "create instance scene");
    if (!lease->handle)
      throw std::runtime_error("OptiX returned a null instance scene");
    driver.stream_synchronize(nullptr);
    update_memory();
  }
  ~CudaOptixScene() override {
    if (!lease)
      return;
    auto fault = lease->context->fault;
    if (!calls_safe(fault)) {
      lease.release();
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      if (lease->handle) {
        const auto &api = lease->context->lease->api;
        auto destroy = is_instances() ? api.destroy_instance_scene
                       : is_gas() ? api.destroy_triangle_gas : api.destroy_triangle_scene;
        const auto result = destroy(lease->handle);
        if (result != TI_FORGE_OPTIX_SUCCESS) {
          TI_WARN("OptiX scene cleanup failed (result {}).", int(result));
          lease.release();
          return;
        }
      }
      lease.reset();
    } catch (...) {
      lease.release();
    }
  }
  Arch arch() const override { return Arch::cuda; }
  Device &device() const override { return context()->device(); }
  size_t workspace_bytes() const override { return bytes_; }
  TiForgeOptixProgramScene program_scene() const override {
    if (is_gas())
      throw std::invalid_argument("OptiX program scene requires a triangle scene or IAS");
    return {lease->handle, uint32_t(is_instances())};
  }
  void update_memory() {
    TiForgeOptixSceneMemory memory{sizeof(memory)};
    const auto &api = lease->context->lease->api;
    auto query_memory = is_instances() ? api.get_instance_scene_memory
                        : is_gas() ? api.get_triangle_gas_memory : api.get_scene_memory;
    check(api, query_memory(lease->handle, &memory), "query scene memory");
    uint64_t bytes = is_instances() ? 0 : geometry().indices.bytes;
    for (auto value : {memory.gas_bytes, memory.ias_bytes,
                       memory.build_update_scratch_bytes, memory.instance_bytes,
                       memory.launch_params_bytes}) {
      if (value > std::numeric_limits<size_t>::max() - bytes)
        throw std::overflow_error("OptiX scene storage accounting overflows");
      bytes += value;
    }
    bytes_ = size_t(bytes);
    if (memory.shared_pipeline_sbt_bytes > std::numeric_limits<size_t>::max())
      throw std::overflow_error("OptiX SBT storage accounting overflows");
    lease->context->sbt_bytes = size_t(memory.shared_pipeline_sbt_bytes);
  }
  std::unique_ptr<SceneLease> lease;

 private:
  size_t bytes_{0};
};

// Keep the adapter query, its device tables and complete scene/context lease
// together, including when backend failure makes destruction unsafe.
struct OptixQueryLease {
  std::shared_ptr<CudaOptixScene> scene;
  std::vector<std::shared_ptr<NativeSampledImage>> images;
  DeviceAllocationUnique workspace;
  TiForgeOptixPreparedQuery prepared{nullptr};
};

class CudaOptixQuery final : public NativeOperation {
 public:
  explicit CudaOptixQuery(const OptixQueryOperation &op)
      : scene_(std::dynamic_pointer_cast<CudaOptixScene>(op.scene)),
        op_(op) {
    if (!scene_)
      throw std::invalid_argument("OptiX query requires a native scene");
    lease_ = std::make_unique<OptixQueryLease>();
    lease_->scene = scene_;
    for (const auto &mask : op_.alpha_masks)
      if (mask) lease_->images.push_back(mask->image);
  }
  void initialize() {
    auto &context = *scene_->lease->context;
    const auto &api = context.lease->api;
    const auto &queries = context.lease->queries;
    const bool alpha = !lease_->images.empty();
    const bool use_prepared = queries.prepare && !alpha;
    faces_ = op_.front_only || bool(op_.faces) ||
             std::any_of(op_.face_rules.begin(), op_.face_rules.end(),
                         [](const auto &rule) { return rule.mode != 0; });
    if (faces_ && alpha && !(api.info.features & TI_FORGE_OPTIX_FEATURE_FACE_FILTER_ALPHA))
      throw std::invalid_argument("OptiX adapter does not support face and alpha filtering together");
    extended_ = faces_ || op_.occlusion || op_.any_hit || alpha;
    typed_trace_ = scene_->is_instances() ? api.trace_instance_scene_typed : api.trace_typed;
    extended_trace_ = scene_->is_instances()
                          ? (op_.occlusion ? api.trace_instance_occlusion : api.trace_instance_alpha)
                          : (op_.occlusion ? api.trace_occlusion : api.trace_alpha);
    face_trace_ = scene_->is_instances() ? api.trace_instance_faces : api.trace_faces;
    if (!use_prepared)
      check(api, (faces_ ? api.prepare_faces
                        : extended_ ? api.prepare_alpha : api.prepare_typed)(
                   context.lease->handle),
          "prepare query pipeline");
    if (!use_prepared && scene_->is_instances() &&
        std::any_of(scene_->instances().sbt_offsets.begin(),
                    scene_->instances().sbt_offsets.end(),
                    [](uint32_t offset) { return offset != 0; })) {
      uint64_t bytes = 0;
      check(api, api.prepare_instance_sbt(scene_->lease->handle,
                                         faces_ ? 4 : extended_ ? 2 : 1, &bytes),
            "prepare fixed instance SBT");
    }
    face_records_offset_ = use_prepared ? 0 : 64;
    const bool legacy_opaque_first =
        !use_prepared && op_.any_hit && !op_.occlusion && !faces_;
    const bool mask_table = alpha || legacy_opaque_first;
    size_t mask_records_offset = 0;
    if (faces_ || (extended_ && !use_prepared)) {
      if (faces_ && scene_->instance_count() >
                        (std::numeric_limits<size_t>::max() - face_records_offset_) /
                            sizeof(TiForgeOptixFaceRule))
        throw std::overflow_error("OptiX instance face records size overflows");
      const size_t table_offset = face_records_offset_ +
          size_t(scene_->instance_count()) * sizeof(TiForgeOptixFaceRule);
      bytes_ = faces_ ? table_offset : 48;
      if (op_.faces) {
        if (op_.faces->bytes > std::numeric_limits<size_t>::max() - bytes_)
          throw std::overflow_error("OptiX face workspace size overflows");
        bytes_ += op_.faces->bytes;
      }
      if (mask_table) {
        mask_records_offset = bytes_;
        if (scene_->instance_count() >
            (std::numeric_limits<size_t>::max() - bytes_) / sizeof(TiForgeOptixAlphaMask))
          throw std::overflow_error("OptiX mask table size overflows");
        bytes_ += size_t(scene_->instance_count()) * sizeof(TiForgeOptixAlphaMask);
      }
      auto allocation = device().allocate_memory_unique({bytes_});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("OptiX query workspace allocation failed");
      lease_->workspace = std::move(allocation.first);
      workspace_address_ = reinterpret_cast<uint64_t>(
          static_cast<cuda::CudaDevice &>(device()).get_memory_addr(*lease_->workspace));
      if (mask_table) {
        // Upload immutable mask descriptions once. Older adapters also require
        // zero-texture records for fully opaque first-hit traversal.
        std::vector<TiForgeOptixAlphaMask> masks(scene_->instance_count());
        for (size_t i = 0; i < op_.alpha_masks.size(); ++i) {
          if (!op_.alpha_masks[i]) continue;
          const auto &mask = *op_.alpha_masks[i];
          const auto &geometry = scene_->is_instances()
              ? *scene_->lease->gas[i] : *scene_;
          // Use the same frozen topology as GAS construction, not a public
          // index array that the consumer may subsequently update.
          masks[i] = {address(mask.uvs), geometry.lease->desc.indices,
                      mask.image->cuda_sampled_object(), mask.cutoff, mask.channel};
        }
        CUDADriver::get_instance().memcpy_host_to_device(
            reinterpret_cast<void *>(workspace_address_ + mask_records_offset), masks.data(),
            masks.size() * sizeof(TiForgeOptixAlphaMask));
      }
      if (faces_) {
        std::vector<TiForgeOptixFaceRule> records(scene_->instance_count(), {0, 1, 0});
        auto &driver = CUDADriver::get_instance();
        if (!op_.face_rules.empty()) {
          // Keep live device bindings. No host validation/copy of table values:
          // an earlier kernel may produce them on every execution.
          for (size_t i = 0; i < records.size(); ++i) {
            const auto &rule = op_.face_rules[i];
            records[i] = {rule.primitives ? address(*rule.primitives) : 0,
                          rule.mode, 0};
          }
        } else if (op_.faces) {
          std::vector<uint32_t> rules(size_t(scene_->face_table_size()));
          driver.context_synchronize();
          driver.memcpy_device_to_host(rules.data(),
                                       reinterpret_cast<void *>(address(*op_.faces)),
                                       op_.faces->bytes);
          for (auto value : rules)
            if (value > 1)
              throw std::invalid_argument("OptiX face rules must be 0 or 1");
          uint64_t pointer = workspace_address_ + table_offset;
          driver.memcpy_host_to_device(reinterpret_cast<void *>(pointer),
                                       rules.data(), op_.faces->bytes);
          const auto counts = scene_->primitive_counts();
          for (size_t i = 0; i < records.size(); ++i) {
            records[i] = {pointer, 2, 0};
            pointer += uint64_t(counts[i]) * 4;
          }
        }
        driver.memcpy_host_to_device(
            reinterpret_cast<void *>(workspace_address_ + face_records_offset_), records.data(),
            records.size() * sizeof(TiForgeOptixFaceRule));
      }
    }
    query_ = {uint32_t(faces_ ? sizeof(TiForgeOptixFaceTraceDesc)
                               : sizeof(TiForgeOptixAlphaTraceDesc)),
              op_.ray_count, address(op_.rays), address(op_.hits),
              op_.hit_indices ? address(*op_.hit_indices) : 0,
              0, 0, workspace_address_, 0, uint32_t(op_.any_hit)};
    if (mask_table) {
      query_.masks = workspace_address_ + mask_records_offset;
      query_.mask_count = scene_->instance_count();
    }
    if (use_prepared) {
      const TiForgeOptixFixedQueryDesc desc{
          sizeof(desc), uint32_t(scene_->is_instances()), scene_->lease->handle,
          op_.ray_count, uint32_t(op_.occlusion),
          uint32_t(op_.occlusion || op_.any_hit),
          faces_ ? scene_->instance_count() : 0,
          query_.rays, query_.hits, query_.hit_indices,
          faces_ ? workspace_address_ + face_records_offset_ : 0};
      check(api, queries.prepare(context.lease->handle, &desc, &lease_->prepared),
            "prepare fixed query");
      if (!lease_->prepared)
        throw std::runtime_error("OptiX returned a null prepared query");
      TiForgeOptixFixedQueryMemory memory{sizeof(memory)};
      check(api, queries.memory(lease_->prepared, &memory), "query fixed storage");
      if (memory.device_bytes > std::numeric_limits<size_t>::max() - bytes_)
        throw std::overflow_error("OptiX fixed query storage accounting overflows");
      bytes_ += size_t(memory.device_bytes);
    }
    scene_->update_memory();
  }
  ~CudaOptixQuery() override {
    if (!lease_ || (!lease_->workspace && !lease_->prepared))
      return;
    if (!calls_safe(scene_->lease->context->fault)) {
      lease_.release();
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      if (lease_->prepared) {
        const auto &queries = scene_->lease->context->lease->queries;
        const auto result = queries.destroy(lease_->prepared);
        if (result != TI_FORGE_OPTIX_SUCCESS) {
          TI_WARN("OptiX fixed query cleanup failed (result {}).", int(result));
          lease_.release();
          return;
        }
      }
      lease_.reset();
    } catch (...) {
      lease_.release();
    }
  }
  Arch arch() const override { return Arch::cuda; }
  Device &device() const override { return scene_->device(); }
  size_t workspace_bytes() const override { return bytes_; }
  void validate_recording(NativeRecordingState &state) const override {
    for (const auto &image : lease_->images) image->validate_sampling(state);
  }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return scene_;
  }
  void run_cuda_ordered(void *stream) override {
    const auto &api = scene_->lease->context->lease->api;
    if (lease_->prepared) {
      check(api, scene_->lease->context->lease->queries.launch(
                     lease_->prepared, reinterpret_cast<uint64_t>(stream)),
            "launch fixed query");
      return;
    }
    query_.cuda_stream = reinterpret_cast<uint64_t>(stream);
    if (faces_) {
      TiForgeOptixFaceTraceDesc desc{query_, workspace_address_ + face_records_offset_, scene_->instance_count(),
                                    uint32_t(op_.occlusion)};
      check(api, face_trace_(scene_->lease->handle, &desc), "trace faces");
    } else if (extended_) {
      check(api, extended_trace_(scene_->lease->handle, &query_),
            "trace extended query");
    } else {
      TiForgeOptixTypedTraceDesc desc{sizeof(desc), query_.ray_count, query_.rays,
                                      query_.hits, query_.hit_indices,
                                      query_.cuda_stream};
      check(api, typed_trace_(scene_->lease->handle, &desc), "trace typed");
    }
  }

 private:
  std::shared_ptr<CudaOptixScene> scene_;
  OptixQueryOperation op_;
  std::unique_ptr<OptixQueryLease> lease_;
  size_t face_records_offset_{0};
  uint64_t workspace_address_{0};
  size_t bytes_{0};
  bool faces_{false}, extended_{false};
  TiForgeOptixAlphaTraceDesc query_{};
  TiForgeOptixTraceTypedFn typed_trace_{nullptr};
  TiForgeOptixTraceAlphaFn extended_trace_{nullptr};
  TiForgeOptixResult (*face_trace_)(void *, const TiForgeOptixFaceTraceDesc *){nullptr};
};

class CudaOptixRefit final : public NativeOperation {
 public:
  explicit CudaOptixRefit(const OptixRefitOperation &op)
      : scene_(std::dynamic_pointer_cast<CudaOptixScene>(op.scene)),
        vertices_(address(op.vertices)) {
    if (!scene_)
      throw std::invalid_argument("OptiX refit requires a native scene");
  }
  explicit CudaOptixRefit(const OptixInstanceRefitOperation &op)
      : scene_(std::dynamic_pointer_cast<CudaOptixScene>(op.scene)),
        vertices_(op.transforms ? address(*op.transforms) : 0) {
    if (!scene_)
      throw std::invalid_argument("OptiX instance refit requires a native scene");
  }
  Arch arch() const override { return Arch::cuda; }
  Device &device() const override { return scene_->device(); }
  size_t workspace_bytes() const override { return 0; }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return scene_;
  }
  void run_cuda_ordered(void *stream) override {
    const auto &api = scene_->lease->context->lease->api;
    if (scene_->is_instances()) {
      TiForgeOptixInstanceUpdateDesc desc{sizeof(desc), scene_->instance_count(),
                                          vertices_, reinterpret_cast<uint64_t>(stream)};
      check(api, api.update_instance_scene(scene_->lease->handle, &desc), "refit instances");
      return;
    }
    auto desc = scene_->lease->desc;
    desc.vertices = vertices_;
    desc.cuda_stream = reinterpret_cast<uint64_t>(stream);
    auto update = scene_->is_gas() ? api.update_triangle_gas : api.update_triangle_scene;
    check(api, update(scene_->lease->handle, &desc), "refit");
  }

 private:
  std::shared_ptr<CudaOptixScene> scene_;
  uint64_t vertices_;
};

template <typename Owner, typename... Args>
std::shared_ptr<Owner> prepare(NativePreparationContext &context,
                               Args &&...args) {
  if (!context.fault_domain ||
      !dynamic_cast<cuda::CudaDevice *>(&context.device))
    throw std::invalid_argument("OptiX requires a CUDA owner and fault domain");
  auto &cuda = CUDAContext::get_instance();
  auto lock = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  // Construct ownership before any initialization that may submit GPU work.
  auto owner = std::make_shared<Owner>(std::forward<Args>(args)...);
  try {
    owner->initialize();
    return owner;
  } catch (const BackendRuntimeError &error) {
    context.fault_domain->report_backend_error(error, 0);
    throw;
  }
}
#endif

template <typename Description>
void register_query(NativeOperationRegistry &registry) {
  registry.define<Description>([](Device &device, const Description &op) {
    if constexpr (Description::version >= 2) {
      if (op.face_rules.empty())
        throw std::invalid_argument("OptiX query requires per-instance face rules");
    }
    validate_query(device, op);
  });
  std::vector<std::string> optional_ports{"hit_indices"};
  std::vector<NativeOperationRegistry::BindingGroup> groups;
  if constexpr (Description::version >= 2)
    groups.push_back({"face_table_count", {"faces"}});
  else
    optional_ports.push_back("faces");
  if constexpr (Description::version == 3)
    groups.push_back({"mask_count", {"uvs", "image"}});
  registry.register_schema<Description>(
      "optix", "query", {"scene", "rays", "hits"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<Description> {
        using namespace operation_json;
        if constexpr (Description::version == 3)
          fields(p, {"ray_count", "occlusion", "any_hit", "face_rules",
                     "face_table_count", "alpha_masks", "mask_count"});
        else if constexpr (Description::version == 2)
          fields(p, {"ray_count", "occlusion", "any_hit", "face_rules", "face_table_count"});
        else
          fields(p, {"ray_count", "occlusion", "any_hit", "front_only"});
        const auto rays = count(p, "ray_count");
        const bool occlusion = optional_bool(p, "occlusion"),
                   any_hit = optional_bool(p, "any_hit", occlusion),
                   front_only = optional_bool(p, "front_only");
        if (occlusion && !any_hit)
          throw std::invalid_argument("OptiX occlusion requires any-hit acceptance");
        std::vector<uint32_t> modes;
        if constexpr (Description::version >= 2) {
          const auto &rules = p["face_rules"];
          if (!rules.is_arr() || rules.arr.inner.empty() ||
              rules.arr.inner.size() > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("OptiX face rules require a nonempty instance array");
          modes.reserve(rules.arr.inner.size());
          uint64_t tables = 0;
          for (const auto &rule : rules.arr.inner) {
            const auto value = name(rule);
            if (value == "two_sided")
              modes.push_back(0);
            else if (value == "front_only")
              modes.push_back(1);
            else if (value == "table") {
              modes.push_back(2);
              ++tables;
            } else {
              throw std::invalid_argument("OptiX face rule must be two_sided, front_only or table");
            }
          }
          if (integer(p["face_table_count"]) != tables)
            throw std::invalid_argument("OptiX face table count does not match face rules");
        }
        std::vector<std::optional<std::pair<float, uint32_t>>> masks;
        if constexpr (Description::version == 3) {
          const auto &entries = p["alpha_masks"];
          if (!entries.is_arr() || entries.arr.inner.size() != modes.size())
            throw std::invalid_argument("OptiX alpha masks must match face-rule instances");
          uint64_t active = 0;
          for (const auto &entry : entries.arr.inner) {
            if (entry.is_null()) {
              masks.push_back(std::nullopt);
              continue;
            }
            fields(entry, {"cutoff", "channel"});
            const float cutoff = optional_f32(entry, "cutoff", 0.5f);
            const auto channel = optional_integer(entry, "channel").value_or(0);
            if (cutoff < 0 || cutoff > 1 || channel > 3)
              throw std::invalid_argument("OptiX alpha cutoff/channel is out of range");
            masks.push_back(std::make_pair(cutoff, uint32_t(channel)));
            ++active;
          }
          if (integer(p["mask_count"]) != active)
            throw std::invalid_argument("OptiX mask count does not match active masks");
        }
        return [=](const NativeOperationBindings &b) {
          Description op;
          op.scene = b.at("scene").resource<OptixSceneResource>();
          op.rays = b.at("rays").array();
          op.hits = b.at("hits").array();
          op.ray_count = rays;
          op.occlusion = occlusion;
          op.any_hit = any_hit;
          op.front_only = front_only;
          if (b.count("hit_indices")) op.hit_indices = b.at("hit_indices").array();
          if constexpr (Description::version >= 2) {
            size_t table = 0;
            op.face_rules.reserve(modes.size());
            for (auto mode : modes) {
              OptixBoundFaceRule rule{mode};
              if (mode == 2)
                rule.primitives = b.at("faces_" + std::to_string(table++)).array();
              op.face_rules.push_back(std::move(rule));
            }
          } else if (b.count("faces")) {
            op.faces = b.at("faces").array();
          }
          if constexpr (Description::version == 3) {
            size_t binding = 0;
            for (const auto &mask : masks) {
              if (!mask) {
                op.alpha_masks.push_back(std::nullopt);
                continue;
              }
              const auto suffix = "_" + std::to_string(binding++);
              op.alpha_masks.push_back(OptixBoundAlphaMask{
                  b.at("uvs" + suffix).array(),
                  b.at("image" + suffix).resource<NativeSampledImage>(),
                  mask->first, mask->second});
            }
          }
          return op;
        };
      }, {}, optional_ports, groups);
#ifdef TI_WITH_CUDA
  registry.register_context_factory<Description>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const Description &op) {
        return prepare<CudaOptixQuery>(c, op);
      });
#endif
}

template <typename Description>
void register_geometry(NativeOperationRegistry &registry, const char *operation) {
  registry.define<Description>(
      [](Device &device, const Description &d) { validate_scene(device, d); },
      NativeObjectKind::resource);
  registry.register_schema<Description>(
      "optix", operation, {"context", "vertices", "indices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<Description> {
        operation_json::fields(p, {"vertex_count", "triangle_count", "allow_update"});
        const auto vertices = count(p, "vertex_count"), triangles = count(p, "triangle_count");
        const bool update = operation_json::optional_bool(p, "allow_update");
        return [=](const NativeOperationBindings &b) {
          Description d;
          d.context = b.at("context").resource<OptixContextResource>();
          d.vertices = b.at("vertices").array();
          d.indices = b.at("indices").array();
          d.vertex_count = vertices;
          d.triangle_count = triangles;
          d.allow_update = update;
          return d;
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<Description>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const Description &d) {
        return prepare<CudaOptixScene>(c, d);
      });
#endif
}
template <typename Description>
void register_instances(NativeOperationRegistry &registry) {
  registry.define<Description>(validate_instances, NativeObjectKind::resource);
  registry.register_schema<Description>(
      "optix", "instance_scene", {"context", "transforms"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<Description> {
        using namespace operation_json;
        if constexpr (Description::version == 2)
          fields(p, {"instance_count", "custom_indices", "visibility_masks", "opaque", "allow_update", "sbt_offsets"});
        else
          fields(p, {"instance_count", "custom_indices", "visibility_masks", "opaque", "allow_update"});
        const auto size = count(p, "instance_count");
        const auto ids = instance_values(p, "custom_indices", size, 0xffffffu, 0);
        const auto masks = instance_values(p, "visibility_masks", size, 0xffu, 0xffu);
        const auto opaque = instance_values(p, "opaque", size, 1, 0, true);
        std::vector<uint32_t> offsets;
        if constexpr (Description::version == 2)
          offsets = instance_values(p, "sbt_offsets", size, UINT32_MAX, 0);
        const bool update = optional_bool(p, "allow_update");
        return [=](const NativeOperationBindings &b) {
          Description d;
          d.context = b.at("context").resource<OptixContextResource>();
          d.transforms = b.at("transforms").array();
          d.custom_indices = ids;
          d.visibility_masks = masks;
          d.opaque = opaque;
          d.sbt_offsets = offsets;
          d.allow_update = update;
          d.gas.reserve(size);
          for (uint32_t i = 0; i < size; ++i)
            d.gas.push_back(b.at("gas_" + std::to_string(i)).resource<OptixSceneResource>());
          return d;
        };
      }, {}, {}, {{"instance_count", {"gas"}}});
#ifdef TI_WITH_CUDA
  NativeOperationRegistry::ContextProbe availability;
  if constexpr (Description::version == 2) availability = probe_instance_sbt;
  registry.register_context_factory<Description>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const Description &d) {
        return prepare<CudaOptixScene>(c, static_cast<const OptixInstancesDescription &>(d));
      }, availability);
#endif
}
}  // namespace

void register_optix_operations(NativeOperationRegistry &registry) {
  registry.define<OptixContextDescription>(
      [](Device &, const OptixContextDescription &) {}, NativeObjectKind::resource);
  registry.register_schema<OptixContextDescription>(
      "optix", "context", {}, [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<OptixContextDescription> {
        operation_json::fields(p, {"validation"});
        const bool validation = operation_json::optional_bool(p, "validation");
        return [validation](const NativeOperationBindings &) {
          return OptixContextDescription{validation};
        };
      });
  register_geometry<OptixSceneDescription>(registry, "triangle_scene");
  register_geometry<OptixGasDescription>(registry, "triangle_gas");
  register_instances<OptixInstancesDescription>(registry);
  register_instances<OptixSbtInstancesDescription>(registry);
  register_query<OptixQueryOperation>(registry);
  register_query<OptixBoundFaceQueryOperation>(registry);
  register_query<OptixAlphaQueryOperation>(registry);
  registry.define<OptixRefitOperation>(validate_refit);
  registry.register_schema<OptixRefitOperation>(
      "optix", "refit", {"scene", "vertices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<OptixRefitOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return OptixRefitOperation{b.at("scene").resource<OptixSceneResource>(),
                                     b.at("vertices").array()};
        };
      });
  registry.define<OptixInstanceRefitOperation>(validate_instance_refit);
  registry.register_schema<OptixInstanceRefitOperation>(
      "optix", "instance_refit", {"scene"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<OptixInstanceRefitOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          OptixInstanceRefitOperation op{b.at("scene").resource<OptixSceneResource>()};
          if (b.count("transforms"))
            op.transforms = b.at("transforms").array();
          return op;
        };
      }, {}, {"transforms"});
#ifdef TI_WITH_CUDA
  registry.register_context_factory<OptixContextDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &context, const OptixContextDescription &d) {
        if (!context.fault_domain || !dynamic_cast<cuda::CudaDevice *>(&context.device))
          throw std::invalid_argument("OptiX requires a CUDA owner and fault domain");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        auto result = std::make_shared<CudaOptixContext>(context);
        try {
          result->initialize(d.validation);
          return result;
        } catch (const BackendRuntimeError &error) {
          context.fault_domain->report_backend_error(error, 0);
          throw;
        }
      }, probe);
  registry.register_context_factory<OptixRefitOperation>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &, const OptixRefitOperation &op) {
        return std::make_shared<CudaOptixRefit>(op);
      });
  registry.register_context_factory<OptixInstanceRefitOperation>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &, const OptixInstanceRefitOperation &op) {
        return std::make_shared<CudaOptixRefit>(op);
      });
#endif
}
}  // namespace taichi::lang
