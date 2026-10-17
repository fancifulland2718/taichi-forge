#include "taichi/runtime/native_operation_registry.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include "taichi/common/operation_json.h"
#include "taichi/optix/forge_optix_runtime_provider.h"
#include "taichi/runtime/native_array_contract.h"
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/provider_completion_resource.h"
#endif

namespace taichi::lang {
namespace {
struct OptixContextDescription {
  static constexpr const char *kind = "optix_context";
  static constexpr uint32_t version = 1;
  bool validation{false};
};

class OptixContextResource : public NativeResource {
};

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
  std::vector<uint32_t> custom_indices, visibility_masks, opaque;
  bool allow_update{false};
};

class OptixSceneResource : public NativeResource {
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
  std::shared_ptr<OptixContextResource> context() const {
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

struct OptixQueryOperation {
  static constexpr const char *kind = "optix_query";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixSceneResource> scene;
  NativeArrayBinding rays, hits;
  std::optional<NativeArrayBinding> hit_indices, faces;
  uint32_t ray_count{0};
  bool occlusion{false}, any_hit{false}, front_only{false};
  std::vector<OptixBoundFaceRule> face_rules;
};

// Version 1 freezes a flat table; version 2 binds live per-instance tables.
// Both lower to the same query owner, output ABI and ordered execution path.
struct OptixBoundFaceQueryOperation : OptixQueryOperation {
  static constexpr uint32_t version = 2;
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
                           bool integer) {
  if (!count || array.pointer.device != &device ||
      (integer ? array.scalar_type != PrimitiveType::i32 &&
                     array.scalar_type != PrimitiveType::u32
               : array.scalar_type != PrimitiveType::f32) ||
      count > std::numeric_limits<size_t>::max() / width / 4 ||
      array.bytes != count * width * 4)
    throw std::invalid_argument("OptiX requires matching local compact arrays");
  std::vector<uint64_t> shape{count};
  if (width > 1)
    shape.push_back(width);
  validate_native_array_layout(array, {shape, {}, 4, 4, true});
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
                           op.scene->instance_count(), 12, false));
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
      d.visibility_masks.size() != d.gas.size() || d.opaque.size() != d.gas.size())
    throw std::invalid_argument("OptiX instance description is inconsistent");
  for (const auto &gas : d.gas)
    if (!gas || !gas->is_gas() || gas->context() != d.context ||
        &gas->device() != &device)
      throw std::invalid_argument("OptiX instances require same-context GAS resources");
  layout(device, d.transforms, d.gas.size(), 12, false);
}

void validate_instance_refit(Device &device, const OptixInstanceRefitOperation &op) {
  if (!op.scene || &op.scene->device() != &device ||
      op.scene->arch() != Arch::cuda || !op.scene->is_instances() ||
      !op.scene->allow_update())
    throw std::invalid_argument("OptiX instance refit requires an updateable IAS");
  if (op.transforms)
    layout(device, *op.transforms, op.scene->instance_count(), 12, false);
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
const NativeProviderDefinition definition{
    "optix", TI_FORGE_OPTIX_RUNTIME_PROVIDER_QUERY_SYMBOL,
    TI_FORGE_OPTIX_RUNTIME_VERSION_FAMILY};

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

uint64_t address(const NativeArrayBinding &array) {
  return resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes)
      .memory_offset;
}

void check(const TiForgeOptixProviderApi &api,
           TiForgeOptixResult result,
           const char *operation) {
  if (result == TI_FORGE_OPTIX_SUCCESS)
    return;
  const auto bytes = api.get_last_error(nullptr, 0);
  std::vector<char> message(std::max(size_t(1), bytes), 0);
  api.get_last_error(message.data(), message.size());
  message.back() = 0;
  throw std::runtime_error(std::string("OptiX ") + operation + " (" +
                           std::to_string(int(result)) + "): " + message.data());
}

bool calls_safe(const std::shared_ptr<RuntimeFaultDomain> &fault) {
  return fault && fault->backend_calls_safe() &&
         fault->state() != RuntimeLifecycleState::kFinalized;
}

// On a lost/finalized device, preserve the complete adapter/context lease.
// OptiX's common library lease alone cannot account for an abandoned context.
struct ContextLease {
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeOptixProviderApi api;
  TiForgeOptixContext handle{nullptr};
};

class CudaOptixContext final : public OptixContextResource {
 public:
  explicit CudaOptixContext(NativePreparationContext &context)
      : device_(context.device), fault(context.fault_domain) {
    lease = std::make_unique<ContextLease>();
    lease->runtime = context.providers.runtime(definition);
    lease->api = execution_api(*lease->runtime);
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
    const auto &api = lease->context->lease->api;
    // The shared adapter checks finite/invertible transforms and vendor ID
    // limits before constructing its IAS; no second transform policy here.
    check(api, api.create_instance_scene(lease->context->lease->handle,
                                         &desc, &lease->handle),
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

class CudaOptixQuery final : public NativeOperation {
 public:
  explicit CudaOptixQuery(const OptixQueryOperation &op)
      : scene_(std::dynamic_pointer_cast<CudaOptixScene>(op.scene)),
        op_(op) {
    if (!scene_)
      throw std::invalid_argument("OptiX query requires a native scene");
  }
  void initialize() {
    auto &context = *scene_->lease->context;
    const auto &api = context.lease->api;
    faces_ = op_.front_only || bool(op_.faces) ||
             std::any_of(op_.face_rules.begin(), op_.face_rules.end(),
                         [](const auto &rule) { return rule.mode != 0; });
    extended_ = faces_ || op_.occlusion || op_.any_hit;
    typed_trace_ = scene_->is_instances() ? api.trace_instance_scene_typed : api.trace_typed;
    extended_trace_ = scene_->is_instances()
                          ? (op_.occlusion ? api.trace_instance_occlusion : api.trace_instance_alpha)
                          : (op_.occlusion ? api.trace_occlusion : api.trace_alpha);
    face_trace_ = scene_->is_instances() ? api.trace_instance_faces : api.trace_faces;
    check(api, (faces_ ? api.prepare_faces
                        : extended_ ? api.prepare_alpha : api.prepare_typed)(
                   context.lease->handle),
          "prepare query pipeline");
    scene_->update_memory();
    if (extended_) {
      if (faces_ && scene_->instance_count() >
                        (std::numeric_limits<size_t>::max() - 64) / sizeof(TiForgeOptixFaceRule))
        throw std::overflow_error("OptiX instance face records size overflows");
      const size_t table_offset = 64 + size_t(scene_->instance_count()) * sizeof(TiForgeOptixFaceRule);
      bytes_ = faces_ ? table_offset : 48;
      if (op_.faces) {
        if (op_.faces->bytes > std::numeric_limits<size_t>::max() - bytes_)
          throw std::overflow_error("OptiX face workspace size overflows");
        bytes_ += op_.faces->bytes;
      }
      auto allocation = device().allocate_memory_unique({bytes_});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("OptiX query workspace allocation failed");
      workspace_ = std::move(allocation.first);
      workspace_address_ = reinterpret_cast<uint64_t>(
          static_cast<cuda::CudaDevice &>(device()).get_memory_addr(*workspace_));
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
            reinterpret_cast<void *>(workspace_address_ + 64), records.data(),
            records.size() * sizeof(TiForgeOptixFaceRule));
      }
    }
    query_ = {uint32_t(faces_ ? sizeof(TiForgeOptixFaceTraceDesc)
                               : sizeof(TiForgeOptixAlphaTraceDesc)),
              op_.ray_count, address(op_.rays), address(op_.hits),
              op_.hit_indices ? address(*op_.hit_indices) : 0,
              0, 0, workspace_address_, 0, uint32_t(op_.any_hit)};
  }
  ~CudaOptixQuery() override {
    if (!workspace_)
      return;
    if (!calls_safe(scene_->lease->context->fault)) {
      workspace_.release();
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      workspace_.reset();
    } catch (...) {
      workspace_.release();
    }
  }
  Arch arch() const override { return Arch::cuda; }
  Device &device() const override { return scene_->device(); }
  size_t workspace_bytes() const override { return bytes_; }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return scene_;
  }
  void run_cuda_ordered(void *stream) override {
    const auto &api = scene_->lease->context->lease->api;
    query_.cuda_stream = reinterpret_cast<uint64_t>(stream);
    if (faces_) {
      TiForgeOptixFaceTraceDesc desc{query_, workspace_address_ + 64, scene_->instance_count(),
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
  DeviceAllocationUnique workspace_;
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
    if constexpr (Description::version == 2) {
      if (op.face_rules.empty())
        throw std::invalid_argument("OptiX query v2 requires per-instance face rules");
    }
    validate_query(device, op);
  });
  std::vector<std::string> optional_ports{"hit_indices"};
  std::optional<NativeOperationRegistry::BindingGroup> group;
  if constexpr (Description::version == 2)
    group = NativeOperationRegistry::BindingGroup{"face_table_count", {"faces"}};
  else
    optional_ports.push_back("faces");
  registry.register_schema<Description>(
      "optix", "query", {"scene", "rays", "hits"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<Description> {
        using namespace operation_json;
        if constexpr (Description::version == 2)
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
        if constexpr (Description::version == 2) {
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
          if constexpr (Description::version == 2) {
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
          return op;
        };
      }, {}, optional_ports, group);
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
  registry.define<OptixInstancesDescription>(validate_instances, NativeObjectKind::resource);
  registry.register_schema<OptixInstancesDescription>(
      "optix", "instance_scene", {"context", "transforms"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<OptixInstancesDescription> {
        using namespace operation_json;
        fields(p, {"instance_count", "custom_indices", "visibility_masks", "opaque", "allow_update"});
        const auto size = count(p, "instance_count");
        const auto ids = instance_values(p, "custom_indices", size, 0xffffffu, 0);
        const auto masks = instance_values(p, "visibility_masks", size, 0xffu, 0xffu);
        const auto opaque = instance_values(p, "opaque", size, 1, 0, true);
        const bool update = optional_bool(p, "allow_update");
        return [=](const NativeOperationBindings &b) {
          OptixInstancesDescription d;
          d.context = b.at("context").resource<OptixContextResource>();
          d.transforms = b.at("transforms").array();
          d.custom_indices = ids;
          d.visibility_masks = masks;
          d.opaque = opaque;
          d.allow_update = update;
          d.gas.reserve(size);
          for (uint32_t i = 0; i < size; ++i)
            d.gas.push_back(b.at("gas_" + std::to_string(i)).resource<OptixSceneResource>());
          return d;
        };
      }, {}, {}, NativeOperationRegistry::BindingGroup{"instance_count", {"gas"}});
  register_query<OptixQueryOperation>(registry);
  register_query<OptixBoundFaceQueryOperation>(registry);
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
  registry.register_context_factory<OptixInstancesDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const OptixInstancesDescription &d) {
        return prepare<CudaOptixScene>(c, d);
      });
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
