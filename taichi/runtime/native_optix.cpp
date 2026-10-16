#include "taichi/runtime/native_operation_registry.h"

#include <algorithm>
#include <limits>
#include <optional>
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

class OptixSceneResource : public NativeResource {
 public:
  explicit OptixSceneResource(OptixSceneDescription description)
      : description(std::move(description)) {
  }
  const OptixSceneDescription description;
};

struct OptixQueryOperation {
  static constexpr const char *kind = "optix_query";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixSceneResource> scene;
  NativeArrayBinding rays, hits;
  std::optional<NativeArrayBinding> hit_indices, faces;
  uint32_t ray_count{0};
  bool occlusion{false}, any_hit{false}, front_only{false};
};

struct OptixRefitOperation {
  static constexpr const char *kind = "optix_refit";
  static constexpr uint32_t version = 1;
  std::shared_ptr<OptixSceneResource> scene;
  NativeArrayBinding vertices;
};

uint32_t count(const liong::json::JsonValue &p, const char *key) {
  const auto value = operation_json::integer(p[key]);
  if (!value || value > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("OptiX counts must fit the positive uint32 ABI");
  return uint32_t(value);
}

NativeStorageRange layout(Device &device,
                           const NativeArrayBinding &array,
                           uint32_t count,
                           uint32_t width,
                           bool integer) {
  if (!count || array.pointer.device != &device ||
      (integer ? array.scalar_type != PrimitiveType::i32 &&
                     array.scalar_type != PrimitiveType::u32
               : array.scalar_type != PrimitiveType::f32) ||
      uint64_t(count) * width > std::numeric_limits<size_t>::max() / 4 ||
      array.bytes != uint64_t(count) * width * 4)
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
      op.scene->arch() != Arch::cuda ||
      bool(op.hit_indices) == op.occlusion || (op.front_only && op.faces) ||
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
                           op.scene->description.triangle_count, 1, true));
  // Scene inputs can remain public arguments; a query must not corrupt them
  // through a differently named output or an imported/subview alias.
  reads.push_back(layout(device, op.scene->description.vertices,
                         op.scene->description.vertex_count, 3, false));
  reads.push_back(layout(device, op.scene->description.indices,
                         op.scene->description.triangle_count, 3, true));
  validate_native_disjoint_writes(reads, writes);
}

void validate_refit(Device &device, const OptixRefitOperation &op) {
  if (!op.scene || &op.scene->device() != &device ||
      op.scene->arch() != Arch::cuda || !op.scene->description.allow_update)
    throw std::invalid_argument("OptiX refit requires a local updateable scene");
  layout(device, op.vertices, op.scene->description.vertex_count, 3, false);
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
      !api.trace_faces || !api.trace_occlusion || !api.get_last_error)
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

struct SceneLease {
  std::shared_ptr<CudaOptixContext> context;
  DeviceAllocationUnique indices;
  TiForgeOptixTriangleScene handle{nullptr};
  TiForgeOptixTriangleSceneDesc desc{};
};

class CudaOptixScene final : public OptixSceneResource,
                              public CudaProviderCompletionResource {
 public:
  explicit CudaOptixScene(OptixSceneDescription description)
      : OptixSceneResource(std::move(description)),
        lease(std::make_unique<SceneLease>()) {
    lease->context =
        std::dynamic_pointer_cast<CudaOptixContext>(this->description.context);
    if (!lease->context)
      throw std::invalid_argument("OptiX requires its native context owner");
  }
  void initialize() {
    const auto &d = description;
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
    check(api, api.create_triangle_scene(lease->context->lease->handle,
                                         &lease->desc, &lease->handle),
          "create triangle scene");
    if (!lease->handle)
      throw std::runtime_error("OptiX returned a null scene");
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
        const auto result = lease->context->lease->api.destroy_triangle_scene(
            lease->handle);
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
  Device &device() const override { return description.context->device(); }
  size_t workspace_bytes() const override { return bytes_; }
  void update_memory() {
    TiForgeOptixSceneMemory memory{sizeof(memory)};
    const auto &api = lease->context->lease->api;
    check(api, api.get_scene_memory(lease->handle, &memory), "query scene memory");
    uint64_t bytes = description.indices.bytes;
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
    faces_ = op_.front_only || bool(op_.faces);
    extended_ = faces_ || op_.occlusion || op_.any_hit;
    check(api, (faces_ ? api.prepare_faces
                        : extended_ ? api.prepare_alpha : api.prepare_typed)(
                   context.lease->handle),
          "prepare query pipeline");
    scene_->update_memory();
    if (extended_) {
      bytes_ = faces_ ? 80 : 48;
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
        TiForgeOptixFaceRule rule{0, 1, 0};
        auto &driver = CUDADriver::get_instance();
        if (op_.faces) {
          std::vector<uint32_t> rules(scene_->description.triangle_count);
          driver.context_synchronize();
          driver.memcpy_device_to_host(rules.data(),
                                       reinterpret_cast<void *>(address(*op_.faces)),
                                       op_.faces->bytes);
          for (auto value : rules)
            if (value > 1)
              throw std::invalid_argument("OptiX face rules must be 0 or 1");
          rule = {workspace_address_ + 80, 2, 0};
          driver.memcpy_host_to_device(reinterpret_cast<void *>(rule.primitives),
                                       rules.data(), op_.faces->bytes);
        }
        driver.memcpy_host_to_device(
            reinterpret_cast<void *>(workspace_address_ + 64), &rule, sizeof(rule));
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
      TiForgeOptixFaceTraceDesc desc{query_, workspace_address_ + 64, 1,
                                    uint32_t(op_.occlusion)};
      check(api, api.trace_faces(scene_->lease->handle, &desc), "trace faces");
    } else if (extended_) {
      check(api, (op_.occlusion ? api.trace_occlusion : api.trace_alpha)(
                     scene_->lease->handle, &query_),
            "trace extended query");
    } else {
      TiForgeOptixTypedTraceDesc desc{sizeof(desc), query_.ray_count, query_.rays,
                                      query_.hits, query_.hit_indices,
                                      query_.cuda_stream};
      check(api, api.trace_typed(scene_->lease->handle, &desc), "trace typed");
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
};

class CudaOptixRefit final : public NativeOperation {
 public:
  explicit CudaOptixRefit(const OptixRefitOperation &op)
      : scene_(std::dynamic_pointer_cast<CudaOptixScene>(op.scene)),
        vertices_(address(op.vertices)) {
    if (!scene_)
      throw std::invalid_argument("OptiX refit requires a native scene");
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
    auto desc = scene_->lease->desc;
    desc.vertices = vertices_;
    desc.cuda_stream = reinterpret_cast<uint64_t>(stream);
    const auto &api = scene_->lease->context->lease->api;
    check(api, api.update_triangle_scene(scene_->lease->handle, &desc), "refit");
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
  registry.define<OptixSceneDescription>(validate_scene, NativeObjectKind::resource);
  registry.register_schema<OptixSceneDescription>(
      "optix", "triangle_scene", {"context", "vertices", "indices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<OptixSceneDescription> {
        operation_json::fields(p, {"vertex_count", "triangle_count", "allow_update"});
        const auto vertices = count(p, "vertex_count"), triangles = count(p, "triangle_count");
        const bool update = operation_json::optional_bool(p, "allow_update");
        return [=](const NativeOperationBindings &b) {
          return OptixSceneDescription{b.at("context").resource<OptixContextResource>(),
                                       b.at("vertices").array(), b.at("indices").array(),
                                       vertices, triangles, update};
        };
      });
  registry.define<OptixQueryOperation>(validate_query);
  registry.register_schema<OptixQueryOperation>(
      "optix", "query", {"scene", "rays", "hits"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<OptixQueryOperation> {
        using namespace operation_json;
        fields(p, {"ray_count", "occlusion", "any_hit", "front_only"});
        const auto rays = count(p, "ray_count");
        const bool occlusion = optional_bool(p, "occlusion"),
                   any_hit = optional_bool(p, "any_hit", occlusion),
                   front_only = optional_bool(p, "front_only");
        if (occlusion && !any_hit)
          throw std::invalid_argument("OptiX occlusion requires any-hit acceptance");
        return [=](const NativeOperationBindings &b) {
          OptixQueryOperation op{b.at("scene").resource<OptixSceneResource>(),
                                  b.at("rays").array(), b.at("hits").array()};
          op.ray_count = rays;
          op.occlusion = occlusion;
          op.any_hit = any_hit;
          op.front_only = front_only;
          if (b.count("hit_indices")) op.hit_indices = b.at("hit_indices").array();
          if (b.count("faces")) op.faces = b.at("faces").array();
          return op;
        };
      }, {}, {"hit_indices", "faces"});
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
  registry.register_context_factory<OptixSceneDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const OptixSceneDescription &d) {
        return prepare<CudaOptixScene>(c, d);
      });
  registry.register_context_factory<OptixQueryOperation>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const OptixQueryOperation &op) {
        return prepare<CudaOptixQuery>(c, op);
      });
  registry.register_context_factory<OptixRefitOperation>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &, const OptixRefitOperation &op) {
        return std::make_shared<CudaOptixRefit>(op);
      });
#endif
}
}  // namespace taichi::lang
