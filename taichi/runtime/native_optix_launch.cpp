#include "taichi/runtime/native_optix_program.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/native_image.h"
#include "taichi/common/operation_json.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
using Json = liong::json::JsonValue;
using namespace operation_json;

uint32_t u32(const Json &v) {
  const auto n = integer(v);
  if (n > UINT32_MAX)
    throw std::invalid_argument("OptiX launch integer exceeds uint32 ABI");
  return uint32_t(n);
}
const std::vector<Json> &array(const Json &v) {
  if (!v.is_arr() || v.arr.inner.size() > UINT32_MAX)
    throw std::invalid_argument("OptiX launch requires an ABI-sized array");
  return v.arr.inner;
}
struct Reference {
  enum Kind { buffer, texture, scene } kind;
  uint32_t index, offset;
};
struct Payload {
  uint64_t offset;
  uint32_t size, group{0};
  std::vector<Reference> references;
};
struct ArrayLayout {
  NativeArrayLayoutRequirement required;
  bool writable;
};
struct LaunchDescription {
  static constexpr const char *kind = "optix_prepared_launch";
  static constexpr uint32_t version = 1;
  std::shared_ptr<NativeOptixProgram> program;
  std::shared_ptr<NativeHostBlobResource> data;
  std::array<uint32_t, 3> dimensions;
  std::string expected;
  Payload parameters, raygen;
  std::vector<Payload> miss, hit;
  std::vector<ArrayLayout> layouts;
  std::vector<NativeArrayBinding> arrays;
  std::vector<std::shared_ptr<NativeSampledImage>> images;
  std::vector<std::shared_ptr<NativeOptixScene>> scenes;
};

Payload payload(const Json &v,
                bool record,
                const std::array<uint32_t, 3> &counts,
                std::array<std::set<uint32_t>, 3> &used) {
  if (record)
    fields(v, {"offset", "size", "group", "references"});
  else
    fields(v, {"offset", "size", "references"});
  Payload result{integer(v["offset"]), u32(v["size"])};
  if (record)
    result.group = u32(v["group"]);
  std::set<uint32_t> offsets;
  for (const auto &r : array(v["references"])) {
    fields(r, {"kind", "index", "offset"});
    const auto kind = name(r["kind"]);
    Reference ref;
    if (kind == "buffer")
      ref.kind = Reference::buffer;
    else if (kind == "texture")
      ref.kind = Reference::texture;
    else if (kind == "scene")
      ref.kind = Reference::scene;
    else
      throw std::invalid_argument("Unsupported OptiX resource reference kind");
    ref.index = u32(r["index"]);
    ref.offset = u32(r["offset"]);
    if (ref.index >= counts[ref.kind] || ref.offset % 8 ||
        ref.offset > result.size || result.size - ref.offset < 8 ||
        !offsets.insert(ref.offset).second)
      throw std::invalid_argument(
          "OptiX reference is misaligned, repeated or out of range");
    used[ref.kind].insert(ref.index);
    result.references.push_back(ref);
  }
  return result;
}

LaunchDescription parameters(const Json &p) {
  fields(p,
         {"expected_program", "dimensions", "parameters", "raygen", "miss",
          "hit", "array_count", "image_count", "scene_count", "array_layouts"});
  LaunchDescription d;
  d.expected = name(p["expected_program"]);
  if (d.expected.size() != 64 ||
      d.expected.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument(
        "OptiX launch requires a SHA256 program contract");
  const auto &dimensions = array(p["dimensions"]);
  if (dimensions.size() != 3)
    throw std::invalid_argument("OptiX launch requires three dimensions");
  for (size_t i = 0; i < 3; ++i) {
    d.dimensions[i] = u32(dimensions[i]);
    if (!d.dimensions[i])
      throw std::invalid_argument("OptiX launch dimensions must be positive");
  }
  const std::array<uint32_t, 3> counts{
      u32(p["array_count"]), u32(p["image_count"]), u32(p["scene_count"])};
  const auto &layouts = array(p["array_layouts"]);
  if (layouts.size() != counts[0])
    throw std::invalid_argument("OptiX array layout count mismatch");
  for (const auto &v : layouts) {
    fields(v, {"dtype", "shape", "element_shape", "record_bytes", "alignment",
               "access"});
    ArrayLayout layout;
    auto &r = layout.required;
    r.dtype = name(v["dtype"]);
    r.record_bytes = integer(v["record_bytes"]);
    r.alignment = integer(v["alignment"]);
    if (!r.alignment || (r.alignment & (r.alignment - 1)))
      throw std::invalid_argument(
          "OptiX array alignment must be a power of two");
    for (const auto &axis : array(v["shape"]))
      r.minimum_shape.push_back(integer(axis));
    for (const auto &axis : array(v["element_shape"]))
      r.element_shape.push_back(integer(axis));
    r.exact_shape = true;
    const auto access = name(v["access"]);
    if (access != "read" && access != "write" && access != "read_write")
      throw std::invalid_argument("OptiX array access is invalid");
    layout.writable = access != "read";
    d.layouts.push_back(std::move(layout));
  }
  std::array<std::set<uint32_t>, 3> used;
  d.parameters = payload(p["parameters"], false, counts, used);
  d.raygen = payload(p["raygen"], true, counts, used);
  for (const auto &v : array(p["miss"]))
    d.miss.push_back(payload(v, true, counts, used));
  for (const auto &v : array(p["hit"]))
    d.hit.push_back(payload(v, true, counts, used));
  for (size_t i = 0; i < 3; ++i)
    if (used[i].size() != counts[i])
      throw std::invalid_argument("OptiX launch has unused resource bindings");
  return d;
}

void same_device(Device &device, const NativeResource &resource) {
  if (resource.arch() != Arch::cuda || &resource.device() != &device)
    throw std::invalid_argument("OptiX launch requires the same CUDA device");
}
void validate(Device &device, const LaunchDescription &d) {
  same_device(device, *d.program);
  same_device(device, *d.data);
  if (d.expected != d.program->contract())
    throw std::invalid_argument("OptiX launch program contract mismatch");
  if (d.parameters.size != d.program->parameter_bytes())
    throw std::invalid_argument("OptiX launch parameter size mismatch");
  const auto &bytes = d.data->bytes();
  auto check = [&](const Payload &p, int kind) {
    if (p.offset > bytes.size() || p.size > bytes.size() - p.offset)
      throw std::invalid_argument("OptiX payload exceeds its host blob");
    if (kind >= 0 && d.program->group_kind(p.group) != uint32_t(kind))
      throw std::invalid_argument("OptiX SBT record has the wrong group kind");
    for (const auto &ref : p.references)
      for (size_t i = 0; i < 8; ++i)
        if (bytes[size_t(p.offset) + ref.offset + i])
          throw std::invalid_argument(
              "OptiX resource slots must be zero before binding");
  };
  check(d.parameters, -1);
  check(d.raygen, 0);
  for (const auto &p : d.miss)
    check(p, 1);
  for (const auto &p : d.hit)
    check(p, 2);
  std::map<std::pair<uint64_t, uint64_t>, std::pair<NativeStorageRange, bool>>
      unique;
  for (size_t i = 0; i < d.arrays.size(); ++i) {
    const auto &a = d.arrays[i];
    if (a.pointer.device != &device)
      throw std::invalid_argument("OptiX array requires the same CUDA device");
    validate_native_array_layout(a, d.layouts[i].required);
    auto range = resolve_native_storage_range(Arch::cuda, a.pointer, a.bytes);
    // Alias names for exactly the same range refer to one owner, as in JIT.
    auto [prior, inserted] =
        unique.emplace(std::make_pair(range.memory_offset, range.bytes),
                       std::make_pair(range, d.layouts[i].writable));
    if (!inserted)
      prior->second.second |= d.layouts[i].writable;
  }
  std::vector<NativeStorageRange> reads;
  std::vector<NativeStorageWrite> writes;
  for (const auto &[key, entry] : unique) {
    const auto &[range, writable] = entry;
    if (writable)
      writes.push_back({range, std::nullopt});
    else
      reads.push_back(range);
  }
  validate_native_disjoint_writes(reads, writes);
  for (const auto &image : d.images)
    same_device(device, *image);
  for (const auto &scene : d.scenes) {
    same_device(device, *scene);
    if (scene->context() != d.program->context())
      throw std::invalid_argument(
          "OptiX launch scene belongs to another context");
    scene->program_scene();  // Reject a bare GAS before adapter allocation.
  }
}

class LaunchAccess : public NativeResource,
                     public CudaProviderCompletionResource {
 public:
  virtual void validate_recording(NativeRecordingState &) const = 0;
  virtual void run(void *stream) = 0;
};
struct LaunchOperation {
  static constexpr const char *kind = "optix_program_launch";
  static constexpr uint32_t version = 1;
  std::shared_ptr<LaunchAccess> launch;
};

#ifdef TI_WITH_CUDA
class PreparedLaunch final : public LaunchAccess {
  struct Lease {
    std::shared_ptr<NativeOptixProgram> program;
    std::vector<std::shared_ptr<NativeSampledImage>> images;
    std::vector<std::shared_ptr<NativeOptixScene>> scenes;
    TiForgeOptixPreparedLaunch handle{nullptr};
  };

 public:
  explicit PreparedLaunch(const LaunchDescription &d)
      : lease_(std::make_unique<Lease>()) {
    lease_->program = d.program;
    lease_->images = d.images;
    lease_->scenes = d.scenes;
  }
  void initialize(const LaunchDescription &d) {
    const auto &api = d.program->program_api();
    const auto context = d.program->context();
    const auto &provider = context->provider_api();
    std::array<std::vector<uint64_t>, 3> addresses;
    for (const auto &a : d.arrays)
      addresses[0].push_back(
          resolve_native_storage_range(Arch::cuda, a.pointer, a.bytes)
              .memory_offset);
    for (const auto &image : d.images)
      addresses[1].push_back(image->cuda_sampled_object());
    std::vector<TiForgeOptixProgramScene> scenes;
    for (const auto &scene : d.scenes) {
      const auto binding = scene->program_scene();
      TiForgeOptixProgramSceneInfo info{sizeof(info)};
      native_optix::check(
          provider, api.scene_info(context->context_handle(), &binding, &info),
          "query program scene");
      addresses[2].push_back(info.traversable);
      scenes.push_back(binding);
    }
    std::vector<std::vector<uint8_t>> payloads;
    payloads.reserve(2 + d.miss.size() + d.hit.size());
    auto pack = [&](const Payload &p) {
      const auto &source = d.data->bytes();
      payloads.emplace_back(source.begin() + size_t(p.offset),
                            source.begin() + size_t(p.offset) + p.size);
      auto &bytes = payloads.back();
      for (const auto &ref : p.references) {
        const auto value = addresses[ref.kind][ref.index];
        for (size_t i = 0; i < 8; ++i)
          bytes[ref.offset + i] = uint8_t(value >> (8 * i));
      }
      return TiForgeOptixSbtRecord{p.group, p.size,
                                   bytes.empty() ? nullptr : bytes.data()};
    };
    const auto params = pack(d.parameters), raygen = pack(d.raygen);
    std::vector<TiForgeOptixSbtRecord> miss, hit;
    for (const auto &p : d.miss)
      miss.push_back(pack(p));
    for (const auto &p : d.hit)
      hit.push_back(pack(p));
    const TiForgeOptixPrepareLaunchDesc desc{sizeof(desc),
                                             d.dimensions[0],
                                             d.dimensions[1],
                                             d.dimensions[2],
                                             params.data,
                                             params.data_size,
                                             raygen,
                                             miss.data(),
                                             uint32_t(miss.size()),
                                             hit.data(),
                                             uint32_t(hit.size()),
                                             scenes.data(),
                                             uint32_t(scenes.size())};
    native_optix::check(
        provider,
        api.prepare(d.program->program_handle(), &desc, &lease_->handle),
        "prepare program launch");
    if (!lease_->handle)
      throw std::runtime_error("OptiX returned a null prepared launch");
    TiForgeOptixProgramMemory memory{sizeof(memory)};
    native_optix::check(provider, api.memory(lease_->handle, &memory),
                        "query program launch memory");
    if (memory.device_data_bytes > std::numeric_limits<size_t>::max())
      throw std::overflow_error("OptiX launch device storage overflows");
    bytes_ = size_t(memory.device_data_bytes);
    // Ordered CUDA executions use the owner's default stream. Finish the sole
    // initialization upload before publishing; replay only invokes launch().
    native_optix::check(provider, api.initialize(lease_->handle, 0),
                        "initialize program launch");
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  ~PreparedLaunch() override {
    if (!lease_->handle)
      return;
    if (!native_optix::calls_safe(lease_->program->context()->fault_domain())) {
      lease_.release();
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      const auto result =
          lease_->program->program_api().destroy_launch(lease_->handle);
      if (result != TI_FORGE_OPTIX_SUCCESS) {
        TI_WARN("OptiX program launch cleanup failed (result {}).",
                int(result));
        lease_.release();
      }
    } catch (...) {
      lease_.release();
    }
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return lease_->program->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  size_t device_payload_bytes() const override {
    return bytes_;
  }
  bool retains_input(const std::string &port) const override {
    return port != "data";
  }
  void validate_recording(NativeRecordingState &state) const override {
    for (const auto &image : lease_->images)
      image->validate_sampling(state);
  }
  void run(void *stream) override {
    native_optix::check(lease_->program->context()->provider_api(),
                        lease_->program->program_api().launch(
                            lease_->handle, reinterpret_cast<uint64_t>(stream)),
                        "launch program");
  }

 private:
  std::unique_ptr<Lease> lease_;
  size_t bytes_{0};
};
class RunLaunch final : public NativeOperation {
 public:
  explicit RunLaunch(std::shared_ptr<LaunchAccess> launch)
      : launch_(std::move(launch)) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return launch_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  void validate_recording(NativeRecordingState &state) const override {
    launch_->validate_recording(state);
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return launch_;
  }
  void run_cuda_ordered(void *stream) override {
    launch_->run(stream);
  }

 private:
  std::shared_ptr<LaunchAccess> launch_;
};
#endif
}  // namespace

void register_optix_launch_operations(NativeOperationRegistry &registry) {
  registry.define<LaunchDescription>(validate, NativeObjectKind::resource);
  registry.register_schema<LaunchDescription>(
      "optix", "prepared_launch", {"program", "data"},
      [](const Json &p) {
        const auto desc = parameters(p);
        const auto images = u32(p["image_count"]),
                   scenes = u32(p["scene_count"]);
        return [desc, images, scenes](const NativeOperationBindings &b) {
          auto d = desc;
          d.program = b.at("program").resource<NativeOptixProgram>();
          d.data = b.at("data").resource<NativeHostBlobResource>();
          for (size_t i = 0; i < d.layouts.size(); ++i)
            d.arrays.push_back(b.at("array_" + std::to_string(i)).array());
          for (uint32_t i = 0; i < images; ++i)
            d.images.push_back(b.at("image_" + std::to_string(i))
                                   .resource<NativeSampledImage>());
          for (uint32_t i = 0; i < scenes; ++i)
            d.scenes.push_back(b.at("scene_" + std::to_string(i))
                                   .resource<NativeOptixScene>());
          return d;
        };
      },
      {}, {},
      {{"array_count", {"array"}},
       {"image_count", {"image"}},
       {"scene_count", {"scene"}}});
  registry.define<LaunchOperation>(
      [](Device &device, const LaunchOperation &d) {
        same_device(device, *d.launch);
      });
  registry.register_schema<LaunchOperation>(
      "optix", "launch", {"launch"}, [](const Json &p) {
        fields(p, {});
        return [](const NativeOperationBindings &b) {
          return LaunchOperation{b.at("launch").resource<LaunchAccess>()};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<LaunchDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const LaunchDescription &d) {
        if (!c.fault_domain || !dynamic_cast<cuda::CudaDevice *>(&c.device))
          throw std::invalid_argument(
              "OptiX launch requires its CUDA owner and fault domain");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        auto result = std::make_shared<PreparedLaunch>(d);
        try {
          result->initialize(d);
          return result;
        } catch (const BackendRuntimeError &error) {
          c.fault_domain->report_backend_error(error, 0);
          throw;
        }
      });
  registry.register_factory<LaunchOperation>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](Arch, Device &, const LaunchOperation &d) {
        return std::make_shared<RunLaunch>(d.launch);
      });
#endif
}
}  // namespace taichi::lang
