#include "taichi/runtime/native_image.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/texture_storage.h"
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
struct SampledImageDescription {
  static constexpr const char *kind = "sampled_image";
  static constexpr uint32_t version = 1;
  ImageParams parameters;
  std::optional<NativeArrayBinding> pixels;
};
struct ImageUploadOperation {
  static constexpr const char *kind = "image_upload";
  static constexpr uint32_t version = 1;
  std::shared_ptr<NativeSampledImage> image;
  NativeArrayBinding pixels;
};

BufferFormat image_format(const std::string &name) {
#define PER_BUFFER_FORMAT(format) \
  if (name == #format)            \
    return BufferFormat::format;
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_BUFFER_FORMAT
  throw std::invalid_argument("Unknown sampled image format: " + name);
}

ImageParams parameters(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p, {"shape", "format", "filter", "address_modes"});
  const auto &shape = p["shape"];
  if (!shape.is_arr() || shape.arr.inner.empty() || shape.arr.inner.size() > 3)
    throw std::invalid_argument(
        "Sampled image requires one to three dimensions");
  ImageParams result{};
  result.dimension = shape.arr.inner.size() == 1   ? ImageDimension::d1D
                     : shape.arr.inner.size() == 2 ? ImageDimension::d2D
                                                   : ImageDimension::d3D;
  uint32_t *extents[] = {&result.x, &result.y, &result.z};
  for (size_t i = 0; i < shape.arr.inner.size(); ++i) {
    const auto extent = integer(shape.arr.inner[i]);
    if (!extent || extent > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
          "Image extent must fit the positive uint32 RHI ABI");
    *extents[i] = uint32_t(extent);
  }
  result.format = image_format(name(p["format"]));
  if (!is_float_sampled_texture_format(result.format))
    throw std::invalid_argument(
        "Image format does not support floating sampling");
  if (p.obj.inner.count("filter")) {
    const auto filter = name(p["filter"]);
    if (filter != "nearest" && filter != "linear")
      throw std::invalid_argument("Image filter must be nearest or linear");
    result.sampler_config.min_filter = result.sampler_config.mag_filter =
        filter == "nearest" ? ImageFilter::nearest : ImageFilter::linear;
  }
  if (p.obj.inner.count("address_modes")) {
    const auto &modes = p["address_modes"];
    if (!modes.is_arr() || modes.arr.inner.size() != shape.arr.inner.size())
      throw std::invalid_argument(
          "Image address modes must match dimensionality");
    ImageAddressMode *targets[] = {&result.sampler_config.address_mode_u,
                                   &result.sampler_config.address_mode_v,
                                   &result.sampler_config.address_mode_w};
    for (size_t i = 0; i < modes.arr.inner.size(); ++i) {
      const auto mode = name(modes.arr.inner[i]);
      if (mode == "repeat")
        *targets[i] = ImageAddressMode::repeat;
      else if (mode == "mirrored_repeat")
        *targets[i] = ImageAddressMode::mirrored_repeat;
      else if (mode == "clamp_to_edge")
        *targets[i] = ImageAddressMode::clamp_to_edge;
      else
        throw std::invalid_argument("Unsupported image address mode");
    }
  }
  return result;
}

void validate_pixels(Device &device,
                     const ImageParams &p,
                     const NativeArrayBinding &pixels) {
  const auto [dtype, channels] = buffer_format2type_channels(p.format);
  if (pixels.pointer.device != &device || pixels.scalar_type != dtype)
    throw std::invalid_argument(
        "Image pixels require same-device matching channel type");
  std::vector<uint64_t> shape{p.x};
  if (p.dimension != ImageDimension::d1D)
    shape.push_back(p.y);
  if (p.dimension == ImageDimension::d3D)
    shape.push_back(p.z);
  const size_t scalar_bytes = data_type_size(dtype);
  size_t bytes = scalar_bytes * channels;
  for (auto extent : shape) {
    if (extent > std::numeric_limits<size_t>::max() / bytes)
      throw std::overflow_error("Image pixel storage size overflows");
    bytes *= size_t(extent);
  }
  NativeArrayLayoutRequirement required{
      shape, {}, scalar_bytes, scalar_bytes, true};
  if (!pixels.element_shape.empty()) {
    required.element_shape = {uint64_t(channels)};
    required.record_bytes *= channels;
  } else if (channels > 1) {
    required.minimum_shape.push_back(channels);
  }
  validate_native_array_layout(pixels, required);
  if (pixels.bytes != bytes)
    throw std::invalid_argument("Image pixels require exact packed storage");
}

#ifdef TI_WITH_CUDA
class CudaSampledImage final : public NativeSampledImage {
 public:
  CudaSampledImage(NativePreparationContext &context,
                   const SampledImageDescription &d)
      : NativeSampledImage(d.parameters, bool(d.pixels)),
        device_(context.device),
        fault_(context.fault_domain) {
  }
  void initialize(const SampledImageDescription &d) {
    storage_ = std::make_unique<CudaTextureStorage>(d.parameters);
    if (d.pixels) {
      CUDADriver::get_instance().context_synchronize();
      storage_->upload_device(resolve_native_storage_range(Arch::cuda,
                                                           d.pixels->pointer,
                                                           d.pixels->bytes)
                                  .memory_offset);
    }
  }
  ~CudaSampledImage() override {
    if (!storage_)
      return;
    if (!fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized) {
      storage_.release();
      return;
    }
    try {
      auto lock = CUDAContext::get_instance().get_submission_lock_guard();
      storage_.reset();
    } catch (...) {
      storage_.release();
    }
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  size_t logical_bytes() const override {
    return storage_->logical_bytes();
  }
  uint64_t cuda_sampled_object() const override {
    return storage_->sampled_object();
  }
  void upload_cuda(uint64_t source, void *stream) override {
    storage_->upload_device_async(source, stream);
  }

 private:
  Device &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<CudaTextureStorage> storage_;
};

class CudaImageUpload final : public NativeOperation {
 public:
  explicit CudaImageUpload(const ImageUploadOperation &op)
      : image_(op.image),
        source_(resolve_native_storage_range(Arch::cuda,
                                             op.pixels.pointer,
                                             op.pixels.bytes)
                    .memory_offset) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return image_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void validate_recording(NativeRecordingState &state) const override {
    state.initialized.insert(image_.get());
  }
  void record_cuda(void *stream) override {
    image_->upload_cuda(source_, stream);
  }

 private:
  std::shared_ptr<NativeSampledImage> image_;
  uint64_t source_;
};

NativeOperationAvailability available() {
  auto &d = CUDADriver::get_instance();
  return {d.get_provider() == CUDADriverProvider::nvidia_cuda &&
              d.array_3d_create.available() && d.array_destroy.available() &&
              d.tex_object_create.available() &&
              d.tex_object_destroy.available() && d.memcpy_3d.available() &&
              d.memcpy_3d_async.available(),
          0};
}
#endif
}  // namespace

void register_image_operations(NativeOperationRegistry &registry) {
  registry.define<SampledImageDescription>(
      [](Device &device, const SampledImageDescription &d) {
        if (d.pixels)
          validate_pixels(device, d.parameters, *d.pixels);
      },
      NativeObjectKind::resource);
  registry.register_schema<SampledImageDescription>(
      "image", "sampled", {},
      [](const liong::json::JsonValue &p) {
        const auto config = parameters(p);
        return [config](const NativeOperationBindings &b) {
          SampledImageDescription d{config};
          if (b.count("pixels"))
            d.pixels = b.at("pixels").array();
          return d;
        };
      },
      {}, {"pixels"});
  registry.define<ImageUploadOperation>([](Device &device,
                                           const ImageUploadOperation &op) {
    if (!op.image || &op.image->device() != &device)
      throw std::invalid_argument("Image upload requires a local image owner");
    validate_pixels(device, op.image->parameters(), op.pixels);
  });
  registry.register_schema<ImageUploadOperation>(
      "image", "upload", {"image", "pixels"},
      [](const liong::json::JsonValue &p) {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return ImageUploadOperation{
              b.at("image").resource<NativeSampledImage>(),
              b.at("pixels").array()};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<SampledImageDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &c, const SampledImageDescription &d) {
        if (!c.fault_domain || !dynamic_cast<cuda::CudaDevice *>(&c.device))
          throw std::invalid_argument(
              "Sampled image requires a CUDA owner and fault domain");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        auto owner = std::make_shared<CudaSampledImage>(c, d);
        try {
          owner->initialize(d);
        } catch (const BackendRuntimeError &error) {
          c.fault_domain->report_backend_error(error, 0);
          throw;
        }
        return owner;
      },
      [](NativeProviderSession &) { return available(); });
  registry.register_factory<ImageUploadOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &, const ImageUploadOperation &op) {
        return std::make_shared<CudaImageUpload>(op);
      },
      available);
#endif
}
}  // namespace taichi::lang
