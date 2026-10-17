#include "taichi/runtime/native_image.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/program/texture.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/texture_storage.h"
#include "taichi/program/runtime_fault.h"
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
void NativeImageKernelBinding::validate_parameter(
    Arch arch,
    Device &device,
    const CallableBase &kernel) const {
  if (!image_ || (arch != Arch::cuda && arch != Arch::vulkan) ||
      image_->arch() != arch || &image_->device() != &device)
    throw std::invalid_argument(
        "Kernel sampled image requires a qualified same-device CUDA/Vulkan "
        "owner");
  const auto found = kernel.nested_parameters.find(argument_);
  if (found == kernel.nested_parameters.end() ||
      found->second.ptype != ParameterType::kTexture ||
      found->second.needs_grad)
    throw std::invalid_argument(
        "Managed image requires a sampled Texture kernel parameter");
  const auto &p = image_->parameters();
  const auto dimensions = p.dimension == ImageDimension::d1D   ? 1u
                          : p.dimension == ImageDimension::d2D ? 2u
                          : p.dimension == ImageDimension::d3D ? 3u
                                                               : 0u;
  if (!dimensions || found->second.total_dim != dimensions ||
      p.mip_levels != 1 || !is_float_sampled_texture_format(p.format))
    throw std::invalid_argument(
        "Sampled image does not match the kernel dimension/format contract");
  for (auto extent : {p.x, p.y, p.z})
    if (!extent || extent > std::numeric_limits<std::int32_t>::max())
      throw std::invalid_argument(
          "Kernel sampled-image shape must fit its positive i32 ABI");
}

NativeImageKernelBinding::NativeImageKernelBinding(
    std::shared_ptr<NativeSampledImage> image,
    std::vector<int> argument,
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    LaunchContextBuilder &context)
    : image_(std::move(image)),
      argument_(std::move(argument)),
      kernel_(&kernel),
      context_(&context) {
  validate_parameter(arch, device, kernel);
  if (arch == Arch::cuda) {
    cuda_object_ = image_->cuda_sampled_object();
    if (!cuda_object_)
      throw std::invalid_argument("Managed CUDA sampled object is null");
    context.set_arg_texture_impl(argument_,
                                 reinterpret_cast<intptr_t>(&cuda_object_));
  } else {
    gfx_image_ = image_->gfx_sampled_allocation();
    if (gfx_image_.device != &device)
      throw std::invalid_argument(
          "Managed GFX sampled image is foreign or null");
    context.set_arg_texture_impl(argument_,
                                 reinterpret_cast<intptr_t>(&gfx_image_));
  }
  const auto &p = image_->parameters();
  const std::uint32_t shape[] = {p.x, p.y, p.z};
  auto shape_index = argument_;
  shape_index.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
  shape_index.push_back(0);
  for (int i = 0; i < 3; ++i) {
    shape_index.back() = i;
    context.set_struct_arg(shape_index, static_cast<std::int32_t>(shape[i]));
  }
}

void NativeImageKernelBinding::validate(
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    const LaunchContextBuilder &context,
    const NativeRecordingState &state) const {
  validate_parameter(arch, device, kernel);
  const auto pointer = context.array_ptrs.find(argument_);
  const auto kind = context.device_allocation_type.find(argument_);
  const void *descriptor = arch == Arch::cuda
                               ? static_cast<const void *>(&cuda_object_)
                               : static_cast<const void *>(&gfx_image_);
  if (kernel_ != &kernel || context_ != &context ||
      pointer == context.array_ptrs.end() || pointer->second != descriptor ||
      kind == context.device_allocation_type.end() ||
      kind->second != LaunchContextBuilder::DevAllocType::kTexture)
    throw std::invalid_argument(
        "Kernel sampled-image binding changed after qualification");
  image_->validate_sampling(state);
}

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
  fields(p, {"shape", "format", "filter", "address_modes", "storage"});
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
  result.usage = ImageAllocUsage::Sampled;
  if (optional_bool(p, "storage"))
    result.usage = result.usage | ImageAllocUsage::Storage;
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
  std::vector<std::uint64_t> cuda_binding_frame_invariants() const override {
    // CUDA array/multidimensional copies cannot change operands in an
    // instantiated graph. In-place source contents remain fully dynamic.
    return {image_->cuda_sampled_object(), source_};
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

#ifdef TI_WITH_VULKAN
// Vulkan keeps native X-contiguous image order. Pixel arrays describe packed
// storage, as with managed JIT uploads; no transpose kernel or host copy is
// added.
BufferImageCopyParams packed_image_copy(const ImageParams &p) {
  BufferImageCopyParams copy{};
  copy.buffer_row_length = p.x;
  copy.buffer_image_height = p.y;
  copy.image_extent = {p.x, p.y, p.z};
  if (p.format == BufferFormat::depth16 || p.format == BufferFormat::depth32f)
    copy.image_aspect_flag = VK_IMAGE_ASPECT_DEPTH_BIT;
  return copy;
}

class VulkanSampledImage final : public NativeSampledImage {
 public:
  VulkanSampledImage(NativePreparationContext &context,
                     const SampledImageDescription &d)
      : NativeSampledImage(d.parameters, bool(d.pixels)),
        device_(context.device),
        runtime_(*context.gfx_runtime),
        fault_(context.fault_domain) {
    const auto [type, channels] =
        buffer_format2type_channels(d.parameters.format);
    bytes_ = size_t(data_type_size(type)) * channels;
    for (auto extent : {d.parameters.x, d.parameters.y, d.parameters.z}) {
      if (extent > std::numeric_limits<size_t>::max() / bytes_)
        throw std::overflow_error("Sampled image logical size overflows");
      bytes_ *= extent;
    }
  }
  void initialize(const SampledImageDescription &d) {
    auto params = d.parameters;
    image_ = runtime_.create_image(params);
    if (d.pixels) {
      const auto image = image_;
      const auto source = d.pixels->pointer;
      const auto copy = packed_image_copy(params);
      initialization_pending_ = true;
      auto upload = [image, source, copy](Device *, CommandList *commands) {
        commands->buffer_barrier(source);
        commands->buffer_to_image(image, source, ImageLayout::transfer_dst,
                                  copy);
      };
      const std::vector<ComputeOpImageRef> refs{
          {image, ImageLayout::transfer_dst, ImageLayout::shader_read}};
      if (copy.image_aspect_flag == VK_IMAGE_ASPECT_DEPTH_BIT) {
        runtime_.enqueue_graphics_op_lambda(upload, refs);
      } else {
        runtime_.enqueue_compute_op_lambda(upload, refs);
      }
      runtime_.synchronize();
      initialization_pending_ = false;
    }
  }
  ~VulkanSampledImage() override {
    if (image_.device == nullptr || !fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized)
      return;
    // Fixed execution has retired its command buffers before this owner.
    // Failed cold preparation is covered by the enclosing runtime fault domain.
    try {
      if (initialization_pending_)
        runtime_.synchronize();
      runtime_.untrack_image(image_);
      static_cast<GraphicsDevice &>(device_).destroy_image(image_);
    } catch (...) {
    }
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
  size_t logical_bytes() const override {
    return bytes_;
  }
  DeviceAllocation gfx_sampled_allocation() const override {
    return image_;
  }

 private:
  Device &device_;
  gfx::GfxRuntime &runtime_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  DeviceAllocation image_{kDeviceNullAllocation};
  size_t bytes_{0};
  bool initialization_pending_{false};
};

class VulkanImageUpload final : public NativeOperation {
 public:
  explicit VulkanImageUpload(const ImageUploadOperation &op)
      : image_(op.image),
        source_(op.pixels.pointer),
        copy_(packed_image_copy(op.image->parameters())) {
  }
  Arch arch() const override {
    return Arch::vulkan;
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
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    // Depth transfers are not universally supported by compute-only queues.
    // Use the existing graphics-capable recording for portable correctness.
    return {{{image_->gfx_sampled_allocation(), ImageLayout::transfer_dst}},
            copy_.image_aspect_flag == VK_IMAGE_ASPECT_DEPTH_BIT};
  }
  void record_vulkan(Device *, CommandList *commands) override {
    commands->buffer_barrier(source_);
    commands->buffer_to_image(image_->gfx_sampled_allocation(), source_,
                              ImageLayout::transfer_dst, copy_);
  }

 private:
  std::shared_ptr<NativeSampledImage> image_;
  DevicePtr source_;
  BufferImageCopyParams copy_;
};
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
        if (bool(d.parameters.usage & ImageAllocUsage::Storage))
          throw std::invalid_argument("Managed storage images require Vulkan");
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
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<SampledImageDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &c, const SampledImageDescription &d) {
        if (!c.gfx_runtime || c.gfx_runtime->get_ti_device() != &c.device ||
            !c.fault_domain || !dynamic_cast<vulkan::VulkanDevice *>(&c.device))
          throw std::invalid_argument(
              "Sampled image requires its owner's Vulkan GFX runtime");
        auto owner = std::make_shared<VulkanSampledImage>(c, d);
        try {
          owner->initialize(d);
        } catch (const BackendRuntimeError &error) {
          c.fault_domain->report_backend_error(error, 0);
          throw;
        }
        return owner;
      },
      [](NativeProviderSession &) {
        return NativeOperationAvailability{true, 0};
      });
  registry.register_factory<ImageUploadOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &, const ImageUploadOperation &op) {
        return std::make_shared<VulkanImageUpload>(op);
      },
      [] { return NativeOperationAvailability{true, 0}; });
#endif
}
}  // namespace taichi::lang
