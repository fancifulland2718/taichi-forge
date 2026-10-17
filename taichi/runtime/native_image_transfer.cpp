#include "taichi/runtime/native_image.h"
#include "taichi/runtime/image_transfer.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"
#include "taichi/program/texture.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {
struct Region {
  std::array<uint32_t, 3> offset{}, extent{};
  uint32_t mip{0}, base_layer{0}, layers{1};
};
enum class Transfer { copy, blit, upload, download };
struct TransferDescription {
  Region source, destination;
  std::shared_ptr<NativeSampledImage> input, output;
  NativeArrayBinding buffer;
  size_t byte_offset{0};
  uint32_t row_length{0}, image_height{0};
  bool linear{false};
};
template <Transfer T>
struct ImageTransfer : TransferDescription {
  static constexpr uint32_t version = 1;
  static constexpr const char *kind = T == Transfer::copy ? "image_copy_region"
                                      : T == Transfer::blit ? "image_blit"
                                      : T == Transfer::upload
                                          ? "buffer_to_image"
                                          : "image_to_buffer";
};

uint32_t u32(const liong::json::JsonValue &v) {
  const auto n = operation_json::integer(v);
  if (n > UINT32_MAX)
    throw std::invalid_argument("Image transfer value exceeds uint32");
  return uint32_t(n);
}
Region region(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p, {"offset", "extent", "mip_level", "base_layer", "layer_count"});
  Region r;
  const auto coordinates = [](const liong::json::JsonValue &v) {
    if (!v.is_arr() || v.arr.inner.size() != 3)
      throw std::invalid_argument("Image region requires three coordinates");
    return std::array<uint32_t, 3>{u32(v.arr.inner[0]), u32(v.arr.inner[1]),
                                   u32(v.arr.inner[2])};
  };
  r.extent = coordinates(p["extent"]);
  if (p.obj.inner.count("offset"))
    r.offset = coordinates(p["offset"]);
  if (p.obj.inner.count("mip_level"))
    r.mip = u32(p["mip_level"]);
  if (p.obj.inner.count("base_layer"))
    r.base_layer = u32(p["base_layer"]);
  if (p.obj.inner.count("layer_count"))
    r.layers = u32(p["layer_count"]);
  if (!r.extent[0] || !r.extent[1] || !r.extent[2])
    throw std::invalid_argument("Image region extent must be positive");
  return r;
}
bool depth(BufferFormat f) {
  return f == BufferFormat::depth16 || f == BufferFormat::depth32f ||
         f == BufferFormat::depth24stencil8;
}
void validate_region(Device &device,
                     const std::shared_ptr<NativeSampledImage> &image,
                     const Region &r) {
  if (!image || image->arch() != Arch::vulkan || &image->device() != &device)
    throw std::invalid_argument("Image transfer requires local Vulkan images");
  const auto &p = image->parameters();
  validate_image_transfer_region({p.x, p.y, p.z}, p.mip_levels, r.offset,
                                 r.extent, r.mip, r.base_layer, r.layers,
                                 "native transfer");
}
bool full_image(const NativeSampledImage &image, const Region &r) {
  const auto &p = image.parameters();
  return p.mip_levels == 1 && r.mip == 0 &&
         r.offset == std::array<uint32_t, 3>{} &&
         r.extent == std::array<uint32_t, 3>{p.x, p.y, p.z};
}

template <Transfer T>
void validate_transfer(Device &device, const ImageTransfer<T> &d) {
  if constexpr (T != Transfer::upload)
    validate_region(device, d.input, d.source);
  if constexpr (T != Transfer::download)
    validate_region(device, d.output, d.destination);
  if constexpr (T == Transfer::copy || T == Transfer::blit) {
    const auto &a = d.input->parameters(), &b = d.output->parameters();
    if (d.input == d.output)
      throw std::invalid_argument(
          "Image transfer source and destination alias");
    if (a.format != b.format || a.dimension != b.dimension || depth(a.format))
      throw std::invalid_argument(
          "Image copy/blit requires matching color formats and dimensions");
    if constexpr (T == Transfer::copy)
      if (d.source.extent != d.destination.extent)
        throw std::invalid_argument("Image copy extents must match");
  } else {
    const auto &image = T == Transfer::upload ? d.output : d.input;
    const auto &r = T == Transfer::upload ? d.destination : d.source;
    const auto format = image->parameters().format;
    if (format == BufferFormat::depth24stencil8)
      throw std::invalid_argument(
          "Packed depth/stencil transfers require separate aspect inputs");
    const auto [type, channels] = buffer_format2type_channels(format);
    const auto texel_bytes = size_t(data_type_size(type)) * channels;
    const auto bytes = required_buffer_image_bytes(
        d.byte_offset, d.row_length, d.image_height, r.extent, texel_bytes);
    if (d.buffer.pointer.device != &device || bytes > d.buffer.bytes)
      throw std::invalid_argument(
          "Image transfer exceeds its local finite buffer view");
    resolve_native_storage_range(Arch::vulkan, d.buffer.pointer,
                                 d.buffer.bytes);
    // The range resolver proved pointer.offset + bytes cannot overflow.
    validate_buffer_image_alignment(d.buffer.pointer.offset + d.byte_offset,
                                    texel_bytes, depth(format));
  }
}

template <Transfer T>
auto decode_transfer(const liong::json::JsonValue &p) {
  using namespace operation_json;
  ImageTransfer<T> d;
  if constexpr (T == Transfer::copy || T == Transfer::blit) {
    if constexpr (T == Transfer::copy)
      fields(p, {"source_region", "destination_region"});
    else
      fields(p, {"source_region", "destination_region", "filter"});
    d.source = region(p["source_region"]);
    d.destination = region(p["destination_region"]);
    if constexpr (T == Transfer::blit) {
      const auto found = p.obj.inner.find("filter");
      const auto filter =
          found == p.obj.inner.end() ? "nearest" : name(found->second);
      if (filter != "nearest" && filter != "linear")
        throw std::invalid_argument(
            "Image blit filter must be nearest or linear");
      d.linear = filter == "linear";
    }
  } else {
    fields(p, {"region", "byte_offset", "row_length", "image_height"});
    (T == Transfer::upload ? d.destination : d.source) = region(p["region"]);
    const auto offset = optional_integer(p, "byte_offset").value_or(0);
    if (offset > SIZE_MAX)
      throw std::invalid_argument(
          "Image buffer offset exceeds address representation");
    d.byte_offset = size_t(offset);
    if (p.obj.inner.count("row_length"))
      d.row_length = u32(p["row_length"]);
    if (p.obj.inner.count("image_height"))
      d.image_height = u32(p["image_height"]);
  }
  return [d](const NativeOperationBindings &b) mutable {
    auto result = d;
    if constexpr (T == Transfer::copy || T == Transfer::blit) {
      result.input = b.at("source").resource<NativeSampledImage>();
      result.output = b.at("destination").resource<NativeSampledImage>();
    } else {
      (T == Transfer::upload ? result.output : result.input) =
          b.at("image").resource<NativeSampledImage>();
      result.buffer = b.at("buffer").array();
    }
    return result;
  };
}

#ifdef TI_WITH_VULKAN
template <Transfer T>
class VulkanTransfer final : public NativeOperation {
 public:
  explicit VulkanTransfer(Device &device, const ImageTransfer<T> &d)
      : device_(static_cast<vulkan::VulkanDevice &>(device)), d_(d) {
    auto qualify = [&](const std::shared_ptr<NativeSampledImage> &image,
                       bool write) {
      if (!image)
        return;
      const auto allocation = image->gfx_sampled_allocation();
      if (allocation.device != &device_)
        throw std::invalid_argument(
            "Image transfer has a foreign image allocation");
      const auto handle = std::get<0>(device_.get_vk_image(allocation));
      const auto usage = write ? VK_IMAGE_USAGE_TRANSFER_DST_BIT
                               : VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      if (!handle || !(handle->usage & usage))
        throw std::invalid_argument(
            "Image transfer requires image transfer usage");
    };
    qualify(d_.input, false);
    qualify(d_.output, true);
    if constexpr (T == Transfer::copy || T == Transfer::blit) {
      if (std::get<0>(device_.get_vk_image(d_.input->gfx_sampled_allocation()))
              ->image ==
          std::get<0>(device_.get_vk_image(d_.output->gfx_sampled_allocation()))
              ->image)
        throw std::invalid_argument("Image transfer allocations alias");
      if constexpr (T == Transfer::blit) {
        if (!device_.image_blit_supported(d_.input->parameters().format,
                                          d_.output->parameters().format,
                                          d_.linear))
          throw std::invalid_argument(
              "Image blit format/filter is unsupported by the device");
        for (const auto &r : {d_.source, d_.destination})
          for (size_t i = 0; i < 3; ++i)
            if (uint64_t(r.offset[i]) + r.extent[i] > INT32_MAX)
              throw std::invalid_argument(
                  "Image blit end coordinate exceeds signed RHI "
                  "representation");
      }
    } else {
      const auto buffer = device_.get_vkbuffer(d_.buffer.pointer);
      const auto usage = T == Transfer::upload
                             ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                             : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      if (!buffer || !(buffer->usage & usage))
        throw std::invalid_argument(
            "Image transfer requires buffer transfer usage");
      buffer_ = d_.buffer.pointer;
      buffer_.offset += d_.byte_offset;
      const auto &r = T == Transfer::upload ? d_.destination : d_.source;
      const auto &image = T == Transfer::upload ? d_.output : d_.input;
      buffer_copy_.buffer_row_length = d_.row_length;
      buffer_copy_.buffer_image_height = d_.image_height;
      buffer_copy_.image_offset = {r.offset[0], r.offset[1], r.offset[2]};
      buffer_copy_.image_extent = {r.extent[0], r.extent[1], r.extent[2]};
      buffer_copy_.image_mip_level = r.mip;
      buffer_copy_.image_aspect_flag = depth(image->parameters().format)
                                           ? VK_IMAGE_ASPECT_DEPTH_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
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
  void validate_recording(NativeRecordingState &state) const override {
    if (d_.input)
      d_.input->validate_sampling(state);
    if (d_.output) {
      if (full_image(*d_.output, d_.destination))
        state.initialized.insert(d_.output.get());
      else
        d_.output->validate_sampling(state);
    }
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    NativeVulkanRecordingRequirements r;
    if (d_.input)
      r.images.push_back(
          {d_.input->gfx_sampled_allocation(), ImageLayout::transfer_src});
    if (d_.output)
      r.images.push_back(
          {d_.output->gfx_sampled_allocation(), ImageLayout::transfer_dst});
    r.graphics_queue = T == Transfer::blit || buffer_copy_.image_aspect_flag ==
                                                  VK_IMAGE_ASPECT_DEPTH_BIT;
    return r;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    if constexpr (T == Transfer::upload) {
      commands->buffer_barrier(buffer_);
      commands->buffer_to_image(d_.output->gfx_sampled_allocation(), buffer_,
                                ImageLayout::transfer_dst, buffer_copy_);
    } else if constexpr (T == Transfer::download) {
      commands->image_to_buffer(buffer_, d_.input->gfx_sampled_allocation(),
                                ImageLayout::transfer_src, buffer_copy_);
      commands->buffer_barrier(buffer_);
    } else if constexpr (T == Transfer::copy) {
      ImageCopyParams p{};
      p.width = d_.source.extent[0];
      p.height = d_.source.extent[1];
      p.depth = d_.source.extent[2];
      p.source_offset = {int32_t(d_.source.offset[0]),
                         int32_t(d_.source.offset[1]),
                         int32_t(d_.source.offset[2])};
      p.destination_offset = {int32_t(d_.destination.offset[0]),
                              int32_t(d_.destination.offset[1]),
                              int32_t(d_.destination.offset[2])};
      p.source_mip_level = d_.source.mip;
      p.destination_mip_level = d_.destination.mip;
      commands->copy_image(d_.output->gfx_sampled_allocation(),
                           d_.input->gfx_sampled_allocation(),
                           ImageLayout::transfer_dst, ImageLayout::transfer_src,
                           p);
    } else {
      ImageBlitParams p{};
      const auto set = [](ImageBlitParams::Region &to, const Region &from) {
        to.offset = {int32_t(from.offset[0]), int32_t(from.offset[1]),
                     int32_t(from.offset[2])};
        to.extent = {from.extent[0], from.extent[1], from.extent[2]};
        to.mip_level = from.mip;
      };
      set(p.source, d_.source);
      set(p.destination, d_.destination);
      p.linear_filter = d_.linear;
      commands->blit_image(d_.output->gfx_sampled_allocation(),
                           d_.input->gfx_sampled_allocation(),
                           ImageLayout::transfer_dst, ImageLayout::transfer_src,
                           p);
    }
  }

 private:
  vulkan::VulkanDevice &device_;
  const ImageTransfer<T> d_;
  DevicePtr buffer_{};
  BufferImageCopyParams buffer_copy_{};
};
#endif

template <Transfer T>
void register_transfer(NativeOperationRegistry &registry,
                       const char *operation) {
  registry.define<ImageTransfer<T>>(validate_transfer<T>);
  registry.register_schema<ImageTransfer<T>>(
      "image", operation,
      T == Transfer::copy || T == Transfer::blit
          ? std::vector<std::string>{"source", "destination"}
          : std::vector<std::string>{"image", "buffer"},
      decode_transfer<T>);
#ifdef TI_WITH_VULKAN
  registry.register_factory<ImageTransfer<T>>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const ImageTransfer<T> &d) {
        return std::make_shared<VulkanTransfer<T>>(device, d);
      },
      [] { return NativeOperationAvailability{true, 0}; });
#endif
}
}  // namespace
void register_image_transfer_operations(NativeOperationRegistry &registry) {
  register_transfer<Transfer::copy>(registry, "copy_region");
  register_transfer<Transfer::blit>(registry, "blit");
  register_transfer<Transfer::upload>(registry, "buffer_to_image");
  register_transfer<Transfer::download>(registry, "image_to_buffer");
}
}  // namespace taichi::lang
