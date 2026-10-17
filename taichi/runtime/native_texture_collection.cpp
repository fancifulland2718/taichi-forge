#include "taichi/runtime/native_texture_collection.h"

#include "taichi/common/operation_json.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/texture_collection.h"

namespace taichi::lang {
NativeTextureCollection::NativeTextureCollection(
    Device &device,
    int dimensions,
    const std::vector<std::shared_ptr<NativeSampledImage>> &members)
    : device_(device),
      dimensions_(dimensions),
      snapshot_id_(allocate_texture_collection_snapshot_id()) {
  validate_texture_collection_capacity(Arch::vulkan, device.get_caps(),
                                       members.size());
  allocations_.reserve(members.size());
  std::unordered_set<const NativeSampledImage *> unique;
  for (const auto &image : members) {
    if (!image || image->arch() != Arch::vulkan || &image->device() != &device)
      throw std::invalid_argument(
          "TextureCollection requires same-device Vulkan image owners");
    if (unique.insert(image.get()).second) {
      const auto &p = image->parameters();
      validate_texture_collection_member(dimensions, p.dimension, p.format);
      if (!bool(p.usage & ImageAllocUsage::Sampled))
        throw std::invalid_argument(
            "TextureCollection requires sampled-image usage");
      owners_.push_back(image);
    }
    const auto allocation = image->gfx_sampled_allocation();
    if (allocation.device != &device)
      throw std::invalid_argument(
          "TextureCollection image allocation belongs to another device");
    allocations_.push_back(allocation);
  }
}

void NativeTextureCollection::validate_sampling(
    const NativeRecordingState &state) const {
  for (const auto &image : owners_)
    image->validate_sampling(state);
}

void NativeTextureCollectionKernelBinding::validate_parameter(
    Arch arch,
    Device &device,
    const CallableBase &kernel) const {
  if (!collection_ || arch != Arch::vulkan || &collection_->device() != &device)
    throw std::invalid_argument(
        "Kernel TextureCollection requires a same-device Vulkan owner");
  const auto found = kernel.nested_parameters.find(argument_);
  if (found == kernel.nested_parameters.end())
    throw std::invalid_argument("TextureCollection kernel argument is missing");
  const auto &p = found->second;
  if (p.ptype != ParameterType::kTextureCollection || p.needs_grad ||
      p.total_dim != collection_->dimensions() || p.element_shape.size() != 1 ||
      p.element_shape[0] <= 0 ||
      size_t(p.element_shape[0]) != collection_->allocations().size())
    throw std::invalid_argument(
        "TextureCollection does not match the kernel dimension/capacity");
}

NativeTextureCollectionKernelBinding::NativeTextureCollectionKernelBinding(
    std::shared_ptr<NativeTextureCollection> collection,
    std::vector<int> argument,
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    LaunchContextBuilder &context)
    : collection_(std::move(collection)),
      argument_(std::move(argument)),
      kernel_(&kernel),
      context_(&context) {
  validate_parameter(arch, device, kernel);
  context.set_arg_texture_collection_impl(argument_, collection_->dimensions(),
                                          collection_->allocations(),
                                          collection_->snapshot_id());
}

void NativeTextureCollectionKernelBinding::validate(
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    const LaunchContextBuilder &context,
    const NativeRecordingState &state) const {
  validate_parameter(arch, device, kernel);
  const auto table = context.texture_collections.find(argument_);
  const auto snapshot = context.texture_collection_snapshot_ids.find(argument_);
  const auto kind = context.device_allocation_type.find(argument_);
  if (kernel_ != &kernel || context_ != &context ||
      table == context.texture_collections.end() ||
      table->second != collection_->allocations() ||
      snapshot == context.texture_collection_snapshot_ids.end() ||
      snapshot->second != collection_->snapshot_id() ||
      kind == context.device_allocation_type.end() ||
      kind->second != LaunchContextBuilder::DevAllocType::kTextureCollection)
    throw std::invalid_argument(
        "Kernel TextureCollection binding changed after qualification");
  collection_->validate_sampling(state);
}

namespace {
struct CollectionDescription {
  static constexpr const char *kind = "sampled_image_collection";
  static constexpr uint32_t version = 1;
  int dimensions;
  std::vector<std::shared_ptr<NativeSampledImage>> members;
};
}  // namespace

void register_texture_collection_operations(NativeOperationRegistry &registry) {
  registry.define<CollectionDescription>(
      [](Device &, const CollectionDescription &) {},
      NativeObjectKind::resource);
  registry.register_schema<CollectionDescription>(
      "image", "collection", {},
      [](const liong::json::JsonValue &p) {
        operation_json::fields(p, {"dimensions", "image_count"});
        const auto dimensions = operation_json::integer(p["dimensions"]);
        const auto count = operation_json::integer(p["image_count"]);
        if (dimensions < 1 || dimensions > 3 || !count || count > INT32_MAX)
          throw std::invalid_argument(
              "TextureCollection requires 1-3 dimensions and positive i32 "
              "capacity");
        return [dimensions, count](const NativeOperationBindings &b) {
          CollectionDescription d{int(dimensions)};
          d.members.reserve(size_t(count));
          for (size_t i = 0; i < count; ++i)
            d.members.push_back(b.at("image_" + std::to_string(i))
                                    .resource<NativeSampledImage>());
          return d;
        };
      },
      {}, {}, {{"image_count", {"image"}}});
#ifdef TI_WITH_VULKAN
  registry.register_factory<CollectionDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const CollectionDescription &d) {
        return std::make_shared<NativeTextureCollection>(device, d.dimensions,
                                                         d.members);
      },
      [] { return NativeOperationAvailability{true, 0}; });
#endif
}
}  // namespace taichi::lang
