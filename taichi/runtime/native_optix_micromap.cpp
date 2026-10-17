#include "taichi/runtime/native_optix_micromap.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"
#include "picosha2.h"
#include <cstring>

namespace taichi::lang {
NativeOptixMicromap::NativeOptixMicromap(
    Device &device,
    uint32_t triangles,
    std::shared_ptr<NativeHostBlobResource> data,
    const NativeHostBlobResource &descriptors,
    const NativeHostBlobResource *indices)
    : device_(device), triangles_(triangles), data_(std::move(data)) {
  const auto &entries = descriptors.bytes();
  static_assert(sizeof(TiForgeOptixMicromapEntry) == 8);
  if (entries.size() % 8 || entries.size() / 8 > UINT32_MAX ||
      entries.empty() != data_->bytes().empty() ||
      (indices ? indices->bytes().size() != uint64_t(triangles) * 4
               : entries.size() / 8 != triangles))
    throw std::invalid_argument(
        "OptiX micromap packed input sizes are inconsistent");
  // Copy into correctly aligned native objects; no typed alias into byte blobs.
  descriptors_.resize(entries.size() / 8);
  if (!entries.empty())
    std::memcpy(descriptors_.data(), entries.data(), entries.size());
  if (indices) {
    indices_.resize(triangles);
    std::memcpy(indices_.data(), indices->bytes().data(),
                indices->bytes().size());
  }
  picosha2::hash256_one_by_one hash;
  const std::string tag = "forge-optix-baked-omm1";
  hash.process(tag.begin(), tag.end());
  for (uint64_t size :
       {uint64_t(data_->bytes().size()), uint64_t(entries.size()),
        indices ? uint64_t(indices->bytes().size()) : 0}) {
    uint8_t bytes[8];
    for (size_t i = 0; i < 8; ++i)
      bytes[i] = uint8_t(size >> (8 * i));
    hash.process(bytes, bytes + 8);
  }
  for (const auto *blob :
       {static_cast<const NativeHostBlobResource *>(data_.get()), &descriptors,
        indices})
    if (blob)
      hash.process(blob->bytes().begin(), blob->bytes().end());
  hash.finish();
  fingerprint_ = picosha2::get_hash_hex_string(hash);
}

TiForgeOptixMicromapDesc NativeOptixMicromap::description() const {
  return {sizeof(TiForgeOptixMicromapDesc),
          uint32_t(descriptors_.size()),
          uint64_t(data_->bytes().size()),
          data_->bytes().data(),
          descriptors_.data(),
          triangles_,
          0,
          indices_.empty() ? nullptr : indices_.data()};
}

namespace {
struct Description {
  static constexpr const char *kind = "optix_opacity_micromap";
  static constexpr uint32_t version = 1;
  uint32_t triangles;
  std::shared_ptr<NativeHostBlobResource> data, descriptors, indices;
};
}  // namespace

void register_optix_micromap_resources(NativeOperationRegistry &registry) {
  registry.define<Description>(
      [](Device &device, const Description &d) {
        for (const auto &blob : {d.data, d.descriptors, d.indices})
          if (blob &&
              (&blob->device() != &device || blob->arch() != Arch::cuda))
            throw std::invalid_argument(
                "OptiX micromap requires local CUDA host blobs");
        if (!d.data || !d.descriptors)
          throw std::invalid_argument(
              "OptiX micromap requires data and descriptors");
      },
      NativeObjectKind::resource);
  registry.register_schema<Description>(
      "optix", "opacity_micromap", {"data", "descriptors"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<Description> {
        operation_json::fields(p, {"triangle_count"});
        const auto count = operation_json::integer(p["triangle_count"]);
        if (!count || count > UINT32_MAX)
          throw std::invalid_argument(
              "OptiX micromap count must fit positive uint32");
        return [count](const NativeOperationBindings &b) {
          return Description{
              uint32_t(count), b.at("data").resource<NativeHostBlobResource>(),
              b.at("descriptors").resource<NativeHostBlobResource>(),
              b.count("indices")
                  ? b.at("indices").resource<NativeHostBlobResource>()
                  : nullptr};
        };
      },
      {}, {"indices"});
#ifdef TI_WITH_CUDA
  registry.register_context_factory<Description>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const Description &d) {
        return std::make_shared<NativeOptixMicromap>(
            c.device, d.triangles, d.data, *d.descriptors, d.indices.get());
      });
#endif
}
}  // namespace taichi::lang
