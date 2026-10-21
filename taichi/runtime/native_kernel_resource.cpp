#include "taichi/runtime/native_kernel_resource.h"

#include "taichi/runtime/native_image.h"
#include "taichi/runtime/native_texture_collection.h"
#include "taichi/runtime/native_acceleration_structure.h"
#include "taichi/runtime/native_argpack_binding.h"

namespace taichi::lang {
std::shared_ptr<NativeKernelResourceBinding> prepare_native_kernel_resource(
    std::shared_ptr<NativeResource> resource,
    std::vector<int> argument,
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    LaunchContextBuilder &context) {
  if (auto pack = std::dynamic_pointer_cast<NativeArgPackStorage>(resource))
    return std::make_shared<NativeArgPackKernelBinding>(
        std::move(pack), std::move(argument), arch, device, kernel, context);
  if (auto collection =
          std::dynamic_pointer_cast<NativeTextureCollection>(resource))
    return std::make_shared<NativeTextureCollectionKernelBinding>(
        std::move(collection), std::move(argument), arch, device, kernel, context);
  if (auto image = std::dynamic_pointer_cast<NativeSampledImage>(resource))
    return std::make_shared<NativeImageKernelBinding>(
        std::move(image), std::move(argument), arch, device, kernel, context);
  if (auto acceleration_structure =
          std::dynamic_pointer_cast<NativeAccelerationStructure>(resource))
    return std::make_shared<NativeAccelerationStructureKernelBinding>(
        std::move(acceleration_structure), std::move(argument), arch, device,
        kernel, context);
  throw std::invalid_argument(
      "Kernel native resource has no qualified parameter binding");
}
}  // namespace taichi::lang
