#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {

struct NativeDataUpload {
  DevicePtr destination;
  const void *source;
  std::size_t bytes;
};

// Cold construction of newly owned storage only. A selected uploader returns
// after the complete batch is visible, retaining source and destination storage
// through failed submissions as well. It never publishes a partially ready
// resource to a provider. Public/live resource updates use their own contracts.
using NativeDataUploader =
    std::function<void(const std::vector<NativeDataUpload> &)>;

inline void upload_native_data(Device &device,
                               const std::vector<NativeDataUpload> &uploads,
                               const NativeDataUploader &selected = {}) {
  for (const auto &upload : uploads)
    if (upload.destination.device != &device ||
        (upload.bytes && !upload.source))
      throw std::invalid_argument("Invalid owned data upload");
  if (uploads.empty())
    return;
  if (selected) {
    selected(uploads);
    return;
  }
  std::vector<DevicePtr> destinations;
  std::vector<const void *> sources;
  std::vector<std::size_t> sizes;
  for (const auto &upload : uploads) {
    if (!upload.bytes)
      continue;
    destinations.push_back(upload.destination);
    sources.push_back(upload.source);
    sizes.push_back(upload.bytes);
  }
  // Only the RHI int count ABI bounds a batch, never an application limit.
  for (std::size_t begin = 0; begin < sizes.size();) {
    const auto count = static_cast<int>(std::min<std::size_t>(
        sizes.size() - begin, std::numeric_limits<int>::max()));
    if (device.upload_data(destinations.data() + begin, sources.data() + begin,
                           sizes.data() + begin, count) != RhiResult::success)
      throw std::runtime_error("Unable to upload owned initial data");
    begin += count;
  }
}
}  // namespace taichi::lang
