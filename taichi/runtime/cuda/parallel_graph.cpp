#include "taichi/runtime/cuda/parallel_graph.h"

#include <stdexcept>
#include <utility>
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang::cuda {

void ParallelGraphCapture::prepare(std::size_t branch_count) {
  if (branch_count < 2)
    throw std::invalid_argument(
        "CUDA parallel capture requires at least two branches");
  auto &driver = CUDADriver::get_instance();
  if (!prepared_ || branches_.size() != branch_count) {
    release(true);
    // Publish storage before allocating handles, so partial construction is
    // retired by the same owner path after a driver/allocation failure.
    branches_.resize(branch_count);
    driver.event_create(&fork_, CU_EVENT_DISABLE_TIMING);
    for (auto &branch : branches_) {
      driver.stream_create(&branch.stream, CU_STREAM_NON_BLOCKING);
      driver.event_create(&branch.tail, CU_EVENT_DISABLE_TIMING);
    }
    prepared_ = true;
  }
  for (const auto &branch : branches_)
    driver.stream_synchronize(branch.stream);
}

std::uint32_t ParallelGraphCapture::capture(void *root_stream,
                                            const Recorder &record) const {
  if (!prepared_ || !root_stream || !record)
    throw std::invalid_argument("CUDA parallel capture is not prepared");
  auto &driver = CUDADriver::get_instance();
  auto error = driver.event_record.call(fork_, root_stream);
  if (error != CUDA_SUCCESS)
    return error;
  for (std::size_t i = 0; i < branches_.size(); ++i) {
    const auto &branch = branches_[i];
    error = driver.stream_wait_event.call(branch.stream, fork_, 0);
    if (error != CUDA_SUCCESS)
      return error;
    error = record(i, branch.stream);
    if (error != CUDA_SUCCESS)
      return error;
    error = driver.event_record.call(branch.tail, branch.stream);
    if (error != CUDA_SUCCESS)
      return error;
  }
  for (const auto &branch : branches_) {
    error = driver.stream_wait_event.call(root_stream, branch.tail, 0);
    if (error != CUDA_SUCCESS)
      return error;
  }
  return CUDA_SUCCESS;
}

void ParallelGraphCapture::release(bool backend_safe) {
  prepared_ = false;
  for (auto &branch : branches_) {
    if (auto tail = std::exchange(branch.tail, nullptr); backend_safe && tail)
      CUDADriver::get_instance().event_destroy(tail);
    if (auto stream = std::exchange(branch.stream, nullptr);
        backend_safe && stream)
      CUDADriver::get_instance().stream_destroy(stream);
  }
  branches_.clear();
  if (auto fork = std::exchange(fork_, nullptr); backend_safe && fork)
    CUDADriver::get_instance().event_destroy(fork);
}

}  // namespace taichi::lang::cuda
