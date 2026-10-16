#pragma once

#include <cstddef>

namespace taichi::lang::cuda {

bool driver_matrix_mma_f16_f32_available();

// Load the shared PTX before entering CUDA stream capture. Performs no launch.
void driver_prepare_matrix_mma_f16_f32();

std::size_t driver_matrix_mma_f16_f32(void *a,
                                      void *b,
                                      void *output,
                                      int batch_count,
                                      void *stream = nullptr);

}  // namespace taichi::lang::cuda
