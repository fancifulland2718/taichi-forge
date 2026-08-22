#pragma once

#include <cstddef>

namespace taichi::lang::cuda {

bool driver_matrix_mma_f16_f32_available();

std::size_t driver_matrix_mma_f16_f32(void *a,
                                      void *b,
                                      void *output,
                                      int batch_count,
                                      void *stream = nullptr);

}  // namespace taichi::lang::cuda
