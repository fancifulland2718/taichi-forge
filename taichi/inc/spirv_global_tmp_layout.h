#pragma once

#include <cstddef>
#include <cstdint>

namespace taichi::lang::spirv {

// Preserve the historical shader/runtime slot, including exported AOT code.
// Compiler-owned temporaries must skip this runtime-owned word.
inline constexpr uint32_t kRandomCounterWord = 1024;
inline constexpr std::size_t kRandomStateBegin =
    kRandomCounterWord * sizeof(uint32_t);
inline constexpr std::size_t kRandomStateEnd =
    kRandomStateBegin + sizeof(uint32_t);

}  // namespace taichi::lang::spirv
