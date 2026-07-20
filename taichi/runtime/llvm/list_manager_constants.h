#pragma once

#include <cstddef>

// Keep the common case entirely inline while retaining the legacy 128K-chunk
// capacity through grow-on-demand directory pages.
constexpr std::size_t kLlvmListManagerMaxNumChunks = 128UL << 10;
constexpr std::size_t kLlvmListManagerInlineChunks = 16;
constexpr std::size_t kLlvmListManagerChunksPerDirectory = 1024;
constexpr std::size_t kLlvmListManagerDirectoryCount =
    (kLlvmListManagerMaxNumChunks - kLlvmListManagerInlineChunks +
     kLlvmListManagerChunksPerDirectory - 1) /
    kLlvmListManagerChunksPerDirectory;
constexpr std::size_t kLlvmListManagerDirectoryPageBytes =
    kLlvmListManagerChunksPerDirectory * sizeof(void *);
constexpr std::size_t kLlvmListManagerAllocationAlignment = 4096;
constexpr std::size_t kLlvmElementListMinChunkElements = 64;
constexpr std::size_t kLlvmElementListMaxChunkElements = 64UL << 10;
// Inclusive power-of-two classes: 64, 128, ..., 65536.
constexpr std::size_t kLlvmElementListChunkSizeClasses = 11;

// Keep a small amount of destroyed CPU sparse payload hot for fast SNodeTree
// reconstruction, but do not let large transient trees permanently pin the
// process-wide HostMemoryPool. The budget covers ListManager chunks and
// directories plus direct ambient allocations; manager objects remain reusable.
constexpr std::size_t kLlvmHostSparseRecycledPayloadBudgetBytes = 16UL << 20;

// Host-side CUDA pool sizing cannot use sizeof(ListManager), because the
// runtime class is compiled into backend bitcode. This mirrors its two pointer
// arrays and leaves one allocation-alignment unit for the small POD tail and
// alignment padding.
constexpr std::size_t kLlvmListManagerFixedAllocationBudgetBytes =
    (kLlvmListManagerInlineChunks + kLlvmListManagerDirectoryCount) *
        sizeof(void *) +
    kLlvmListManagerAllocationAlignment;
constexpr std::size_t kLlvmListManagerDirectoryAllocationBudgetBytes =
    kLlvmListManagerDirectoryPageBytes +
    kLlvmListManagerAllocationAlignment;

constexpr std::size_t llvm_list_manager_directory_pages_for_chunks(
    std::size_t chunks) {
  return chunks <= kLlvmListManagerInlineChunks
             ? 0
             : (chunks - kLlvmListManagerInlineChunks +
                kLlvmListManagerChunksPerDirectory - 1) /
                   kLlvmListManagerChunksPerDirectory;
}
