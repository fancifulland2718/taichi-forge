#pragma once

// Shared result-buffer layout for the LLVM runtime-module diagnostic query.
// Keep this header POD-only: runtime.cpp is compiled into backend bitcode.
enum LlvmSparseTreeStatisticIndex {
  kLlvmSparseRuntimeMetadataRequestedBytes = 0,
  kLlvmSparseAllocatorPayloadReservedBytes,
  kLlvmSparseAllocatorPayloadUsedBytes,
  kLlvmSparseAllocatorBookkeepingReservedBytes,
  kLlvmSparseActiveListReservedBytes,
  kLlvmSparseActiveListUsedBytes,
  kLlvmSparseAllocatorInUseElements,
  kLlvmSparseAllocatorFreeElements,
  kLlvmSparseAllocatorRecycledElements,
  kLlvmSparseTreeStatisticCount,
};
