#pragma once

#include <cstdint>
#include <string>

#include "taichi/rhi/arch.h"

namespace taichi::lang {

struct CompileConfig;
struct DeviceCapabilityConfig;
class Program;
class IRNode;
class SNode;
class Kernel;

// [P-Compile-2-A] Cache schema version, decoupled from TI_VERSION_*.
//
// Bump this constant whenever the offline cache key algorithm changes in a
// way that should invalidate previously written .tic files written by the
// same Taichi binary version (e.g. adding/removing a CompileConfig field
// from the key, changing the IR hash, adding device-cap fields, ...).
//
// Bumping forces every cache lookup to miss and silently fall back to
// recompile + rewrite. We do NOT bump TI_VERSION_PATCH for this because the
// metadata version check there serves a different purpose (binary-format
// compatibility of the metadata file itself).
//
// IMPORTANT — version 1 is special: it produces the SAME hash as the
// pre-P-Compile-2-A algorithm, i.e. no schema tag is mixed into the
// hasher. This preserves all existing .tic caches written by older builds
// of this fork. From version 2 onward we inject a schema tag so any key
// algo change is observable in the hash, and old v1 caches naturally miss
// without breaking anything.
//
// History:
//   1 - initial schema (2026-04, baseline before P-Compile-2-A landed).
//       Hash-equivalent to no schema versioning at all.
//   2 - P-Compile-1 phase 1 (2026-04). Adds CompileConfig::use_fused_passes
//       to the serialized key. With v>=2, a "tcs:N" schema tag is mixed
//       into the hasher, so all v1 caches naturally miss once. New cache
//       artifacts capture the use_fused_passes setting alongside other
//       config bits.
//   3 - CS-3.B/C (2026-05). LLVMRuntime listgen-reuse state changed from a
//       scalar dirty epoch to per-SNode arrays and list-version dependencies.
//       Old CUDA LLVM kernels embed LLVMRuntime field offsets, so reusing old
//       .tic files after the runtime layout change can launch kernels with
//       stale offsets. Force a cache miss and silent recompile.
//   4 - CS-3.D (2026-05). LLVM OffloadedTask serialization now carries sparse
//       listgen host-elision metadata (clear/listgen child/parent ids and
//       topology-mutation flag). Old cached kernels do not have the metadata,
//       so force a miss before host-side current-list launch skipping is used.
//   5 - CS-3.F (2026-05). OffloadedTask topology-mutation metadata is refined
//       from a boolean to an optional SNode id so CUDA host-side listgen reuse
//       can invalidate only the mutated list when safe. Old cached kernels lack
//       this id and would fall back to over-conservative/global behavior.
//   6 - VS-3 (2026-05). SPIR-V TaskAttributes serialization now carries
//       sparse listgen host-elision metadata (list child/parent ids and
//       topology-mutation id). Old cached Vulkan kernels lack the metadata,
//       so force a miss before GFX runtime listgen skipping is used.
//   7 - VS-2.3/G-3.1 (2026-05). SPIR-V BufferBind serialization now carries
//       conservative read/write access metadata for deferred dispatch barrier
//       coalescing. Old cached Vulkan kernels lack access bits, so force a
//       miss and recompile before runtime consumes this metadata.
//   8 - G-6 (2026-05). SPIR-V adaptive optimizer config was added to the
//       offline-cache key because it changes the per-task optimizer pass chain
//       and therefore emitted SPIR-V bytes.
constexpr std::uint32_t kOfflineCacheSchemaVersion = 8;

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel);
void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace taichi::lang
