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
constexpr std::uint32_t kOfflineCacheSchemaVersion = 2;

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel);
void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace taichi::lang
