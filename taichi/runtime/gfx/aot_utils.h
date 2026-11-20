#pragma once

#include <vector>
#include <map>

#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/aot/module_loader.h"
#include "taichi/aot/device_range_dispatch.h"
#include "taichi/runtime/gfx/aot_snode_resources.h"

namespace taichi::lang {
namespace gfx {

struct AotKernelMetadata {
  std::size_t num_snode_trees{0};
  std::vector<int> used_snode_tree_ids;

  TI_IO_DEF(num_snode_trees, used_snode_tree_ids);
};

struct AotKernelAlias {
  std::string name;
  std::size_t kernel_index{0};

  TI_IO_DEF(name, kernel_index);
};

// Position in the complete shader. The containing module retains its own
// declarations, decorations, constants and descriptor ABI.
struct AotShaderFunction {
  size_t offset_words{0};
  size_t size_words{0};
  std::string sha256;
  // A nonempty base makes sha256 the digest of the per-word differences.
  std::string base_sha256;
  // A compact relocation stores a bit per word, followed by the nonzero deltas.
  size_t relocation_bytes{0};
  // A single-use relocation follows the preceding instructions in the root.
  bool relocation_in_root{false};
  TI_IO_DEF(offset_words, size_words, sha256, base_sha256, relocation_bytes,
            relocation_in_root);
};

// Entry-specific execution policy is separate from shared shader identity.
using AotDeviceRangeDispatch = aot::DeviceRangeDispatch;

// Export names do not affect execution. Everything else, including task names,
// ABI, launch attributes and tree dependencies, must agree before sharing.
inline std::string kernel_identity(spirv::TaichiKernelAttributes attributes,
                                   const AotKernelMetadata &metadata) {
  attributes.name.clear();
  return liong::json::print(liong::json::serialize(attributes)) +
         liong::json::print(liong::json::serialize(metadata));
}

/**
 * AOT module data for the Unified Device API backend.
 */
struct TaichiAotData {
  // v6 reconstructs sparse resources; v7 adds descriptor ndarray byte offsets;
  // v8 records exact cooperative-matrix dimensions and fixed subgroup width.
  // v9 stores signed, typed range bounds shared with LLVM compiled tasks.
  // v10 preserves the final-IR device-range dispatch qualification.
  // v11 persists per-entry range recording selection without duplicating code.
  // v12 carries host-argument launch bounds with each qualified task.
  // v13 adds affine bounds and their intermediate-overflow input interval.
  // v16 preserves source-owned block geometry for cold tuning admission.
  // v17 retains compiler-proven affine array arguments, checked against SPIR-V.
  // v18 binds model workgroup geometry and shared allocations per pipeline.
  // v19 allows logical tasks to reference an identical stored shader.
  // v20 identifies distinct shaders that have the same logical task name.
  // v21 stores identical SPIR-V function bodies once across task shaders.
  // v22 can store the operands of same-shape functions relative to one base.
  // v23 extends single-hop sharing to complete instruction ranges.
  // v24 can omit unchanged operands and keep a single-use relocation in its root.
  static constexpr uint32_t kMetadataVersion = 24;
  static std::string indexed_shader(size_t kernel, size_t task) {
    return "shader_blobs/" + std::to_string(kernel) + "_" + std::to_string(task);
  }

  uint32_t metadata_version{0};
  //   BufferMetaData metadata;
  std::vector<std::vector<std::vector<uint32_t>>> spirv_codes;
  std::vector<spirv::TaichiKernelAttributes> kernels;
  std::vector<AotKernelMetadata> kernel_metadata;
  std::vector<AotKernelAlias> kernel_aliases;
  std::map<std::string, std::string> shader_aliases;
  std::map<size_t, std::map<size_t, std::string>> task_shader_sources;
  std::map<std::string, std::vector<AotShaderFunction>> shader_functions;
  std::vector<AotDeviceRangeDispatch> device_range_dispatch;
  std::vector<aot::CompiledFieldData> fields;
  std::map<std::string, uint32_t> required_caps;
  // root_buffer_size remains the first-tree compatibility view. New loaders
  // consume root_buffer_sizes so every artifact-local tree has an explicit
  // layout allocation.
  size_t root_buffer_size{0};
  std::vector<size_t> root_buffer_sizes;
  std::vector<aot::InitialRootData> initial_root_data;
  std::vector<AotSNodeTreeResources> snode_tree_resources;

  TI_IO_DEF(metadata_version,
            kernels,
            kernel_metadata,
            kernel_aliases,
            shader_aliases,
            task_shader_sources,
            shader_functions,
            device_range_dispatch,
            fields,
            required_caps,
            root_buffer_size,
            root_buffer_sizes,
            initial_root_data,
            snode_tree_resources);
};

}  // namespace gfx
}  // namespace taichi::lang
