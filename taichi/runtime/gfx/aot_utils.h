#pragma once

#include <vector>
#include <map>

#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/aot/module_loader.h"

namespace taichi::lang {
namespace gfx {

struct AotKernelMetadata {
  std::size_t num_snode_trees{0};
  std::vector<int> used_snode_tree_ids;

  TI_IO_DEF(num_snode_trees, used_snode_tree_ids);
};

/**
 * AOT module data for the Unified Device API backend.
 */
struct TaichiAotData {
  static constexpr uint32_t kMetadataVersion = 1;

  uint32_t metadata_version{0};
  //   BufferMetaData metadata;
  std::vector<std::vector<std::vector<uint32_t>>> spirv_codes;
  std::vector<spirv::TaichiKernelAttributes> kernels;
  std::vector<AotKernelMetadata> kernel_metadata;
  std::vector<aot::CompiledFieldData> fields;
  std::map<std::string, uint32_t> required_caps;
  // root_buffer_size remains the first-tree compatibility view. New loaders
  // consume root_buffer_sizes so every artifact-local tree has an explicit
  // layout allocation.
  size_t root_buffer_size{0};
  std::vector<size_t> root_buffer_sizes;

  TI_IO_DEF(metadata_version,
            kernels,
            kernel_metadata,
            fields,
            required_caps,
            root_buffer_size,
            root_buffer_sizes);
};

}  // namespace gfx
}  // namespace taichi::lang
