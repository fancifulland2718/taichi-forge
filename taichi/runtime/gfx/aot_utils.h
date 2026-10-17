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

struct AotKernelAlias {
  std::string name;
  std::size_t kernel_index{0};

  TI_IO_DEF(name, kernel_index);
};

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
  // v5 qualifies immutable sampled-image collection arguments. Existing
  // scalar/array and storage-image artifacts keep their older minimum version.
  static constexpr uint32_t kMetadataVersion = 5;

  uint32_t metadata_version{0};
  //   BufferMetaData metadata;
  std::vector<std::vector<std::vector<uint32_t>>> spirv_codes;
  std::vector<spirv::TaichiKernelAttributes> kernels;
  std::vector<AotKernelMetadata> kernel_metadata;
  std::vector<AotKernelAlias> kernel_aliases;
  std::vector<aot::CompiledFieldData> fields;
  std::map<std::string, uint32_t> required_caps;
  // root_buffer_size remains the first-tree compatibility view. New loaders
  // consume root_buffer_sizes so every artifact-local tree has an explicit
  // layout allocation.
  size_t root_buffer_size{0};
  std::vector<size_t> root_buffer_sizes;
  std::vector<aot::InitialRootData> initial_root_data;

  TI_IO_DEF(metadata_version,
            kernels,
            kernel_metadata,
            kernel_aliases,
            fields,
            required_caps,
            root_buffer_size,
            root_buffer_sizes,
            initial_root_data);
};

}  // namespace gfx
}  // namespace taichi::lang
