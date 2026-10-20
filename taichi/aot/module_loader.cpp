#include "taichi/aot/module_loader.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/program/storage_view.h"
#include "picosha2.h"
#include <limits>

#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/dx12/aot_module_loader_impl.h"

namespace taichi::lang {
namespace aot {
DataType Module::get_ndarray_layout(const std::string &name) {
  if (is_corrupted())
    throw ArtifactLoadError("Cannot query layout from an invalid AOT module");
  if (!ndarray_layouts_loaded_) {
    try {
      ndarray_layouts_ = load_struct_layout_artifact(data_source());
    } catch (const std::exception &error) {
      throw ArtifactLoadError(error.what());
    }
    ndarray_layouts_loaded_ = true;
  }
  const auto found = ndarray_layouts_.find(name);
  if (found == ndarray_layouts_.end())
    throw std::invalid_argument("Unknown AOT ndarray layout: " + name);
  return found->second;
}

Module::BoundRootStorageView Module::get_root_storage_view(
    const std::string &name) {
  if (is_corrupted())
    throw ArtifactLoadError("Cannot bind storage from an invalid AOT module");
  const auto found = root_storage_views_.find(name);
  if (found == root_storage_views_.end())
    throw std::invalid_argument("Unknown AOT root storage view: " + name);
  const auto &view = found->second;
  const auto properties = view.properties();
  const auto begin = static_cast<std::uint64_t>(properties.reachable_begin);
  const auto bytes = static_cast<std::uint64_t>(properties.reachable_end) - begin;
  auto root = get_root_storage(view.tree_id);
  if (!root.pointer.device || begin > root.bytes ||
      bytes > root.bytes - begin ||
      begin >
          std::numeric_limits<std::uint64_t>::max() - root.pointer.offset)
    throw ArtifactLoadError(
        "AOT root storage view exceeds its module allocation");
  root.pointer.offset += begin;
  return {view, root.pointer, bytes, view.byte_offset - begin,
          properties.compact_contiguous && properties.ndarray_abi_compatible};
}

namespace {

std::string make_kernel_key(
    const std::vector<KernelTemplateArg> &template_args) {
  TI_NOT_IMPLEMENTED;
  return "";
}

}  // namespace

Kernel *KernelTemplate::get_kernel(
    const std::vector<KernelTemplateArg> &template_args) {
  const auto key = make_kernel_key(template_args);
  auto itr = loaded_kernels_.find(key);
  if (itr != loaded_kernels_.end()) {
    return itr->second.get();
  }
  auto k = make_new_kernel(template_args);
  auto *kptr = k.get();
  loaded_kernels_[key] = std::move(k);
  return kptr;
}

std::unique_ptr<Module> Module::load(Arch arch, std::any mod_params) {
  if (arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    return gfx::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::opengl) {
#ifdef TI_WITH_OPENGL
    return gfx::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::gles) {
#ifdef TI_WITH_OPENGL
    return gfx::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::dx11) {
#ifdef TI_WITH_DX11
    return gfx::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::dx12) {
#ifdef TI_WITH_DX12
    return directx12::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::metal) {
#ifdef TI_WITH_METAL
    return gfx::make_aot_module(mod_params, arch);
#endif
  }
  TI_NOT_IMPLEMENTED;
}

Kernel *Module::get_kernel(const std::string &name) {
  if (is_corrupted())
    throw ArtifactLoadError("Cannot load a kernel from an invalid AOT module");
  auto itr = loaded_kernels_.find(name);
  if (itr != loaded_kernels_.end()) {
    return itr->second.get();
  }
  auto k = make_new_kernel(name);
  if (k) {
    try {
      if (!value_parameters_loaded_) {
        value_parameters_ = load_value_parameter_artifact(data_source());
        value_parameters_loaded_ = true;
      }
      const auto values = value_parameters_.find(name);
      if (values != value_parameters_.end()) {
        for (const auto &[indices, logical] : values->second) {
          const auto parameter = k->nested_parameters.find(indices);
          if (parameter == k->nested_parameters.end() ||
              parameter->second.is_array || parameter->second.is_argpack ||
              parameter->second.ptype != ParameterType::kScalar)
            throw std::invalid_argument("Value parameter does not name a pure value");
          // GFX ArgAttributes can contain an unknown dtype for aggregates.
          // The compiled args_type remains the authoritative physical ABI.
          validate_value_parameter_type(
              logical, compiled_value_parameter_type(*k, indices));
        }
        k->value_parameter_types = values->second;
      }
    } catch (const std::exception &error) {
      throw ArtifactLoadError(error.what());
    }
  }
  auto *kptr = k.get();
  loaded_kernels_[name] = std::move(k);
  return kptr;
}

std::vector<std::uint8_t> Module::load_data(
    const std::string &sha256,
    std::uint64_t expected_bytes) const {
  return load_data_from(data_source(), sha256, expected_bytes);
}

std::vector<std::uint8_t> Module::load_data_from(const io::VirtualDir *source,
                                                 const std::string &sha256,
                                                 std::uint64_t expected_bytes) {
  if (sha256.size() != 64 ||
      sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw ArtifactLoadError("Invalid AOT data payload key");
  const auto path = "data/" + sha256 + ".bin";
  std::size_t size = 0;
  if (!source || !source->get_file_size(path, size) || size != expected_bytes)
    throw ArtifactLoadError(
        "AOT data payload is missing or has an incompatible size: " + sha256);
  std::vector<std::uint8_t> bytes(size);
  if (source->load_file(path, bytes.data(), size) != size ||
      picosha2::hash256_hex_string(bytes) != sha256)
    throw ArtifactLoadError("AOT data payload failed integrity validation: " +
                            sha256);
  return bytes;
}

void Module::initialize_root_data(
    const InitialRootData &entry,
    DevicePtr destination,
    const std::vector<std::uint8_t> *preloaded) const {
  std::vector<std::uint8_t> loaded;
  if (!preloaded) {
    loaded = load_data(entry.sha256, entry.size);
    preloaded = &loaded;
  }
  TI_ASSERT(preloaded->size() == entry.size);
  if (preloaded->empty())
    return;
  const void *source = preloaded->data();
  size_t bytes = preloaded->size();
  const auto status =
      destination.device->upload_data(&destination, &source, &bytes, 1);
  if (status != RhiResult::success)
    throw ArtifactLoadError("Failed to initialize AOT root " +
                            std::to_string(entry.tree_id));
}

KernelTemplate *Module::get_kernel_template(const std::string &name) {
  auto itr = loaded_kernel_templates_.find(name);
  if (itr != loaded_kernel_templates_.end()) {
    return itr->second.get();
  }
  auto kt = make_new_kernel_template(name);
  auto *kt_ptr = kt.get();
  loaded_kernel_templates_[name] = std::move(kt);
  return kt_ptr;
}

Field *Module::get_snode_tree(const std::string &name) {
  auto itr = loaded_fields_.find(name);
  if (itr != loaded_fields_.end()) {
    return itr->second.get();
  }
  auto k = make_new_field(name);
  auto *kptr = k.get();
  loaded_fields_[name] = std::move(k);
  return kptr;
}

}  // namespace aot
}  // namespace taichi::lang
