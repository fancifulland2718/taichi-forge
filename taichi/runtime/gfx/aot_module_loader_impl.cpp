#include "taichi/runtime/gfx/aot_module_loader_impl.h"

#include <algorithm>
#include <type_traits>

#include "taichi/runtime/gfx/runtime.h"
#include "taichi/aot/graph_data.h"

namespace taichi::lang {
namespace gfx {
namespace {
class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(GfxRuntime *runtime, const aot::CompiledFieldData &field)
      : runtime_(runtime), field_(field) {
  }

 private:
  GfxRuntime *const runtime_;
  aot::CompiledFieldData field_;
};

class AotModuleImpl : public aot::Module {
 public:
  explicit AotModuleImpl(const AotModuleParams &params, Arch device_api_backend)
      : source_(params.source),
        runtime_(params.runtime),
        device_api_backend_(device_api_backend) {
    if (!source_ && params.dir == nullptr) {
      source_ = io::VirtualDir::from_fs_dir(params.module_path);
    }
    const io::VirtualDir *dir = source_ ? source_.get() : params.dir;

    {
      std::vector<uint8_t> metadata_json{};
      bool succ = dir->load_file("metadata.json", metadata_json) != 0;

      if (!succ) {
        mark_corrupted();
        TI_WARN("'metadata.json' cannot be read");
        return;
      }
      try {
        auto json = liong::json::parse(
            (const char *)metadata_json.data(),
            (const char *)(metadata_json.data() + metadata_json.size()));
        liong::json::deserialize(json, ti_aot_data_);
        for (const auto &[name, level] : ti_aot_data_.required_caps) {
          required_caps_.set(str2devcap(name), level);
        }
      } catch (const std::exception &e) {
        mark_corrupted();
        TI_WARN("Invalid GFX AOT metadata: {}", e.what());
        return;
      }
    }

    if (ti_aot_data_.metadata_version == 0) {
      // Backward-compatible view for legacy single-tree artifacts.
      ti_aot_data_.root_buffer_sizes = {ti_aot_data_.root_buffer_size};
      if (!ti_aot_data_.kernel_metadata.empty()) {
        mark_corrupted();
        TI_WARN("Legacy GFX AOT artifact contains unexpected kernel metadata");
        return;
      }
      ti_aot_data_.kernel_metadata.resize(
          ti_aot_data_.kernels.size(),
          AotKernelMetadata{/*num_snode_trees=*/1,
                            /*used_snode_tree_ids=*/{0}});
    } else if (ti_aot_data_.metadata_version !=
               TaichiAotData::kMetadataVersion) {
      mark_corrupted();
      TI_WARN("Unsupported GFX AOT metadata version {}",
              ti_aot_data_.metadata_version);
      return;
    } else {
      const size_t compatibility_root_size =
          ti_aot_data_.root_buffer_sizes.empty()
              ? 0
              : ti_aot_data_.root_buffer_sizes.front();
      if (ti_aot_data_.root_buffer_size != compatibility_root_size) {
        mark_corrupted();
        TI_WARN("GFX AOT first-root compatibility size is inconsistent");
        return;
      }
    }
    if (ti_aot_data_.kernel_metadata.size() != ti_aot_data_.kernels.size()) {
      mark_corrupted();
      TI_WARN("GFX AOT kernel metadata count does not match kernel count");
      return;
    }
    for (std::size_t i = 0; i < ti_aot_data_.kernel_metadata.size(); ++i) {
      const auto &metadata = ti_aot_data_.kernel_metadata[i];
      if (metadata.num_snode_trees != ti_aot_data_.root_buffer_sizes.size()) {
        mark_corrupted();
        TI_WARN("GFX AOT kernel {} has an inconsistent SNodeTree count", i);
        return;
      }
      if (!std::is_sorted(metadata.used_snode_tree_ids.begin(),
                          metadata.used_snode_tree_ids.end()) ||
          std::adjacent_find(metadata.used_snode_tree_ids.begin(),
                             metadata.used_snode_tree_ids.end()) !=
              metadata.used_snode_tree_ids.end() ||
          std::any_of(metadata.used_snode_tree_ids.begin(),
                      metadata.used_snode_tree_ids.end(), [this](int tree_id) {
                        return tree_id < 0 ||
                               static_cast<std::size_t>(tree_id) >=
                                   ti_aot_data_.root_buffer_sizes.size();
                      })) {
        mark_corrupted();
        TI_WARN("GFX AOT kernel {} has invalid SNodeTree dependencies", i);
        return;
      }
    }
    for (const auto &field : ti_aot_data_.fields) {
      if (field.snode_tree_id < 0 ||
          static_cast<std::size_t>(field.snode_tree_id) >=
              ti_aot_data_.root_buffer_sizes.size()) {
        mark_corrupted();
        TI_WARN("GFX AOT field '{}' has invalid SNodeTree id {}",
                field.field_name, field.snode_tree_id);
        return;
      }
    }

    shader_codes_.resize(ti_aot_data_.kernels.size());
    shader_errors_.resize(ti_aot_data_.kernels.size());
    for (size_t i = 0; i < ti_aot_data_.kernels.size(); ++i) {
      // Legacy exporters may repeat a kernel shared by multiple Graphs.
      // Preserve their first-match lookup behavior without repeated scans.
      kernel_indices_.emplace(ti_aot_data_.kernels[i].name, i);
      if (!source_ && !load_shader(*dir, i)) {
        mark_corrupted();
        TI_WARN("{}", shader_errors_[i]);
        return;
      }
    }

    {
      std::vector<uint8_t> graphs_json{};
      bool succ = dir->load_file("graphs.json", graphs_json) != 0;

      if (!succ) {
        mark_corrupted();
        TI_WARN("'graphs.json' cannot be read");
        return;
      }

      try {
        auto json = liong::json::parse(
            (const char *)graphs_json.data(),
            (const char *)(graphs_json.data() + graphs_json.size()));
        liong::json::deserialize(json, graphs_);
      } catch (const std::exception &e) {
        mark_corrupted();
        TI_WARN("Invalid GFX AOT graph metadata: {}", e.what());
        return;
      }
    }
  }

  std::unique_ptr<aot::CompiledGraph> get_graph(
      const std::string &name) override {
    auto it = graphs_.find(name);
    if (it == graphs_.end()) {
      TI_DEBUG("Cannot find graph {}", name);
      return nullptr;
    }

    std::vector<aot::CompiledDispatch> dispatches;
    for (auto &dispatch : it->second.dispatches) {
      auto *kernel = get_kernel(dispatch.kernel_name);
      if (kernel == nullptr) {
        throw aot::ArtifactLoadError("Graph references missing kernel: " +
                                     dispatch.kernel_name);
      }
      dispatches.push_back(
          {dispatch.kernel_name, {}, dispatch.symbolic_args, kernel});
    }
    aot::CompiledGraph graph{dispatches};
    return std::make_unique<aot::CompiledGraph>(std::move(graph));
  }

  size_t get_root_size() const override {
    return ti_aot_data_.root_buffer_size;
  }
  std::vector<size_t> get_root_sizes() const override {
    return ti_aot_data_.root_buffer_sizes;
  }

  const DeviceCapabilityConfig &get_required_caps() const override {
    return required_caps_;
  }

  // Module metadata
  Arch arch() const override {
    return device_api_backend_;
  }
  uint64_t version() const override {
    TI_NOT_IMPLEMENTED;
  }

 private:
  bool get_field_data_by_name(const std::string &name,
                              aot::CompiledFieldData &field) {
    for (int i = 0; i < ti_aot_data_.fields.size(); ++i) {
      if (ti_aot_data_.fields[i].field_name.rfind(name, 0) == 0) {
        field = ti_aot_data_.fields[i];
        return true;
      }
    }
    return false;
  }

  bool get_kernel_params_by_name(const std::string &name,
                                 GfxRuntime::RegisterParams &kernel) {
    auto found = kernel_indices_.find(name);
    if (found != kernel_indices_.end()) {
      const size_t i = found->second;
      if (source_ && !load_shader(*source_, i)) {
        throw aot::ArtifactLoadError(shader_errors_[i]);
      }
      kernel.kernel_attribs = ti_aot_data_.kernels[i];
      kernel.shared_task_spirv_source_codes = shader_codes_[i];
      kernel.num_snode_trees = ti_aot_data_.kernel_metadata[i].num_snode_trees;
      kernel.snode_tree_ids =
          ti_aot_data_.kernel_metadata[i].used_snode_tree_ids;
      return true;
    }
    return false;
  }

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override {
    GfxRuntime::RegisterParams kparams;
    if (!get_kernel_params_by_name(name, kparams)) {
      TI_DEBUG("Failed to load kernel {}", name);
      return nullptr;
    }
    return std::make_unique<KernelImpl>(runtime_, std::move(kparams));
  }

  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(
      const std::string &name) override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }

  std::unique_ptr<aot::Field> make_new_field(const std::string &name) override {
    aot::CompiledFieldData field;
    if (!get_field_data_by_name(name, field)) {
      TI_DEBUG("Failed to load field {}", name);
      return nullptr;
    }
    return std::make_unique<FieldImpl>(runtime_, field);
  }

  bool load_shader(const io::VirtualDir &dir, size_t index) {
    if (shader_codes_[index])
      return true;
    if (!shader_errors_[index].empty())
      return false;
    std::vector<std::vector<uint32_t>> codes;
    for (const auto &task : ti_aot_data_.kernels[index].tasks_attribs) {
      const std::string path = task.name + ".spv";
      std::vector<uint32_t> code;
      if (!dir.load_file(path, code)) {
        shader_errors_[index] = "Cannot read SPIR-V payload: " + path;
        return false;
      }
      if (code.size() < 5 || code[0] != 0x07230203) {
        shader_errors_[index] = "Invalid SPIR-V header: " + path;
        return false;
      }
      codes.emplace_back(std::move(code));
    }
    shader_codes_[index] =
        std::make_shared<const std::vector<std::vector<uint32_t>>>(
            std::move(codes));
    return true;
  }

  std::shared_ptr<const io::VirtualDir> source_;
  TaichiAotData ti_aot_data_;
  DeviceCapabilityConfig required_caps_;
  std::vector<std::shared_ptr<const std::vector<std::vector<uint32_t>>>>
      shader_codes_;
  std::vector<std::string> shader_errors_;
  std::unordered_map<std::string, size_t> kernel_indices_;
  GfxRuntime *runtime_{nullptr};
  Arch device_api_backend_;
};

}  // namespace

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params,
                                             Arch device_api_backend) {
  AotModuleParams params = std::any_cast<AotModuleParams &>(mod_params);
  return std::make_unique<AotModuleImpl>(params, device_api_backend);
}

}  // namespace gfx
}  // namespace taichi::lang
