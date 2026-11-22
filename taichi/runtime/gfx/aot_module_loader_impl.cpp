#include "taichi/runtime/gfx/aot_module_loader_impl.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <set>
#include <type_traits>
#include <unordered_set>

#include "external/PicoSHA2/picosha2.h"
#include "spirv/unified1/spirv.hpp"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/aot_metadata_rows.h"
#include "taichi/runtime/gfx/descriptor_binding.h"
#include "taichi/runtime/gfx/range_dispatch.h"
#include "taichi/runtime/gfx/cooperative_matrix.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/aot/specialization.h"
#include "taichi/aot/word_plane_storage.h"
#include "taichi/rhi/vulkan/shader_specialization.h"
#include "taichi/codegen/spirv/array_layout_abi.h"

namespace taichi::lang {
namespace gfx {
#ifdef TI_WITH_VULKAN
size_t specialize_task_arguments(
    GfxRuntime::RegisterParams &params,
    const std::vector<vulkan::ShaderArgumentConstant> &values,
    size_t max_workers,
    bool layout_words_only) {
  if (values.empty())
    return 0;
  const auto &tasks = params.kernel_attribs.tasks_attribs;
  auto &codes = params.task_spirv_source_codes;
  auto run_task = [&](size_t i) {
    size_t folded = 0;
    for (const auto &bind : tasks[i].buffer_binds) {
      if (bind.buffer.type == TaskAttributes::BufferType::Args && bind.binding >= 0) {
        size_t count = 0;
        const auto bind_arguments = layout_words_only ? vulkan::bind_spirv_layout_words
                                                     : vulkan::specialize_spirv_arguments;
        codes[i] = bind_arguments(codes[i], bind.binding, values, &count);
        folded += count;
      }
    }
    return folded;
  };

  // Only overlap independent large IR contexts. Light shaders and kernels
  // dominated by one large shader get the ordinary sequential pass.
  size_t large = 0;
  if (max_workers > 1 && tasks.size() > 1) {
    constexpr size_t min_parallel_words = 64 * 1024;
    for (size_t i = 0; i < tasks.size(); ++i) {
      const auto &binds = tasks[i].buffer_binds;
      if (codes[i].size() >= min_parallel_words &&
          std::any_of(binds.begin(), binds.end(), [](const auto &bind) {
            return bind.buffer.type == TaskAttributes::BufferType::Args && bind.binding >= 0;
          }))
        ++large;
    }
  }
  size_t folded = 0;
  const auto workers = std::min(max_workers, large);
  if (workers < 2) {
    for (size_t i = 0; i < tasks.size(); ++i)
      folded += run_task(i);
    return folded;
  }
  // Start the longest jobs first. Each task owns its original output slot,
  // and registration sees the results only after every worker succeeds.
  std::vector<size_t> order(tasks.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return codes[a].size() > codes[b].size();
  });
  std::atomic<size_t> next{0};
  auto run = [&] {
    size_t total = 0;
    for (;;) {
      const auto n = next.fetch_add(1, std::memory_order_relaxed);
      if (n >= order.size())
        return total;
      total += run_task(order[n]);
    }
  };
  std::vector<std::future<size_t>> work;
  for (size_t i = 1; i < workers; ++i)
    work.push_back(std::async(std::launch::async | std::launch::deferred, run));
  folded = run();
  for (auto &worker : work)
    folded += worker.get();
  return folded;
}
#endif

namespace {
class SNodeInstanceImpl final : public AotSNodeInstance {
 public:
  SNodeInstanceImpl(std::shared_ptr<aot::Module> code, GfxRuntime *runtime)
      : code_(std::move(code)), runtime_(runtime) {
  }

  ~SNodeInstanceImpl() override {
    try {
      runtime_->release_aot_resources(registrations_, allocated_roots_);
    } catch (...) {
      // A failed drain cannot justify recycling state or releasing the code
      // owner. The Runtime retains the unretired registrations and resources.
      (void)new std::shared_ptr<aot::Module>(std::move(code_));
    }
  }

  KernelImpl *get_kernel(const std::string &name) const override {
    const auto found = kernels_.find(name);
    if (found == kernels_.end())
      throw std::invalid_argument(
          "Kernel is outside the SNode instance's selected closure: " + name);
    return found->second.get();
  }

  const std::vector<int> &root_bindings() const override {
    return root_bindings_;
  }

  std::shared_ptr<aot::Module> code_;
  GfxRuntime *runtime_;
  std::vector<int> allocated_roots_, root_bindings_;
  std::vector<GfxRuntime::KernelHandle> registrations_;
  std::unordered_map<std::string, std::unique_ptr<KernelImpl>> kernels_;
};

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
    if (dir)
      root_storage_views_ = aot::load_root_storage_views(*dir);

    {
      std::vector<uint8_t> metadata_json{};
      bool succ = dir->load_file("metadata.json", metadata_json) != 0;

      if (!succ) {
        mark_corrupted();
        TI_WARN("'metadata.json' cannot be read");
        return;
      }
      try {
        const auto *begin = reinterpret_cast<const char *>(metadata_json.data());
        read_kernel_metadata(begin, begin + metadata_json.size(), ti_aot_data_,
                             device_api_backend_ == Arch::vulkan);
        if (!ti_aot_data_.shader_aliases.empty() || !ti_aot_data_.task_shader_sources.empty() ||
            !ti_aot_data_.shader_functions.empty()) {
          if (ti_aot_data_.metadata_version < 19)
            throw std::invalid_argument("Shader aliases require GFX AOT metadata v19");
          std::unordered_set<std::string> names;
          for (const auto &kernel : ti_aot_data_.kernels)
            for (const auto &task : kernel.tasks_attribs)
              names.insert(task.name);
          auto sources = names;
          for (const auto &[index, tasks] : ti_aot_data_.task_shader_sources) {
            if (ti_aot_data_.metadata_version < 20 || index >= ti_aot_data_.kernels.size())
              throw std::invalid_argument("Invalid GFX AOT shader source kernel");
            for (const auto &[task, source] : tasks) {
              if (task >= ti_aot_data_.kernels[index].tasks_attribs.size())
                throw std::invalid_argument("Invalid GFX AOT shader source task");
              if (source == TaichiAotData::indexed_shader(index, task))
                sources.insert(source);
            }
          }
          const auto valid_source = [&](const std::string &source) {
            return sources.count(source) && !ti_aot_data_.shader_aliases.count(source);
          };
          for (const auto &[alias, source] : ti_aot_data_.shader_aliases) {
            if (!names.count(alias) || !valid_source(source))
              throw std::invalid_argument("Invalid GFX AOT shader alias");
          }
          for (const auto &[_, tasks] : ti_aot_data_.task_shader_sources)
            for (const auto &[task, source] : tasks)
              if (!valid_source(source))
                throw std::invalid_argument("Invalid GFX AOT shader source");
          for (const auto &[name, refs] : ti_aot_data_.shader_functions) {
            if (!valid_source(name) || refs.empty() || ti_aot_data_.metadata_version < 21 ||
                device_api_backend_ != Arch::vulkan)
              throw std::invalid_argument("Invalid Vulkan shared shader function source");
          }
        }
        for (const auto &kernel : ti_aot_data_.kernels) {
          if (ti_aot_data_.metadata_version < 17 && !kernel.runtime_affine_arrays.empty())
            throw std::invalid_argument("Legacy GFX metadata cannot declare affine array arguments");
          for (const auto &task : kernel.tasks_attribs) {
            if ((task.has_model_workgroup() &&
                 ti_aot_data_.metadata_version < 18) ||
                !task.valid_model_workgroup(kernel.ctx_attribs.args_bytes()))
              throw std::invalid_argument(
                  "Invalid Vulkan model workgroup contract");
            if (!task.valid_device_range_dispatch_contract() ||
                (ti_aot_data_.metadata_version < 10 &&
                 task.device_range_dispatch_version != 0)) {
              throw std::invalid_argument("Unsupported device range dispatch proof");
            }
            if (ti_aot_data_.metadata_version >= 9 && task.range_for_attribs &&
                !task.range_for_attribs->valid(kernel.ctx_attribs.args_bytes())) {
              throw std::invalid_argument("Invalid typed task range bounds");
            }
          }
        }
        for (const auto &[name, level] : ti_aot_data_.required_caps) {
          required_caps_.set(str2devcap(name), level);
        }
      } catch (const std::exception &e) {
        mark_corrupted();
        TI_WARN("Invalid GFX AOT metadata: {}", e.what());
        return;
      }
    }

    if (ti_aot_data_.metadata_version < 2 &&
        !ti_aot_data_.kernel_aliases.empty()) {
      mark_corrupted();
      TI_WARN("Legacy GFX AOT artifact contains unexpected kernel aliases");
      return;
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
    } else if (ti_aot_data_.metadata_version >
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
    TI_ERROR_IF((ti_aot_data_.metadata_version < 3 &&
                 !ti_aot_data_.initial_root_data.empty()) ||
                    (ti_aot_data_.metadata_version == 3 &&
                     ti_aot_data_.initial_root_data.empty()),
                "AOT initial root data requires compatible GFX metadata.");
    aot::validate_initial_root_data(ti_aot_data_.initial_root_data,
                                    ti_aot_data_.root_buffer_sizes);
    SNodeResourceTreeMap resource_trees;
    try {
      const auto &resources = ti_aot_data_.snode_tree_resources;
      if (ti_aot_data_.metadata_version < 6) {
        if (!resources.empty())
          throw std::invalid_argument(
              "Sparse resource metadata requires GFX v6");
      } else if (ti_aot_data_.metadata_version == 6 || !resources.empty()) {
        if (resources.size() != ti_aot_data_.root_buffer_sizes.size())
          throw std::invalid_argument(
              "GFX resource count does not match root layouts");
        std::unordered_set<int> producer_ids;
        for (std::size_t i = 0; i < resources.size(); ++i) {
          const auto &entry = resources[i];
          entry.validate();
          if (entry.resources.root_size != ti_aot_data_.root_buffer_sizes[i] ||
              !producer_ids.insert(entry.producer_tree_id).second)
            throw std::invalid_argument(
                "GFX resource layout or producer identity is inconsistent");
          resource_trees.emplace(int(i), entry.resources.snode_ids);
        }
        auto require_plain = [&](int tree_id) {
          if (tree_id < 0 || std::size_t(tree_id) >= resources.size() ||
              resources[tree_id].storage_kind !=
                  AotSNodeTreeResources::StorageKind::PlainDense)
            throw std::invalid_argument(
                "AOT root snapshots and flat views require plain dense "
                "storage");
        };
        for (const auto &entry : ti_aot_data_.initial_root_data)
          require_plain(entry.tree_id);
        for (const auto &[name, view] : root_storage_views_)
          require_plain(view.tree_id);
      }
    } catch (const std::exception &error) {
      mark_corrupted();
      TI_WARN("Invalid GFX AOT SNode resources: {}", error.what());
      return;
    }
    if (!source_) {
      // The legacy borrowed VirtualDir need only live through construction.
      // Keep these verified bytes only until roots are allocated, like eager
      // shader loading on the same legacy path.
      for (const auto &entry : ti_aot_data_.initial_root_data) {
        eager_initial_root_data_.push_back(
            load_data_from(dir, entry.sha256, entry.size));
      }
    }
    for (std::size_t i = 0; i < ti_aot_data_.kernel_metadata.size(); ++i) {
      const auto &context = ti_aot_data_.kernels[i].ctx_attribs;
      if (context.descriptor_array_offsets &&
          (ti_aot_data_.metadata_version < 7 || !context.args_type() ||
           context.args_type()->get_layout() != "1-" ||
           std::none_of(
               context.args().begin(), context.args().end(),
               [](const auto &arg) {
                 return arg.second.ptype == ParameterType::kNdarray;
               }))) {
        mark_corrupted();
        TI_WARN("Invalid GFX AOT descriptor array-offset contract");
        return;
      }
      for (const auto &[indices, argument] :
           ti_aot_data_.kernels[i].ctx_attribs.args()) {
        const bool storage = argument.ptype == ParameterType::kRWTexture;
        if (argument.ptype == ParameterType::kTextureCollection &&
            (ti_aot_data_.metadata_version < 5 || !argument.is_array ||
             argument.is_argpack || argument.element_shape.size() != 1 ||
             argument.element_shape[0] <= 0 || argument.field_dim > 2 ||
             argument.format != BufferFormat::unknown ||
             argument.texture_lod != 0)) {
          mark_corrupted();
          TI_WARN("Invalid GFX AOT texture-collection argument metadata");
          return;
        }
        if ((!storage && argument.texture_lod != 0) ||
            (argument.texture_lod != 0 && ti_aot_data_.metadata_version < 4) ||
            (ti_aot_data_.metadata_version >= 4 && storage &&
             (argument.format == BufferFormat::unknown ||
              argument.texture_lod < 0 || argument.texture_lod > 30 ||
              (argument.texture_lod != 0 && argument.field_dim != 2)))) {
          mark_corrupted();
          TI_WARN("Invalid GFX AOT storage-image argument metadata");
          return;
        }
      }
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
      try {
        SNodeResourceResolver resolver(resource_trees,
                                       metadata.used_snode_tree_ids);
        for (const auto &task : ti_aot_data_.kernels[i].tasks_attribs) {
          if (!ti_aot_data_.snode_tree_resources.empty()) {
            const auto list = resolver.resolve(task.sparse_list_snode_id);
            const auto parent =
                resolver.resolve(task.sparse_list_parent_snode_id);
            resolver.resolve(task.sparse_mutation_snode_id);
            if (list.valid() && parent.valid() &&
                list.tree_id != parent.tree_id)
              throw std::invalid_argument(
                  "Sparse list and parent belong to different trees");
            for (const auto &bind : task.buffer_binds) {
              if (bind.buffer.type !=
                  spirv::TaskAttributes::BufferType::NodeAllocatorPool)
                continue;
              if (bind.buffer.root_id.size() != 1 ||
                  bind.buffer.root_id[0] < 0 || bind.chunk_count != 0)
                throw std::invalid_argument("Invalid AOT pointer pool binding");
              const auto id = resolver.resolve(bind.buffer.root_id[0]);
              const auto &pools = ti_aot_data_.snode_tree_resources[id.tree_id]
                                      .resources.pointer_pools;
              const auto pool =
                  std::lower_bound(pools.begin(), pools.end(), id.snode_id,
                                   [](const auto &entry, int node) {
                                     return entry.contract.snode_id < node;
                                   });
              if (pool == pools.end() ||
                  pool->contract.snode_id != id.snode_id ||
                  pool->independent_size == 0)
                throw std::invalid_argument(
                    "AOT task references an undeclared independent pool");
            }
          } else if (ti_aot_data_.metadata_version >= 7 &&
                     (task.may_mutate_sparse_topology ||
                      std::any_of(task.buffer_binds.begin(),
                                  task.buffer_binds.end(),
                                  [](const auto &bind) {
                                    return bind.buffer.type ==
                                           spirv::TaskAttributes::BufferType::
                                               NodeAllocatorPool;
                                  }))) {
            throw std::invalid_argument(
                "Sparse AOT task lacks resource declarations");
          }
        }
      } catch (const std::exception &error) {
        mark_corrupted();
        TI_WARN("Invalid GFX AOT kernel resources: {}", error.what());
        return;
      }
      for (const auto &task : ti_aot_data_.kernels[i].tasks_attribs) {
        for (const auto &bind : task.buffer_binds) {
          if (bind.buffer.type == spirv::TaskAttributes::BufferType::Root &&
              (bind.buffer.root_id.size() != 1 ||
               !std::binary_search(metadata.used_snode_tree_ids.begin(),
                                   metadata.used_snode_tree_ids.end(),
                                   bind.buffer.root_id.front()))) {
            mark_corrupted();
            TI_WARN("GFX AOT kernel {} has an undeclared root binding", i);
            return;
          }
        }
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
    std::unordered_map<std::string, size_t> identities;
    std::vector<size_t> canonical_indices;
    canonical_indices.reserve(ti_aot_data_.kernels.size());
    for (size_t i = 0; i < ti_aot_data_.kernels.size(); ++i) {
      size_t canonical = i;
      if (ti_aot_data_.metadata_version < 20) {
        canonical = identities.emplace(kernel_identity(ti_aot_data_.kernels[i],
                                        ti_aot_data_.kernel_metadata[i]), i).first->second;
      }
      canonical_indices.push_back(canonical);
      // Legacy exporters may repeat a kernel shared by multiple Graphs.
      // Preserve their first-match lookup behavior without repeated scans.
      const bool inserted = kernel_indices_
                                .emplace(ti_aot_data_.kernels[i].name, canonical)
                                .second;
      if (!inserted && ti_aot_data_.metadata_version >= 2) {
        mark_corrupted();
        TI_WARN("GFX AOT artifact contains a duplicate kernel name");
        return;
      }
      if (!source_ && !load_shader(*dir, canonical)) {
        mark_corrupted();
        TI_WARN("{}", shader_errors_[canonical]);
        return;
      }
    }
    for (const auto &alias : ti_aot_data_.kernel_aliases) {
      if (alias.kernel_index >= canonical_indices.size() || alias.name.empty() ||
          !kernel_indices_
               .emplace(alias.name, canonical_indices[alias.kernel_index])
               .second) {
        mark_corrupted();
        TI_WARN("GFX AOT artifact contains an invalid kernel alias '{}': {}",
                alias.name, alias.kernel_index);
        return;
      }
    }

    try {
      if (ti_aot_data_.metadata_version < 11 && !ti_aot_data_.device_range_dispatch.empty())
        throw std::invalid_argument("Device range selection requires GFX metadata v11");
      for (const auto &entry : ti_aot_data_.device_range_dispatch) {
        const auto found = kernel_indices_.find(entry.name);
        if (device_api_backend_ != Arch::vulkan || entry.version != 1 || entry.tasks.empty() ||
            found == kernel_indices_.end() || !range_selections_.emplace(entry.name, entry.tasks).second)
          throw std::invalid_argument("Invalid GFX AOT range dispatch entry");
        RangeDispatch::validate_selection(ti_aot_data_.kernels[found->second], entry.tasks, kGtmpBufferSize);
      }
    } catch (const std::exception &error) {
      mark_corrupted();
      TI_WARN("Invalid GFX AOT range selection: {}", error.what());
      return;
    }

    {
      try {
        if (aot::load_struct_graph_artifact(*dir, graphs_)) return;
      } catch (const std::exception &e) {
        mark_corrupted();
        TI_WARN("Invalid GFX AOT Graph v2 metadata: {}", e.what());
        return;
      }
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

  ~AotModuleImpl() override {
    if (owns_runtime_resources_) {
      try {
        std::vector<GfxRuntime::KernelHandle> handles;
        handles.reserve(registered_kernels_.size());
        for (const auto &[index, registration] : registered_kernels_) {
          if (registration->handle.get_launch_id() >= 0)
            handles.push_back(registration->handle);
        }
        for (const auto &[key, kernel] : specialized_kernels_)
          handles.push_back(kernel->handle());
        runtime_->release_aot_resources(handles, root_bindings_);
      } catch (const std::string &error) {
        // TI_ERROR reports sparse overflow as a string. Destruction must not
        // terminate the process; the runtime still owns the unretired roots.
        TI_WARN("Cannot retire GFX AOT resources: {}", error);
      } catch (const std::exception &e) {
        // A failed device synchronization must not free resources still in
        // flight or throw from a destructor. The runtime retains them.
        TI_WARN("Cannot retire GFX AOT resources: {}", e.what());
      }
    }
  }

  void allocate_root_buffers() {
    TI_ERROR_IF(is_corrupted() || runtime_ == nullptr,
                "Cannot allocate resources for an invalid GFX AOT module.");
    if (owns_runtime_resources_) {
      return;
    }
    TI_ERROR_IF(!registered_kernels_.empty(),
                "GFX AOT resources must be bound before loading kernels.");
    if (ti_aot_data_.snode_tree_resources.empty()) {
      root_bindings_ =
          runtime_->allocate_aot_root_buffers(ti_aot_data_.root_buffer_sizes);
    } else {
      std::vector<SNodeTreeResourceSource> sources;
      for (const auto &entry : ti_aot_data_.snode_tree_resources)
        sources.push_back({entry.producer_tree_id, &entry.resources});
      root_bindings_ = runtime_->allocate_aot_snode_trees(sources);
    }
    owns_runtime_resources_ = true;
    try {
      for (size_t i = 0; i < ti_aot_data_.initial_root_data.size(); ++i) {
        const auto &entry = ti_aot_data_.initial_root_data[i];
        initialize_root_data(
            entry,
            runtime_->get_root_buffer(root_bindings_.at(entry.tree_id))
                ->get_ptr(),
            source_ ? nullptr : &eager_initial_root_data_.at(i));
      }
      decltype(eager_initial_root_data_){}.swap(eager_initial_root_data_);
    } catch (...) {
      mark_corrupted();
      throw;
    }
  }

  std::unique_ptr<AotSNodeInstance> instantiate(
      std::shared_ptr<aot::Module> code,
      const std::vector<std::string> &entries) {
    TI_ERROR_IF(is_corrupted() || !owns_runtime_resources_,
                "SNode instances require an allocated valid GFX AOT module");
    if (entries.empty())
      throw std::invalid_argument("SNode instances require an entry closure");
    std::set<int> trees;
    std::map<std::string, std::size_t> selected;
    for (const auto &name : entries) {
      const auto found = kernel_indices_.find(name);
      if (found == kernel_indices_.end())
        throw std::invalid_argument("Unknown SNode instance entry: " + name);
      selected.emplace(name, found->second);
      const auto &used =
          ti_aot_data_.kernel_metadata[found->second].used_snode_tree_ids;
      trees.insert(used.begin(), used.end());
    }
    if (!trees.empty() && ti_aot_data_.snode_tree_resources.empty())
      throw std::invalid_argument(
          "SNode instances require authoritative GFX tree resource recipes");

    // Resolve and compile each original entry once, before state allocation.
    // Aliases already share their original executable registration.
    std::map<std::string, KernelImpl *> originals;
    for (const auto &[name, index] : selected)
      originals.emplace(name, dynamic_cast<KernelImpl *>(get_kernel(name)));

    auto result = std::make_unique<SNodeInstanceImpl>(std::move(code), runtime_);
    result->registrations_.reserve(selected.size());
    result->root_bindings_.resize(root_bindings_.size(), -1);
    std::vector<SNodeTreeResourceSource> sources;
    for (int id : trees) {
      const auto &tree = ti_aot_data_.snode_tree_resources.at(id);
      sources.push_back({tree.producer_tree_id, &tree.resources});
    }
    result->allocated_roots_ = runtime_->allocate_aot_snode_trees(sources);
    std::size_t position = 0;
    for (int id : trees)
      result->root_bindings_[id] = result->allocated_roots_[position++];

    std::map<int, GfxRuntime::KernelHandle> views;
    for (const auto &[name, index] : selected) {
      auto *original = originals.at(name);
      TI_ASSERT(original != nullptr);
      auto params = original->params();
      params.root_buffer_bindings = result->root_bindings_;
      params.snode_tree_ids = ti_aot_data_.kernel_metadata[index].used_snode_tree_ids;
      for (int &id : params.snode_tree_ids)
        id = result->root_bindings_.at(id);
      params.sparse_diagnostic_domain = result->allocated_roots_.empty()
          ? -1 : result->allocated_roots_.front();
      const int code_id = original->handle().get_launch_id();
      auto found = views.find(code_id);
      if (found == views.end()) {
        const auto handle = runtime_->register_taichi_kernel_resource_view(
            params, original->handle());
        result->registrations_.push_back(handle);
        found = views.emplace(code_id, handle).first;
      }
      result->kernels_.emplace(name, std::make_unique<KernelImpl>(
          runtime_, std::move(params), &found->second,
          original->device_range_tasks()));
    }
    return result;
  }

  std::unique_ptr<AotSNodeInstance> instantiate_all(std::shared_ptr<aot::Module> code) {
    std::vector<std::string> entries;
    for (const auto &[name, index] : kernel_indices_)
      entries.push_back(name);
    return instantiate(std::move(code), entries);
  }

  std::string fixed_layouts(const std::vector<std::string> &entries) const {
    if (is_corrupted())
      throw std::invalid_argument("Invalid GFX AOT module");
    std::set<int> trees;
    for (const auto &name : entries) {
      const auto found = kernel_indices_.find(name);
      if (found == kernel_indices_.end())
        throw std::invalid_argument("Unknown fixed SNode layout entry: " + name);
      const auto &used = ti_aot_data_.kernel_metadata[found->second].used_snode_tree_ids;
      trees.insert(used.begin(), used.end());
    }
    liong::json::JsonArray result;
    for (int tree : trees) {
      if (static_cast<std::size_t>(tree) >= ti_aot_data_.snode_tree_resources.size())
        throw std::invalid_argument("Missing authoritative GFX fixed SNode layout");
      liong::json::JsonObject entry;
      entry.inner.emplace("tree", tree);
      entry.inner.emplace("layout", liong::json::parse(
          ti_aot_data_.snode_tree_resources[tree].normalized_layout_json()));
      result.inner.emplace_back(std::move(entry));
    }
    return liong::json::print(liong::json::JsonValue(std::move(result)));
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
      aot::CompiledDispatch loaded;
      loaded.kernel_name = dispatch.kernel_name;
      loaded.symbolic_args = dispatch.symbolic_args;
      loaded.compiled_kernel = kernel;
      dispatches.push_back(std::move(loaded));
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

#ifdef TI_WITH_VULKAN
  aot::Kernel *specialize_kernel(aot::Kernel &kernel,
                                 LaunchContextBuilder &context,
                                 const std::vector<uint32_t> &constant_args,
                                 const std::vector<uint32_t> & /*private_workspaces*/) override {
    auto *source = dynamic_cast<KernelImpl *>(&kernel);
    TI_ERROR_IF(device_api_backend_ != Arch::vulkan || !source ||
                    &source->runtime() != runtime_ || !source->device_range_tasks().empty() ||
                    !kernel_indices_.count(kernel.name),
                "Prepared specialization requires an ordinary Vulkan kernel");
    auto constants = aot::specialization_constants(kernel, context, constant_args);
    const auto &source_params = source->params();
    const auto &abi = source_params.kernel_attribs.ctx_attribs;
    // In the descriptor ABI the pointer slot carries the byte displacement
    // inside the aligned binding. Its value belongs to this prepared view.
    if (abi.descriptor_array_offsets && abi.args_type()->get_layout() != "1b") {
      const auto limits = runtime_->get_ti_device()->get_storage_buffer_binding_limits();
      for (const auto &[path, parameter] : kernel.nested_parameters) {
        if (parameter.ptype != ParameterType::kNdarray)
          continue;
        auto member = path;
        member.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
        auto add_bias = [&] {
          const auto bias = descriptor_array_byte_bias(
              limits, context.find_resolved_dense_storage_buffer(member));
          constants.emplace(abi.args_type()->get_element_offset(member),
                            aot::ArgumentConstant{32, bias});
        };
        add_bias();
        if (parameter.needs_grad) {
          member.back() = TypeFactory::GRAD_PTR_POS_IN_NDARRAY;
          add_bias();
        }
      }
    }
    const auto index = kernel_indices_.at(kernel.name);
    const auto code_key = aot::specialization_key(std::to_string(index), constants);
    // Code is shared, while independent SNode instances retain their own
    // resource view. The resource-view cache never rewrites module bindings.
    std::string key = code_key;
    for (auto root : source_params.root_buffer_bindings)
      key += fmt::format("|r:{}", root);
    key += fmt::format("|d:{}", source_params.sparse_diagnostic_domain);
    auto found = specialized_kernels_.find(key);
    if (found != specialized_kernels_.end())
      return found->second.get();
    auto params = source_params;
    const auto shared = specialized_code_.find(code_key);
    std::unique_ptr<KernelImpl> result;
    if (shared != specialized_code_.end()) {
      const auto handle = runtime_->register_taichi_kernel_resource_view(params, shared->second);
      result = std::make_unique<KernelImpl>(runtime_, std::move(params), &handle);
    } else {
      params.task_spirv_source_codes = params.spirv_codes();
      params.shared_task_spirv_source_codes.reset();
      std::vector<vulkan::ShaderArgumentConstant> values;
      for (const auto &[offset, value] : constants)
        values.push_back({static_cast<uint32_t>(offset), value.bits, value.value});
      const auto folded = specialize_task_arguments(params, values, prepare_compile_threads_);
      TI_DEBUG("Prepared Vulkan specialization {} folded {} ABI loads", kernel.name, folded);
      result = std::make_unique<KernelImpl>(runtime_, std::move(params));
      specialized_code_.emplace(code_key, result->handle());
    }
    static_cast<CallableBase &>(*result) = static_cast<const CallableBase &>(kernel);
    auto *callable = result.get();
    specialized_kernels_.emplace(std::move(key), std::move(result));
    return callable;
  }
#endif

  void set_prepare_compile_threads(int threads) {
    prepare_compile_threads_ = std::max(1, threads);
  }

 private:
  int prepare_compile_threads_{1};
  std::map<std::string, std::vector<uint32_t>> range_selections_;

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

  const std::string &shader_source(size_t index, size_t task) const {
    const auto &name = ti_aot_data_.kernels[index].tasks_attribs[task].name;
    const auto overrides = ti_aot_data_.task_shader_sources.find(index);
    if (overrides != ti_aot_data_.task_shader_sources.end()) {
      const auto found = overrides->second.find(task);
      if (found != overrides->second.end())
        return found->second;
    }
    const auto alias = ti_aot_data_.shader_aliases.find(name);
    return alias == ti_aot_data_.shader_aliases.end() ? name : alias->second;
  }

  bool has_shared_functions(const std::string &name) const {
    if (ti_aot_data_.shader_functions.empty())
      return false;
    const auto found = kernel_indices_.find(name);
    if (found != kernel_indices_.end()) {
      for (size_t i = 0; i < ti_aot_data_.kernels[found->second].tasks_attribs.size(); ++i) {
        if (ti_aot_data_.shader_functions.count(shader_source(found->second, i)))
          return true;
      }
    }
    return false;
  }

  bool get_kernel_params_by_name(const std::string &name,
                                 GfxRuntime::RegisterParams &kernel,
                                 bool defer_code) {
    auto found = kernel_indices_.find(name);
    if (found != kernel_indices_.end()) {
      const size_t i = found->second;
      if (defer_code && !shader_codes_[i])
        shader_codes_[i] = std::make_shared<ShaderCode>();
      if (source_ && !defer_code && !load_shader(*source_, i)) {
        throw aot::ArtifactLoadError(shader_errors_[i]);
      }
      kernel.kernel_attribs = ti_aot_data_.kernels[i];
      kernel.kernel_attribs.name = name;
      kernel.shared_task_spirv_source_codes = shader_codes_[i];
      kernel.num_snode_trees = ti_aot_data_.kernel_metadata[i].num_snode_trees;
      kernel.snode_tree_ids =
          ti_aot_data_.kernel_metadata[i].used_snode_tree_ids;
      if (owns_runtime_resources_) {
        kernel.root_buffer_bindings = root_bindings_;
        if (!ti_aot_data_.snode_tree_resources.empty())
          kernel.sparse_diagnostic_domain = root_bindings_.front();
        for (int &id : kernel.snode_tree_ids) {
          id = root_bindings_.at(id);
        }
        std::sort(kernel.snode_tree_ids.begin(), kernel.snode_tree_ids.end());
      }
      return true;
    }
    return false;
  }

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override {
    GfxRuntime::RegisterParams kparams;
    auto range = range_selections_.find(name);
    const bool defer_code = source_ && range == range_selections_.end() &&
                            has_shared_functions(name);
    if (!get_kernel_params_by_name(name, kparams, defer_code)) {
      TI_DEBUG("Failed to load kernel {}", name);
      return nullptr;
    }
    const size_t index = kernel_indices_.at(name);
    auto &registration = registered_kernels_[index];
    if (!registration) {
      registration = std::make_shared<AotKernelRegistration>();
      if (defer_code) {
        registration->load_code = [this, index] {
          if (!load_shader(*source_, index))
            throw aot::ArtifactLoadError(shader_errors_[index]);
        };
      }
    }
    const std::vector<uint32_t> tasks =
        range == range_selections_.end() ? std::vector<uint32_t>{} : range->second;
    // A Vulkan prepare can compile the selected specialization directly.
    if (device_api_backend_ == Arch::vulkan && tasks.empty())
      return std::make_unique<KernelImpl>(
          runtime_, std::move(kparams), nullptr, tasks, registration);
    const auto handle = registration->get(*runtime_, kparams);
    return std::make_unique<KernelImpl>(runtime_, std::move(kparams), &handle, tasks);
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

  using ShaderCode = std::vector<std::vector<uint32_t>>;
  struct LinkedFunction {
    const uint32_t *data;
    size_t words;
    size_t relocation_bytes;
  };

  static std::string function_key(const AotShaderFunction &ref) {
    return ref.base_sha256.empty() ? ref.sha256
                                   : ref.base_sha256 + "/" + ref.sha256;
  }

  void copy_function_base(const io::VirtualDir &dir,
                          const std::string &digest,
                          size_t words,
                          uint32_t *out) {
    const auto linked = linked_functions_.find(digest);
    const size_t size = words * sizeof(uint32_t);
    if (linked != linked_functions_.end()) {
      if (linked->second.words != words)
        throw std::invalid_argument("Shared shader base size mismatch");
      std::memcpy(out, linked->second.data, size);
      return;
    }
    auto stored = function_bases_.find(digest);
    if (stored == function_bases_.end())
      stored =
          function_bases_.emplace(digest, load_data_from(&dir, digest, size))
              .first;
    if (stored->second.size() != size)
      throw std::invalid_argument("Shared shader base size mismatch");
    std::memcpy(out, stored->second.data(), size);
  }

  static void apply_relocation(const uint8_t *bytes,
                               size_t size,
                               const AotShaderFunction &ref,
                               uint32_t *body) {
    size_t cursor = ref.relocation_bytes ? (ref.size_words + 7) / 8 : 0;
    if (cursor > size)
      throw std::invalid_argument("Invalid SPIR-V relocation mask");
    for (size_t i = 0; i < ref.size_words; ++i) {
      if (ref.relocation_bytes && !(bytes[i / 8] & (1u << (i % 8))))
        continue;
      if (size - cursor < sizeof(uint32_t))
        throw std::invalid_argument("Truncated SPIR-V relocation");
      uint32_t word;
      std::memcpy(&word, bytes + cursor, sizeof(word));
      body[i] += word;
      cursor += sizeof(word);
    }
    if (cursor != size)
      throw std::invalid_argument("Invalid SPIR-V relocation length");
  }

  void link_functions(const io::VirtualDir &dir,
                      const std::string &source,
                      const std::vector<AotShaderFunction> &refs,
                      std::vector<uint32_t> &code) {
    std::vector<uint32_t> root;
    if (!aot::load_shader_words(dir, source + ".spvroot", root) || root.size() < 5)
      throw std::invalid_argument("Cannot read Vulkan shader root");
    size_t local = 0;
    const size_t min_words = ti_aot_data_.metadata_version >= 23 ? 1 : 6;
    for (const auto &ref : refs) {
      if (ref.offset_words < std::max(size_t(5), code.size()) ||
          ref.offset_words - code.size() > root.size() - local ||
          ref.size_words < min_words ||
          ref.size_words > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
        throw std::invalid_argument("Invalid shared shader function extent");
      if ((ref.relocation_bytes || ref.relocation_in_root) &&
          (ti_aot_data_.metadata_version < 24 || ref.base_sha256.empty() ||
           ref.relocation_bytes >= ref.size_words * sizeof(uint32_t)))
        throw std::invalid_argument("Invalid compact SPIR-V relocation");
      const size_t next = local + (ref.offset_words - code.size());
      code.insert(code.end(), root.begin() + local, root.begin() + next);
      local = next;
      const size_t delta_size = ref.relocation_bytes ? ref.relocation_bytes
                                                     : ref.size_words * sizeof(uint32_t);
      const uint8_t *inline_delta = nullptr;
      if (ref.relocation_in_root) {
        const size_t span = (delta_size + 3) / 4;
        if (span > root.size() - local)
          throw std::invalid_argument("Truncated SPIR-V root relocation");
        inline_delta = reinterpret_cast<const uint8_t *>(root.data() + local);
        if (picosha2::hash256_hex_string(inline_delta, inline_delta + delta_size) != ref.sha256)
          throw std::invalid_argument("SPIR-V root relocation failed integrity validation");
        local += span;
      }
      const auto cached = linked_functions_.find(function_key(ref));
      if (cached != linked_functions_.end()) {
        if (cached->second.words != ref.size_words ||
            cached->second.relocation_bytes != ref.relocation_bytes)
          throw std::invalid_argument("Shared shader function size mismatch");
        const auto *first = cached->second.data;
        code.insert(code.end(), first, first + ref.size_words);
      } else {
        const size_t start = code.size();
        code.resize(start + ref.size_words);
        auto *body = code.data() + start;
        if (!ref.base_sha256.empty()) {
          if (ti_aot_data_.metadata_version < 22)
            throw std::invalid_argument(
                "Shared shader base requires GFX AOT metadata v22");
          copy_function_base(dir, ref.base_sha256, ref.size_words, body);
          std::vector<uint8_t> external;
          if (!inline_delta)
            external = load_data_from(&dir, ref.sha256, delta_size);
          apply_relocation(inline_delta ? inline_delta : external.data(), delta_size, ref, body);
        } else if (function_bases_.count(ref.sha256)) {
          copy_function_base(dir, ref.sha256, ref.size_words, body);
        } else {
          const auto bytes = load_data_from(&dir, ref.sha256,
                                            ref.size_words * sizeof(uint32_t));
          std::memcpy(body, bytes.data(), bytes.size());
        }
        if (ti_aot_data_.metadata_version < 23) {
          if (code[start] != ((5u << 16) | spv::OpFunction) ||
              code.back() != ((1u << 16) | spv::OpFunctionEnd))
            throw std::invalid_argument("Invalid SPIR-V function body");
        } else {
          for (size_t i = start; i < code.size();) {
            size_t words = code[i] >> 16;
            if (!words || words > code.size() - i)
              throw std::invalid_argument("Invalid shared SPIR-V instruction range");
            i += words;
          }
        }
      }
    }
    code.insert(code.end(), root.begin() + local, root.end());
  }

  bool load_shader(const io::VirtualDir &dir, size_t index) {
    // The cache borrows spans of previously verified immutable task code.
    // Publish them only after the whole kernel passes its ABI checks.
    std::lock_guard<std::mutex> guard(shader_mutex_);
    const auto &tasks = ti_aot_data_.kernels[index].tasks_attribs;
    if (shader_codes_[index] && shader_codes_[index]->size() == tasks.size())
      return true;
    if (!shader_errors_[index].empty())
      return false;
    std::vector<std::vector<uint32_t>> codes;
    std::vector<const std::vector<AotShaderFunction> *> function_refs;
    for (size_t i = 0; i < tasks.size(); ++i) {
      const auto &task = tasks[i];
      const auto &source = shader_source(index, i);
      std::vector<uint32_t> code;
      try {
        const auto refs = ti_aot_data_.shader_functions.find(source);
        if (refs != ti_aot_data_.shader_functions.end()) {
          link_functions(dir, source, refs->second, code);
          function_refs.push_back(&refs->second);
        } else {
          if (!aot::load_shader_words(dir, source + ".spv", code))
            throw std::invalid_argument("Cannot read SPIR-V payload");
          function_refs.push_back(nullptr);
        }
        if (code.size() < 5 || code[0] != 0x07230203)
          throw std::invalid_argument("Invalid SPIR-V header");
        if (spirv::runtime_affine_arrays(code) != ti_aot_data_.kernels[index].runtime_affine_arrays)
          throw std::invalid_argument("GFX affine array metadata differs from compiled SPIR-V");
        validate_cooperative_matrix_metadata(
            task, code, ti_aot_data_.metadata_version);
      } catch (const std::exception &error) {
        shader_errors_[index] = source + ": " + error.what();
        return false;
      }
      codes.emplace_back(std::move(code));
    }
    if (!shader_codes_[index])
      shader_codes_[index] = std::make_shared<ShaderCode>();
    *shader_codes_[index] = std::move(codes);
    for (size_t i = 0; i < function_refs.size(); ++i) {
      if (function_refs[i]) {
        for (const auto &ref : *function_refs[i]) {
          const auto *body = (*shader_codes_[index])[i].data() + ref.offset_words;
          linked_functions_.try_emplace(function_key(ref),
                                        LinkedFunction{body, ref.size_words, ref.relocation_bytes});
          if (ref.base_sha256.empty())
            function_bases_.erase(ref.sha256);
        }
      }
    }
    return true;
  }

  const io::VirtualDir *data_source() const override {
    return source_.get();
  }

  RootStorage get_root_storage(int tree_id) override {
    if (!owns_runtime_resources_ || tree_id < 0 ||
        static_cast<std::size_t>(tree_id) >= root_bindings_.size())
      throw std::invalid_argument(
          "AOT root storage requires an allocated module-local tree");
    return {runtime_->get_root_buffer(root_bindings_[tree_id])->get_ptr(),
            ti_aot_data_.root_buffer_sizes[tree_id]};
  }

  std::shared_ptr<const io::VirtualDir> source_;
  TaichiAotData ti_aot_data_;
  DeviceCapabilityConfig required_caps_;
  std::mutex shader_mutex_;
  std::vector<std::shared_ptr<ShaderCode>> shader_codes_;
  std::unordered_map<std::string, LinkedFunction> linked_functions_;
  // Own a requested base only until a resident shader supplies that span.
  std::unordered_map<std::string, std::vector<uint8_t>> function_bases_;
  std::vector<std::string> shader_errors_;
  std::unordered_map<std::string, size_t> kernel_indices_;
  std::unordered_map<size_t, std::shared_ptr<AotKernelRegistration>> registered_kernels_;
  std::unordered_map<std::string, GfxRuntime::KernelHandle> specialized_code_;
  std::unordered_map<std::string, std::unique_ptr<KernelImpl>> specialized_kernels_;
  std::vector<int> root_bindings_;
  std::vector<std::vector<std::uint8_t>> eager_initial_root_data_;
  bool owns_runtime_resources_{false};
  GfxRuntime *runtime_{nullptr};
  Arch device_api_backend_;
};

}  // namespace

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params,
                                             Arch device_api_backend) {
  AotModuleParams params = std::any_cast<AotModuleParams &>(mod_params);
  return std::make_unique<AotModuleImpl>(params, device_api_backend);
}

void allocate_aot_root_buffers(aot::Module *module) {
  auto *gfx_module = dynamic_cast<AotModuleImpl *>(module);
  TI_ERROR_IF(!gfx_module, "Expected a GFX AOT module.");
  gfx_module->allocate_root_buffers();
}

void set_aot_prepare_compile_threads(aot::Module &module, int threads) {
  if (auto *spirv_module = dynamic_cast<AotModuleImpl *>(&module))
    spirv_module->set_prepare_compile_threads(threads);
}

std::unique_ptr<AotSNodeInstance> make_aot_snode_instance(
    std::shared_ptr<aot::Module> code,
    const std::vector<std::string> &entries) {
  auto *module = dynamic_cast<AotModuleImpl *>(code.get());
  if (!module)
    throw std::invalid_argument("SNode instances require a GFX AOT code owner");
  return module->instantiate(std::move(code), entries);
}

std::string fixed_snode_plan_layouts(
    aot::Module *module, const std::vector<std::string> &entries) {
  auto *gfx = dynamic_cast<AotModuleImpl *>(module);
  if (!gfx)
    throw std::invalid_argument("Fixed SNode layouts require a GFX AOT module");
  return gfx->fixed_layouts(entries);
}

std::unique_ptr<AotSNodeInstance> make_aot_snode_instance(std::shared_ptr<aot::Module> code) {
  auto *module = dynamic_cast<AotModuleImpl *>(code.get());
  if (!module)
    throw std::invalid_argument("SNode instances require a GFX AOT code owner");
  return module->instantiate_all(std::move(code));
}

}  // namespace gfx
}  // namespace taichi::lang
