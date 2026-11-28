#include "unified_aot_module.h"

#include <optional>
#include <set>

#include "taichi_core_impl.h"
#include "argument_binding.h"
#include "taichi/aot/specialization.h"
#include "taichi/codegen/llvm/unified_kernel_reader.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/stmt_use_index.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/array_layout_abi.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/struct/dense_snode_schema.h"
#include "taichi/system/profiler.h"
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/gfx/descriptor_binding.h"
#include "taichi/runtime/gfx/aot_snode_resources.h"
#endif

using taichi::lang::Program;

Program &Runtime::get_unified_compiler() {
  if (!unified_compiler) {
    // Prepared kernels launch on this Runtime's executor.
    unified_compiler = std::make_shared<Program>(arch, /*compile_only=*/true);
  }
  return *unified_compiler;
}

namespace capi {
using namespace taichi::lang;
using taichi::Arch;
using taichi::arch_name;
using taichi::uint64;
using taichi::io::VirtualDir;
namespace {
using ArrayPaths = std::vector<std::vector<int>>;

template <typename BackendKernel>
class DeferredLayoutKernel final : public BackendKernel, public DeferredArrayLayouts {
 public:
  template <typename... Args>
  DeferredLayoutKernel(std::function<ArrayPaths()> load_layouts, Args &&...args)
      : BackendKernel(std::forward<Args>(args)...), load_layouts_(std::move(load_layouts)) {}

  const ArrayPaths &resolve_array_layouts() override {
    std::call_once(once_, [&] {
      layouts_ = load_layouts_();
      load_layouts_ = {};
    });
    return layouts_;
  }

 private:
  std::once_flag once_;
  std::function<ArrayPaths()> load_layouts_;
  ArrayPaths layouts_;
};

const std::string &string_value(const liong::json::JsonValue &v) {
  return static_cast<const std::string &>(v);
}

std::string read(const VirtualDir &source, const std::string &path) {
  size_t size;
  if (!source.get_file_size(path, size))
    throw aot::ArtifactLoadError("Missing unified LLVM library member " + path);
  std::string bytes(size, '\0');
  if (source.load_file(path, bytes.data(), size) != size)
    throw aot::ArtifactLoadError("Incomplete unified LLVM library member " + path);
  return bytes;
}

bool is_digest(const std::string &value) {
  return value.size() == 64 &&
         value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

class SelectedDirectory final : public VirtualDir {
 public:
  SelectedDirectory(std::shared_ptr<const VirtualDir> archive, Arch arch)
      : archive_(std::move(archive)) {
    const auto index = liong::json::parse(read(*archive_, "unified_llvm.json"));
    if (index.is_arr()) {
      // Module.archive writes the same numerical units and records directly.
      // Definition packages address the records through their content table.
      direct_ = true;
      for (const auto &row : index.elems()) {
        if (!entries.emplace(string_value(row["key"]), string_value(row["value"])).second)
          throw aot::ArtifactLoadError("Duplicate unified LLVM entry");
      }
    } else {
      const int version = int(index["version"]);
      if (version != 1 && version != 2)
        throw aot::ArtifactLoadError("Unsupported unified LLVM index");
      const auto &base = version == 2 ? index : index["base"];
      for (auto name : {"entries", "members"}) {
        auto &table = std::string(name) == "entries" ? entries : members_;
        for (const auto &[key, value] : base[name].fields())
          table.emplace(key, string_value(value));
      }
      if (version == 1)
        apply_legacy_changes(index["targets"][arch_name(arch).c_str()]);
    }
    for (const auto &[name, digest] : entries)
      if (!is_digest(digest))
        throw aot::ArtifactLoadError("Invalid unified LLVM root digest");
    for (const auto &[name, file] : members_)
      if (file.compare(0, 8, "records/") != 0 || !is_digest(file.substr(8)))
        throw aot::ArtifactLoadError("Invalid unified LLVM record digest");
    const auto abi = liong::json::parse(read(*this, "unified_abi.json"));
    for (const auto &row : abi.elems()) {
      auto digest = string_value(row["value"]);
      if (!is_digest(digest) ||
          !signatures.emplace(string_value(row["key"]), std::move(digest)).second)
        throw aot::ArtifactLoadError("Invalid unified LLVM entry signature");
    }
    size_t size;
    if (get_file_size("unified_absent_inputs.json", size)) {
      const auto inputs = liong::json::parse(read(*this, "unified_absent_inputs.json"));
      for (const auto &[device, records] : inputs.fields()) {
        if (device != arch_name(arch))
          continue;
        for (const auto &[symbol, slots] : records.fields()) {
          if (!entries.count(symbol))
            throw aot::ArtifactLoadError("Unknown entry in device resource declarations");
          for (const auto &slot : slots.elems())
            absent_inputs_[symbol].push_back(int(slot));
        }
      }
    }
  }

 private:
  void apply_legacy_changes(const liong::json::JsonValue &changes) {
    for (const auto &change : changes.elems()) {
      const auto &path = change[size_t(0)];
      if (path.size() != 2 || (change.size() != 1 && change.size() != 2))
        throw aot::ArtifactLoadError("Invalid unified LLVM index change");
      const auto &name = string_value(path[size_t(0)]);
      if (name != "entries" && name != "members")
        throw aot::ArtifactLoadError("Unknown unified LLVM index table");
      auto &table = name == "entries" ? entries : members_;
      const auto &key = string_value(path[size_t(1)]);
      if (change.size() == 1)
        table.erase(key);
      else
        table[key] = string_value(change[size_t(1)]);
    }
  }

 public:
  bool get_file_size(const std::string &path, size_t &size) const override {
    if (direct_)
      return archive_->get_file_size(path, size);
    auto it = members_.find(path);
    return it != members_.end() && archive_->get_file_size(it->second, size);
  }
  size_t load_file(const std::string &path, void *data, size_t size) const override {
    if (direct_)
      return archive_->load_file(path, data, size);
    auto it = members_.find(path);
    if (it == members_.end())
      return 0;
    // Verify a record against its content-addressed destination even when a
    // caller requests a prefix of it.
    auto bytes = read(*archive_, it->second);
    if (LLVM::aot_artifact_sha256(bytes) != it->second.substr(8))
      throw aot::ArtifactLoadError("Changed unified LLVM record " + path);
    auto count = std::min(size, bytes.size());
    if (count)
      std::memcpy(data, bytes.data(), count);
    return count;
  }
  bool is_immutable() const override { return archive_->is_immutable(); }
  std::string unit(const std::string &digest) const {
    return read(*archive_, "code_units/" + digest + ".bc");
  }
  // Imported helpers are verified by the LLVM reader against their call target.
  std::string root_unit(const std::string &digest) const {
    auto bytes = unit(digest);
    if (LLVM::aot_artifact_sha256(bytes) != digest)
      throw aot::ArtifactLoadError("Changed unified LLVM numerical unit " + digest);
    return bytes;
  }
  std::string signature(const std::string &name) const {
    const auto &digest = signatures.at(name);
    auto bytes = read(*this, "signatures/" + digest + ".json");
    if (direct_ && LLVM::aot_artifact_sha256(bytes) != digest)
      throw aot::ArtifactLoadError("Changed unified LLVM entry signature");
    return bytes;
  }
  std::map<std::string, std::string> entries, signatures;
  const std::vector<int> &absent_inputs(const std::string &symbol) const {
    static const std::vector<int> empty;
    auto it = absent_inputs_.find(symbol);
    return it == absent_inputs_.end() ? empty : it->second;
  }
 private:
  bool direct_{false};
  std::map<std::string, std::vector<int>> absent_inputs_;
  std::shared_ptr<const VirtualDir> archive_;
  std::map<std::string, std::string> members_;
};

using Facts = std::map<std::vector<int>, int>;
using LayoutWords = std::map<uint32_t, uint32_t>;

// Only an outer parallel loop with a single use of the descriptor extent can
// share code across work counts. Other uses (inner loops, indexing, device
// configuration or control flow) still participate in specialization.
std::set<Stmt *> cpu_dispatch_extents(Block *root) {
  std::set<Stmt *> extents;
  StmtUseIndex uses;
  for (const auto &stmt : root->statements) {
    auto *loop = stmt->cast<RangeForStmt>();
    if (!loop || loop->strictly_serialized || loop->model_block_dim ||
        !loop->device_options.empty())
      continue;
    auto *extent = loop->end->cast<ExternalTensorShapeAlongAxisStmt>();
    if (!extent || extent->axis != 0 || extent->parent != root || loop->begin == extent)
      continue;
    if (!uses.valid())
      uses.rebuild(root, root);
    const auto &users = uses.users(extent);
    if (users.size() == 1 && users.count(loop))
      extents.insert(extent);
  }
  return extents;
}
using Dimensions = std::map<std::string, int>;
using TreeDimensions = std::map<int, Dimensions>;

struct StoragePrototype {
  Program &program;
  SNodeTree *tree{nullptr};
  std::vector<SNode *> nodes;

  explicit StoragePrototype(Program &program) : program(program) {}
  ~StoragePrototype() {
    try {
      if (tree)
        program.destroy_snode_tree(tree);
    } catch (...) {
      TI_WARN("Cannot retire unified compile-time storage layout");
    }
  }
};

struct UnifiedLibrary {
  SelectedDirectory directory;
  std::map<std::string, DenseSNodeSchema> schemas;
  std::vector<std::string> layouts;
  std::map<std::string, std::vector<int>> entry_trees;
  std::map<std::pair<int, Dimensions>, std::weak_ptr<StoragePrototype>> prototypes;

  UnifiedLibrary(std::shared_ptr<const VirtualDir> dir, Arch arch) : directory(std::move(dir), arch) {
    size_t size;
    if (directory.get_file_size("unified_storage.json", size)) {
      auto catalog = liong::json::parse_complete(read(directory, "unified_storage.json"));
      for (const auto &digest : catalog["trees"].elems()) {
        auto key = string_value(digest);
        if (!schemas.count(key)) {
          auto json = liong::json::print(catalog["schemas"][key]);
          if (LLVM::aot_artifact_sha256(json) != key)
            throw aot::ArtifactLoadError("Changed unified storage schema");
          schemas.emplace(key, DenseSNodeSchema::from_json(json));
        }
        layouts.push_back(std::move(key));
      }
      for (const auto &[name, slots] : catalog["entries"].fields()) {
        if (!directory.entries.count(name))
          throw aot::ArtifactLoadError("Unknown entry in unified storage catalog");
        for (const auto &value : slots.elems()) {
          if (value.ty != liong::json::L_JSON_INT || value.num_int < 0 ||
              size_t(value.num_int) >= layouts.size())
            throw aot::ArtifactLoadError("Invalid unified storage slot");
          entry_trees[name].push_back(int(value.num_int));
        }
      }
    }
  }
  std::shared_ptr<StoragePrototype> prototype(Program &program, int slot, const Dimensions &dims) {
    auto key = std::make_pair(slot, dims);
    auto found = prototypes.find(key);
    if (found != prototypes.end()) {
      if (auto cached = found->second.lock())
        return cached;
    }
    std::vector<std::vector<SNode *>> projection;
    auto root = schemas.at(layouts.at(slot)).instantiate_projected(dims, projection);
    auto result = std::make_shared<StoragePrototype>(program);
    for (const auto &member : projection) {
      if (member.size() != 1)
        throw aot::ArtifactLoadError("Ambiguous native SNode member");
      result->nodes.push_back(member.front());
    }
    result->tree = program.add_snode_tree(std::move(root), true);
    for (auto it = prototypes.begin(); it != prototypes.end();) {
      if (it->second.expired())
        it = prototypes.erase(it);
      else
        ++it;
    }
    prototypes.emplace(std::move(key), result);
    return result;
  }
};

class UnifiedModule final : public aot::Module {
 public:
  UnifiedModule(::Runtime &runtime, std::shared_ptr<UnifiedLibrary> library,
                std::unique_ptr<LLVM::KernelLauncher> launcher, gfx::GfxRuntime *graphics,
                LlvmRuntimeExecutor *executor, uint64 *result_buffer, TreeDimensions dimensions = {})
      : runtime_(runtime), library_(std::move(library)), directory_(library_->directory),
        launcher_(std::move(launcher)), graphics_(graphics), executor_(executor),
        result_buffer_(result_buffer), dimensions_(std::move(dimensions)) {
#ifdef TI_WITH_CUDA
    // Runtime-owned physical trees are relocated during registration. Fixed
    // replay must acquire the same physical-tree leases as the native loader.
    if (!library_->layouts.empty() && runtime_.arch == Arch::cuda)
      dynamic_cast<cuda::KernelLauncher &>(*launcher_).enable_sparse_graph_replay();
#endif
  }

  std::unique_ptr<UnifiedModule> bind_storage(TreeDimensions dimensions) {
    for (const auto &[slot, values] : dimensions)
      if (slot < 0 || size_t(slot) >= library_->layouts.size())
        throw std::invalid_argument("Unknown unified storage slot");
    return std::make_unique<UnifiedModule>(runtime_, library_,
        executor_ ? make_llvm_aot_launcher(*executor_) : nullptr, graphics_,
        executor_, result_buffer_, std::move(dimensions));
  }
  ~UnifiedModule() override {
    try {
      if (launcher_ && used_) {
        runtime_.wait();
        launcher_->retire_all_kernels();
      }
#ifdef TI_WITH_VULKAN
      if (graphics_ && (!registrations_.empty() || !physical_trees_.empty())) {
        std::vector<gfx::GfxRuntime::KernelHandle> active;
        for (const auto &registration : registrations_)
          if (registration->handle.get_launch_id() >= 0)
            active.push_back(registration->handle);
        graphics_->release_aot_resources(active, physical_trees_);
      }
#endif
      if (executor_ && !physical_trees_.empty()) {
        for (auto &layout : llvm_layouts_)
          executor_->destroy_snode_tree_runtime(layout);
        executor_->release_aot_snode_tree_ids(physical_trees_);
      }
    } catch (const std::exception &e) {
      TI_WARN("Unified AOT kernel retirement failed: {}", e.what());
    } catch (const std::string &e) {
      TI_WARN("Unified AOT kernel retirement failed: {}", e);
    }
  }
  Arch arch() const override { return runtime_.arch; }
  uint64_t version() const override { return 1; }
  size_t get_root_size() const override { return 0; }
  std::vector<size_t> get_root_sizes() const override { return {}; }
  aot::Kernel *specialize_kernel(aot::Kernel &kernel, LaunchContextBuilder &context,
                                 const std::vector<uint32_t> &indices,
                                 const std::vector<uint32_t> &) override {
    const auto &reads = specialization_reads_.at(kernel.name);
    const auto constants = aot::specialization_constants(kernel, context, indices);
    LayoutWords layout;
    for (const auto &path : layout_reads_[kernel.name]) {
      const auto offset = kernel.args_type->get_element_offset(path);
      // These paths came from the target's typed array addressing operations.
      // Descriptor words are present for compact and explicit-stride views.
      TI_ERROR_IF(kernel.args_type->get_element_type(path) != PrimitiveType::i32 ||
                      offset < 0 || size_t(offset) + sizeof(uint32_t) > kernel.args_size,
                  "Invalid unified LLVM array layout member");
      uint32_t word;
      std::memcpy(&word, static_cast<const char *>(context.get_context().arg_buffer) + offset, sizeof(word));
      layout.emplace(offset, word);
    }
#ifdef TI_WITH_VULKAN
    if (const auto it = descriptor_reads_.find(kernel.name); it != descriptor_reads_.end()) {
      const auto limits = runtime_.get().get_storage_buffer_binding_limits();
      for (const auto &member : it->second) {
        const auto bias = gfx::descriptor_array_byte_bias(
            limits, context.find_resolved_dense_storage_buffer(member));
        layout.emplace(kernel.args_type->get_element_offset(member), bias);
      }
    }
#endif
    Facts facts;
    auto bind = [&](const std::vector<int> &path) {
      if (!reads.count(path))
        return;
      const auto offset = kernel.args_type->get_element_offset(path);
      auto it = constants.find(offset);
      if (it != constants.end() && it->second.bits == 32)
        facts[path] = static_cast<int32_t>(it->second.value);
    };
    for (auto index : indices)
      if (kernel.nested_parameters.at({int(index)}).get_dtype() == PrimitiveType::i32)
        bind({int(index)});
    for (const auto &[path, arg] : kernel.nested_parameters) {
      if (arg.ptype != ParameterType::kNdarray)
        continue;
      auto member = path;
      member.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      member.push_back(0);
      for (int axis = 0; axis < int(arg.total_dim - arg.element_shape.size()); ++axis) {
        member.back() = axis;
        bind(member);
      }
    }
    if (auto found = first_dispatch_counts_.find(kernel.name); found != first_dispatch_counts_.end()) {
      // Keep the first fixed launch just as before. When the caller also uses
      // other lengths, one dynamic CPU loop serves all of those bindings.
      for (auto &[path, first] : found->second) {
        auto value = facts.find(path);
        if (value == facts.end())
          continue;
        if (!first)
          first = value->second;
        else if (*first != value->second)
          facts.erase(value);
      }
    }
    if (facts.empty() && layout.empty())
      return &kernel;
    auto key = std::make_tuple(kernel.name, facts, layout);
    auto found = specialized_.find(key);
    if (found != specialized_.end())
      return found->second.get();
    auto result = compile(kernel.name, facts, layout);
    static_cast<CallableBase &>(*result) = static_cast<const CallableBase &>(kernel);
    return specialized_.emplace(std::move(key), std::move(result)).first->second.get();
  }
 protected:
  const VirtualDir *data_source() const override { return &directory_; }
  std::unique_ptr<aot::Kernel> make_new_kernel(const std::string &name) override {
    return directory_.entries.count(name) ? compile(name, {}) : nullptr;
  }
  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(const std::string &) override {
    throw std::invalid_argument("Unified library requires a typed entry");
  }
  std::unique_ptr<aot::Field> make_new_field(const std::string &) override {
    throw std::invalid_argument("Unified library has no named native storage here");
  }
 private:
  // Materialize only the storage connected to this entry. Trees used by one
  // shader share a diagnostic domain, even when other entries use a subset.
  std::vector<std::vector<SNode *>> storage(const std::string &name) {
    const auto found = library_->entry_trees.find(name);
    if (found == library_->entry_trees.end())
      return {};
    const auto &slots = found->second;
    if (!slots.empty() && !nodes_.count(slots.front())) {
      std::set<int> needed(slots.begin(), slots.end());
      size_t old_size;
      do {
        old_size = needed.size();
        for (const auto &[entry, connected] : library_->entry_trees)
          if (std::any_of(connected.begin(), connected.end(), [&](int i) { return needed.count(i); }))
            needed.insert(connected.begin(), connected.end());
      } while (needed.size() != old_size);
      prepare_trees(needed);
    }
    std::vector<std::vector<SNode *>> result;
    for (int slot : slots)
      result.push_back(nodes_.at(slot));
    return result;
  }

  void prepare_trees(const std::set<int> &slots) {
    auto &program = runtime_.get_unified_compiler();
    std::map<int, std::vector<SNode *>> nodes;
    std::map<int, int> bindings;
    std::vector<std::shared_ptr<StoragePrototype>> prototypes;
    std::vector<int> logical;
    for (int slot : slots) {
      auto layout = library_->prototype(program, slot, dimensions_[slot]);
      logical.push_back(layout->tree->id());
      bindings.emplace(layout->tree->id(), -1);
      nodes.emplace(slot, layout->nodes);
      prototypes.push_back(std::move(layout));
    }
    compile_trees_.reserve(compile_trees_.size() + slots.size());
    physical_trees_.reserve(physical_trees_.size() + slots.size());
    std::vector<int> physical;
    if (executor_) {
      auto &compiler = dynamic_cast<LlvmProgramImpl &>(*program.get_program_impl());
      std::vector<LLVM::SNodeTreeLayout> layouts;
      for (int id : logical)
        layouts.push_back(compiler.get_cached_field(id));
      llvm_layouts_.reserve(llvm_layouts_.size() + layouts.size());
      physical = executor_->reserve_aot_snode_tree_ids(logical.size());
      size_t initialized = 0;
      try {
        for (auto &layout : layouts) {
          layout.tree_id = physical[initialized];
          executor_->initialize_llvm_runtime_snodes(layout, result_buffer_);
          ++initialized;
        }
      } catch (...) {
        while (initialized)
          executor_->destroy_snode_tree_runtime(layouts[--initialized]);
        executor_->release_aot_snode_tree_ids(physical);
        throw;
      }
      std::move(layouts.begin(), layouts.end(), std::back_inserter(llvm_layouts_));
    }
#ifdef TI_WITH_VULKAN
    else {
      std::map<int, int> domains = bindings;
      std::vector<gfx::AotSNodeTreeResources> recipes;
      for (int id : logical)
        recipes.push_back(program.gfx_aot_tree_resources(id));
      std::vector<gfx::SNodeTreeResourceSource> sources;
      for (const auto &recipe : recipes)
        sources.push_back({recipe.producer_tree_id, &recipe.resources});
      physical = graphics_->allocate_aot_snode_trees(sources);
      for (auto &[id, domain] : domains)
        domain = physical.front();
      diagnostic_domains_.merge(domains);
    }
#endif
    for (size_t i = 0; i < logical.size(); ++i)
      bindings.at(logical[i]) = physical[i];
    tree_bindings_.merge(bindings);
    physical_trees_.insert(physical_trees_.end(), physical.begin(), physical.end());
    std::move(prototypes.begin(), prototypes.end(), std::back_inserter(compile_trees_));
    nodes_.merge(nodes);
  }

  std::shared_ptr<KernelExecutionHandle> materialize(const std::string &name,
                                                     const CompileConfig &cfg) {
    auto &program = runtime_.get_unified_compiler();
    auto signature = directory_.signature(name);
    const auto &caps = runtime_.get().get_caps();
    auto kernel = LLVM::read_unified_kernel(program, name, directory_.root_unit(directory_.entries.at(name)),
        signature, {}, caps, [this](const std::string &hash) { return directory_.unit(hash); }, storage(name), directory_.absent_inputs(name));
    return program.compile_kernel_execution_handle(cfg, caps, *kernel);
  }

  KernelLaunchHandle register_llvm(const KernelExecutionHandle &execution) {
    const auto &data = dynamic_cast<const LLVM::CompiledKernelData &>(execution.compiled());
    KernelRegistrationState registration;
    KernelLaunchHandle handle;
    if (data.has_snode_tree_dependencies()) {
      LLVM::CompiledKernelData::InternalData rebound(data.get_internal_data());
      LLVM::resolve_aot_tree_ids(*rebound.compiled_data.module, tree_bindings_);
      for (int &id : rebound.used_snode_tree_ids)
        id = tree_bindings_.at(id);
      std::sort(rebound.used_snode_tree_ids.begin(), rebound.used_snode_tree_ids.end());
      LLVM::CompiledKernelData relocated(arch(), std::move(rebound));
      handle = launcher_->register_llvm_kernel(relocated, registration);
    } else {
      handle = launcher_->register_llvm_kernel(data, registration);
    }
    used_ = true;
    return handle;
  }

#ifdef TI_WITH_VULKAN
  gfx::GfxRuntime::RegisterParams gfx_params(const KernelExecutionHandle &execution,
                                             const std::string &name) {
    const auto &data = dynamic_cast<const spirv::CompiledKernelData &>(execution.compiled()).get_internal_data();
    gfx::GfxRuntime::RegisterParams params;
    params.kernel_attribs = data.metadata.kernel_attribs;
    params.kernel_attribs.name = name;
    // A bound width baked into LocalSize no longer passes through
    // bound_pipeline(). Admit its launch and shared storage on this device.
    const auto &limits = static_cast<vulkan::VulkanDevice &>(runtime_.get())
                             .get_vk_physical_device_props().limits;
    for (const auto &task : params.kernel_attribs.tasks_attribs) {
      if (task.has_model_workgroup())
        continue;
      const uint32_t width = task.advisory_num_threads_per_group;
      TI_ERROR_IF(width > limits.maxComputeWorkGroupSize[0] ||
                      width > limits.maxComputeWorkGroupInvocations,
                  "Workgroup size {} exceeds the Vulkan device limit", width);
      TI_ERROR_IF(uint64_t(task.static_shared_array_bytes) > limits.maxComputeSharedMemorySize,
                  "Static workgroup storage exceeds the Vulkan device limit");
    }
    params.task_spirv_source_codes = data.src.spirv_src;
    params.num_snode_trees = data.metadata.num_snode_trees;
    params.snode_tree_ids = data.metadata.used_snode_tree_ids;
    if (!params.snode_tree_ids.empty()) {
      params.root_buffer_bindings.resize(params.num_snode_trees, -1);
      params.sparse_diagnostic_domain = diagnostic_domains_.at(params.snode_tree_ids.front());
      for (int &id : params.snode_tree_ids) {
        params.root_buffer_bindings.at(id) = tree_bindings_.at(id);
        id = tree_bindings_.at(id);
      }
    }
    return params;
  }
#endif

  std::unique_ptr<aot::Kernel> deferred(const std::string &name, Kernel &kernel,
                                        const CompileConfig &cfg) {
    const auto &caps = runtime_.get().get_caps();
    auto load_layouts = [this, name, cfg] {
      auto &program = runtime_.get_unified_compiler();
      const auto &target_caps = runtime_.get().get_caps();
      const auto signature = directory_.signature(name);
      bool needs_binding = false;
      auto generic = LLVM::read_unified_kernel(program, name, directory_.root_unit(directory_.entries.at(name)),
          signature, {}, target_caps, [this](const std::string &digest) { return directory_.unit(digest); },
          storage(name), directory_.absent_inputs(name), &needs_binding);
      return program.unified_kernel_affine_arguments(cfg, target_caps, *generic);
    };
    if (launcher_) {
      LLVM::KernelCallContract abi;
      abi.args.assign(kernel.nested_parameters.begin(), kernel.nested_parameters.end());
      abi.rets = kernel.rets;
      abi.ret_type = kernel.ret_type;
      abi.ret_size = kernel.ret_size;
      abi.args_type = kernel.args_type;
      abi.args_size = kernel.args_size;
      return std::make_unique<DeferredLayoutKernel<llvm_aot::KernelImpl>>(
          std::move(load_layouts), launcher_.get(), KernelLaunchHandle{}, arch(), name, abi, ArrayPaths{},
          [this, name, cfg] { return register_llvm(*materialize(name, cfg)); });
    }
#ifdef TI_WITH_VULKAN
    gfx::GfxRuntime::RegisterParams binding;
    binding.kernel_attribs.name = name;
    binding.kernel_attribs.ctx_attribs = spirv::KernelContextAttributes(kernel, &caps);
    auto registration = std::make_shared<gfx::AotKernelRegistration>();
    registration->load_code = [this, name, cfg, slot = registration.get()] {
      auto params = gfx_params(*materialize(name, cfg), name);
      slot->compiled_params = std::make_unique<gfx::GfxRuntime::RegisterParams>(std::move(params));
    };
    registrations_.push_back(registration);
    return std::make_unique<DeferredLayoutKernel<gfx::KernelImpl>>(std::move(load_layouts),
                                             graphics_, std::move(binding), nullptr,
                                             std::vector<uint32_t>{}, std::move(registration));
#else
    throw std::invalid_argument("Unified Vulkan prepare requires Vulkan support");
#endif
  }

  std::unique_ptr<aot::Kernel> compile(const std::string &name, const Facts &facts,
                                       const LayoutWords &layout = {}) {
    TI_COMPILE_PROFILER("cpp.unified.prepare.compile");
    auto &program = runtime_.get_unified_compiler();
    auto signature = directory_.signature(name);
    auto options = liong::json::parse(signature);
    CompileConfig cfg = program.compile_config();
    cfg.fast_math = bool(options["fast_math"]);
    cfg.debug = bool(options["debug"]);
    // This is target materialization for an AOT module. Use the same compact
    // descriptor addressing and private-value lowering as its native shader
    // producer. Target selection still belongs to the consumer's live device.
    cfg.spirv_omit_value_names = arch() == Arch::vulkan;
    cfg.advanced_optimization = bool(options["advanced_optimization"]);
    cfg.llvm_opt_level = int(options["llvm_opt_level"]);
    cfg.external_optimization_level = int(options["external_optimization_level"]);
    // Older signatures precede per-entry target compilation budgets.
    if (options.obj.inner.count("compile_tier"))
      cfg.compile_tier = string_value(options["compile_tier"]);
    if (options.obj.inner.count("full_simplify_global_iter_cap")) {
      cfg.full_simplify_global_iter_cap = int(options["full_simplify_global_iter_cap"]);
      // Produce already resolved the cap. A full-tier cap of one can be explicit.
      cfg.set_option_source(CompileOption::full_simplify_global_iter_cap,
                            CompileOptionSource::export_override);
    }
    const auto &caps = runtime_.get().get_caps();
    bool needs_bound_device = false;
    const bool unbound = facts.empty() && layout.empty();
    const auto &digest = directory_.entries.at(name);
    const auto &absent = directory_.absent_inputs(name);
    const auto trees = storage(name);
    cfg.llvm_aot_relocatable_roots = !trees.empty();
    auto read_source = [&] {
      return LLVM::read_unified_kernel(program, name, directory_.root_unit(digest),
          signature, facts, caps, [this](const std::string &hash) { return directory_.unit(hash); }, trees,
          absent, unbound ? &needs_bound_device : nullptr);
    };
    // A bound call has already read and verified the generic root and its
    // reachable helpers in this module. The snapshot is immutable. Keep the
    // specialized availability proof on the eager path for absent resources.
    std::function<std::unique_ptr<Kernel>()> restore_source;
    if (!unbound && arch() == Arch::vulkan && cfg.offline_cache &&
        directory_.is_immutable() && absent.empty() && !cfg.print_ir)
      restore_source = read_source;
    auto kernel = restore_source ? LLVM::declare_unified_kernel(
                                      program, name, digest, signature, facts, caps, trees)
                                 : read_source();
    if (unbound) {
      // Shape descriptors are bound for every ndarray. Only a value actually
      // read by this body or an imported helper can change the selected code.
      auto &reads = specialization_reads_[name];
      auto &layout_reads = layout_reads_[name];
      const auto dispatch_extents = arch_is_cpu(arch()) && !needs_bound_device && absent.empty()
                                        ? cpu_dispatch_extents(kernel->ir->as<Block>())
                                        : std::set<Stmt *>{};
      std::set<std::vector<int>> dispatch_paths, other_extents;
      const bool descriptor_offsets = arch() == Arch::vulkan &&
          !caps.get(DeviceCapability::spirv_has_physical_storage_buffer);
      auto note_descriptor = [&](const std::vector<int> &path, bool gradient) {
        if (!descriptor_offsets)
          return;
        // The launch context resolves this displacement inside the aligned
        // binding. The allocation address itself is never a code fact.
        auto member = path;
        member.push_back(gradient ? TypeFactory::GRAD_PTR_POS_IN_NDARRAY :
                                    TypeFactory::DATA_PTR_POS_IN_NDARRAY);
        descriptor_reads_[name].insert(std::move(member));
      };
      irpass::analysis::gather_statements(kernel->ir.get(), [&](Stmt *stmt) {
        if (auto *arg = stmt->cast<ArgLoadStmt>()) {
          if (!arg->is_ptr && arg->ret_type == PrimitiveType::i32) {
            reads.insert(arg->arg_id);
            if (!dispatch_extents.empty() && arg->arg_id.size() > 1)
              other_extents.insert(arg->arg_id);
          }
        } else if (auto *extent = stmt->cast<ExternalTensorShapeAlongAxisStmt>()) {
          auto path = extent->arg_id;
          path.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
          path.push_back(extent->axis);
          if (!dispatch_extents.empty())
            (dispatch_extents.count(stmt) ? dispatch_paths : other_extents).insert(path);
          reads.insert(std::move(path));
        } else if (auto *ptr = stmt->cast<ExternalPtrStmt>(); ptr && arch() == Arch::vulkan) {
          // Array addressing introduces descriptor loads in target codegen,
          // after the portable IR's explicit argument loads were specialized.
          auto *base = ptr->base_ptr->cast<ArgLoadStmt>();
          if (base)
            note_descriptor(base->arg_id, ptr->is_grad);
          if (base && ptr->ndim > 0) {
            auto path = base->arg_id;
            path.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
            path.push_back(0);
            for (int axis = 1; axis < ptr->ndim; ++axis) {
              path.back() = axis;
              layout_reads.insert(path);
            }
            if (ptr->runtime_affine && ptr->byte_stride == 0) {
              path.back() = TypeFactory::affine_mode_pos_in_ndarray(ptr->ndim);
              layout_reads.insert(path);
              for (int axis = 0; axis < ptr->ndim; ++axis) {
                path.back() = TypeFactory::stride_pos_in_ndarray(ptr->ndim, axis);
                layout_reads.insert(path);
              }
            }
          }
        } else if (auto *base = stmt->cast<ExternalTensorBasePtrStmt>()) {
          note_descriptor(base->arg_id, base->is_grad);
        }
        return false;
      });
      for (const auto &path : dispatch_paths)
        if (!other_extents.count(path))
          first_dispatch_counts_[name].try_emplace(path);
      if (needs_bound_device || !reads.empty() || !layout_reads.empty() || descriptor_reads_.count(name))
        return deferred(name, *kernel, cfg);
    }
    auto execution = program.compile_kernel_execution_handle(cfg, caps, *kernel, restore_source);
    if (launcher_) {
      auto &data = dynamic_cast<const LLVM::CompiledKernelData &>(execution->compiled());
      const auto &abi = data.get_internal_data();
      return std::make_unique<llvm_aot::KernelImpl>(launcher_.get(), KernelLaunchHandle{}, arch(),
          name, abi, LLVM::runtime_affine_arrays(*abi.compiled_data.module),
          [this, execution] { return register_llvm(*execution); });
    }
#ifdef TI_WITH_VULKAN
    auto params = gfx_params(*execution, name);
    if (!layout.empty()) {
      TI_COMPILE_PROFILER("cpp.unified.prepare.specialize_shader_layout");
      std::vector<vulkan::ShaderArgumentConstant> values;
      for (const auto &[offset, word] : layout)
        values.push_back({offset, 32, word});
      const auto folded = gfx::specialize_task_arguments(params, values, std::max(1, cfg.num_compile_threads),
                                                         /*layout_words_only=*/true);
      TI_DEBUG("Unified Vulkan specialization {} folded {} ABI loads for {} layout words",
               name, folded, values.size());
    }
    auto registration = std::make_shared<gfx::AotKernelRegistration>();
    registrations_.push_back(registration);
    return std::make_unique<gfx::KernelImpl>(graphics_, std::move(params), nullptr,
                                             std::vector<uint32_t>{}, std::move(registration));
#else
    throw std::invalid_argument("Unified Vulkan prepare requires Vulkan support");
#endif
  }
  ::Runtime &runtime_;
  std::shared_ptr<UnifiedLibrary> library_;
  const SelectedDirectory &directory_;
  std::unique_ptr<LLVM::KernelLauncher> launcher_;
  gfx::GfxRuntime *graphics_;
  LlvmRuntimeExecutor *executor_;
  uint64 *result_buffer_;
  TreeDimensions dimensions_;
  std::map<int, std::vector<SNode *>> nodes_;
  std::vector<std::shared_ptr<StoragePrototype>> compile_trees_;
  std::map<int, int> tree_bindings_;
  std::vector<int> physical_trees_;
  std::vector<LLVM::SNodeTreeLayout> llvm_layouts_;
  std::atomic<bool> used_{false};
  std::map<std::string, std::set<std::vector<int>>> specialization_reads_;
  std::map<std::string, std::map<std::vector<int>, std::optional<int>>> first_dispatch_counts_;
  std::map<std::string, std::set<std::vector<int>>> layout_reads_;
  std::map<std::string, std::set<std::vector<int>>> descriptor_reads_;
  std::map<std::tuple<std::string, Facts, LayoutWords>, std::unique_ptr<aot::Kernel>> specialized_;
#ifdef TI_WITH_VULKAN
  std::map<int, int> diagnostic_domains_;
  std::vector<std::shared_ptr<gfx::AotKernelRegistration>> registrations_;
#endif
};
}

bool is_unified_library(const VirtualDir &dir) {
  size_t ignored;
  return dir.get_file_size("unified_llvm.json", ignored);
}

std::unique_ptr<aot::Module> load_unified_library(
    ::Runtime &runtime, std::shared_ptr<const VirtualDir> source,
    std::unique_ptr<LLVM::KernelLauncher> launcher, gfx::GfxRuntime *graphics,
    LlvmRuntimeExecutor *executor, uint64 *result_buffer) {
  return std::make_unique<UnifiedModule>(runtime, std::make_shared<UnifiedLibrary>(std::move(source), runtime.arch),
                                         std::move(launcher), graphics, executor, result_buffer);
}
}  // namespace capi

TiAotModule ti_bind_unified_snode_module_ext(TiRuntime runtime, TiAotModule module,
                                             uint64_t description_size, const char *description) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module);
  TI_CAPI_ARGUMENT_NULL_RV(description);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto *owner = reinterpret_cast<AotModule *>(module);
  if (!rt.owns_module(owner))
    throw MemoryBindingError("Storage instance requires a live module in this runtime");
  auto *unified = dynamic_cast<capi::UnifiedModule *>(&owner->get());
  if (!unified)
    throw MemoryBindingError("Storage specialization requires a unified module");
  auto specification = liong::json::parse_complete(std::string(description, description_size));
  capi::TreeDimensions dimensions;
  for (const auto &[key, axes] : specification.fields()) {
    size_t end;
    int slot = std::stoi(key, &end);
    if (end != key.size() || !dimensions.emplace(slot, capi::Dimensions{}).second)
      throw MemoryBindingError("Invalid unified storage slot");
    for (const auto &[axis, value] : axes.fields()) {
      if (value.ty != liong::json::L_JSON_INT || value.num_int <= 0 ||
          value.num_int > std::numeric_limits<int>::max())
        throw MemoryBindingError("Storage dimensions require positive i32 integers");
      dimensions.at(slot).emplace(axis, int(value.num_int));
    }
  }
  return reinterpret_cast<TiAotModule>(new AotModule(rt, unified->bind_storage(std::move(dimensions))));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}
