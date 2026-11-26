#include "taichi/runtime/llvm/unified_aot_module_builder.h"

#include <fstream>
#include <filesystem>
#include <unordered_set>

#include "taichi/aot/module_builder.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/codegen/llvm/unified_bitcode.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/struct/dense_snode_schema.h"

namespace taichi::lang::LLVM {
namespace {

void write_file(const std::string &path, const std::string &bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(bytes.data(), bytes.size());
  stream.close();
  TI_ERROR_IF(!stream, "Unable to write unified LLVM artifact {}", path);
}

class UnifiedAotModuleBuilder final : public AotModuleBuilder {
 public:
  UnifiedAotModuleBuilder(const CompileConfig &config,
                          const DeviceCapabilityConfig &caps)
      : config_(config), caps_(caps) {
  }

  std::map<int, int> exported_snode_tree_bindings() const override {
    return tree_bindings_;
  }
  int declare_storage(const SNode &root, const std::vector<DenseSNodeSchema::Parameter> &parameters,
                      const std::string &binding_key = {}) {
    const int id = root.get_snode_tree_id();
    auto [recipe, members] = DenseSNodeSchema::capture_storage(root, parameters);
    auto json = recipe.to_json();
    auto hash = aot_artifact_sha256(json);
    auto old = tree_bindings_.find(id);
    auto named = named_slots_.find(binding_key);
    const int next = static_cast<int>(layouts_.size());
    const int slot = old != tree_bindings_.end() ? old->second :
                     named != named_slots_.end() ? named->second : next;
    TI_ERROR_IF((slot != next && layouts_.at(slot) != hash) ||
                (named != named_slots_.end() && named->second != slot),
                "Unified storage declarations disagree on an existing binding");
    node_ordinals_.insert(members.begin(), members.end());
    schemas_.try_emplace(hash, std::move(json));
    if (!binding_key.empty())
      named_slots_.emplace(binding_key, slot);
    if (slot == next)
      layouts_.push_back(std::move(hash));
    tree_bindings_.emplace(id, slot);
    return slot;
  }
  void set_build_directory(const std::string &directory, bool cache_captures) {
    TI_ERROR_IF(!entries_.empty() || !directory_.empty(),
                "Select the unified build directory before adding entries");
    directory_ = directory;
    cache_captures_ = cache_captures;
    std::filesystem::create_directories(directory_);
    unit_stream_.open(directory_ / "units.bin", std::ios::binary);
    TI_ERROR_IF(!unit_stream_, "Unable to open unified LLVM unit stream");
    if (std::filesystem::exists(directory_ / "roots.tcb"))
      read_from_binary_file(roots_, (directory_ / "roots.tcb").string());
    if (std::filesystem::exists(directory_ / "helpers.tcb"))
      read_from_binary_file(helpers_, (directory_ / "helpers.tcb").string());
    if (std::ifstream index{directory_ / "baseline-members.index"}; index) {
      // The producer has pinned and hashed this archive for the entire link.
      // No consumer or LLVM pass needs the contents of restored units here.
      for (std::string member; index >> member;)
        baseline_members_.insert(std::move(member));
    } else if (std::filesystem::is_regular_file(directory_ / "baseline-path")) {
      baseline_ = io::VirtualDir::open(read_aot_artifact((directory_ / "baseline-path").string()));
    }
  }
  std::map<std::string, std::uint64_t> streaming_export_statistics() override {
    return {{"unified_root_emissions", emissions_}, {"unified_root_reuses", hits_},
            {"unified_capture_restores", restored_}, {"unified_referenced_units", referenced_units_}};
  }
  std::string build_object(const std::string &name) const {
    if (!cache_captures_ || !kernel_snode_tree_ids(name).empty())
      return {};
    using namespace liong::json;
    BuildObject object;
    object.unit = entries_.at(name);
    if (auto it = imports_.find(object.unit); it != imports_.end())
      object.imports = it->second;
    object.signature = entry_signatures_.at(name);
    object.effects = kernel_resource_effects(name);
    if (auto it = value_parameters_.find(name); it != value_parameters_.end())
      object.values = print(serialize(aot::make_value_parameter_artifact({*it})));
    if (auto it = argument_origins_.find(name); it != argument_origins_.end())
      object.origins = print(serialize(aot::make_argument_origin_artifact({*it})));
    return print(serialize(object));
  }
  bool restore_object(const std::string &name, const std::string &record) {
    if (!cache_captures_ || entries_.count(name))
      return false;
    using namespace liong::json;
    BuildObject object;
    aot::NamedValueParameters values;
    aot::NamedArgumentOrigins origins;
    try {
      deserialize(parse_complete(record), object, true);
      if (!object.values.empty()) {
        values = aot::parse_value_parameter_artifact(parse_complete(object.values));
        if (values.size() != 1 || !values.count(name))
          return false;
      }
      if (!object.origins.empty()) {
        origins = aot::parse_argument_origin_artifact(parse_complete(object.origins));
        if (origins.size() != 1 || !origins.count(name))
          return false;
      }
    } catch (const std::exception &) {
      return false;
    }
    // Stage the complete closure before publishing references or copying
    // any new bytes. An incomplete optional cache falls back to capture.
    std::map<std::string, std::string> missing;
    auto stage = [&](const std::string &digest) {
      if (written_units_.count(digest) || missing.count(digest))
        return true;
      if (baseline_members_.count("code_units/" + digest + ".bc"))
        return true;
      std::string bytes;
      if (!read_baseline_unit(digest, bytes))
        return false;
      missing.emplace(digest, std::move(bytes));
      return true;
    };
    if (!stage(object.unit) || !std::all_of(object.imports.begin(), object.imports.end(), stage))
      return false;
    // An intact archive already owns the signature as records/<SHA>. An
    // optional damaged baseline can still salvage individually valid objects.
    if (!signatures_.count(object.signature)) {
      std::string bytes;
      const auto member = "records/" + object.signature;
      if (!baseline_members_.count(member) && !read_baseline_member(member, object.signature, bytes))
        return false;
      signatures_.emplace(object.signature, std::move(bytes));
    }
    for (const auto &[digest, bytes] : missing)
      write_unit(digest, bytes);
    reference_baseline(object.unit);
    for (const auto &digest : object.imports)
      reference_baseline(digest);
    entry_signatures_.emplace(name, std::move(object.signature));
    if (!object.imports.empty())
      imports_.try_emplace(object.unit, std::move(object.imports));
    entries_.emplace(name, std::move(object.unit));
    value_parameters_.merge(values);
    argument_origins_.merge(origins);
    record_sealed_kernel_export(name, object.effects);
    ++restored_;
    return true;
  }
  void dump(const std::string &dir, const std::string &) const override {
    TI_ERROR_IF(!initial_root_data_.empty() || !root_storage_views_.empty(),
                "Unified LLVM needs logical member bindings for initialized "
                "roots or byte storage views");
    using namespace liong::json;
    if (directory_.empty()) {
      const auto units = taichi::join_path(dir, "code_units");
      taichi::create_directories(units);
      for (const auto &[hash, bytes] : units_)
        write_file(taichi::join_path(units, hash + ".bc"), bytes);
    } else {
      unit_stream_.flush();
      TI_ERROR_IF(!unit_stream_, "Unable to flush unified LLVM unit stream");
      std::ostringstream index;
      for (const auto &[digest, span] : written_units_) {
        if (span.second == 0)
          index << digest << " baseline\n";
        else
          index << digest << ' ' << span.first << ' ' << span.second << '\n';
      }
      write_file((directory_ / "units.index").string(), index.str());
      if (cache_captures_) {
        write_to_binary_file(roots_, (directory_ / "roots.tcb").string());
        std::map<std::string, std::string> live_helpers;
        for (const auto &[key, digest] : helpers_)
          if (written_units_.count(digest))
            live_helpers.emplace(key, digest);
        write_to_binary_file(live_helpers, (directory_ / "helpers.tcb").string());
      }
    }
    write_file(taichi::join_path(dir, "unified_llvm.json"),
               print(serialize(entries_)));
    const auto signatures = taichi::join_path(dir, "signatures");
    taichi::create_directories(signatures);
    std::ostringstream referenced_signatures;
    for (const auto &[hash, description] : signatures_) {
      if (description.empty())
        referenced_signatures << hash << '\n';
      else
        write_file(taichi::join_path(signatures, hash + ".json"), description);
    }
    if (referenced_signatures.tellp() > 0)
      write_file((directory_ / "signatures.index").string(), referenced_signatures.str());
    write_file(taichi::join_path(dir, "unified_abi.json"),
               print(serialize(entry_signatures_)));
    if (!layouts_.empty()) {
      JsonObject recipes, bindings;
      for (const auto &[digest, schema] : schemas_)
        recipes.inner.emplace(digest, parse(schema));
      for (const auto &[name, trees] : entry_trees_) {
        std::vector<int> local;
        for (int tree : trees)
          local.push_back(tree_bindings_.at(tree));
        bindings.inner.emplace(name, serialize(local));
      }
      write_file(taichi::join_path(dir, "unified_storage.json"),
                 print(JsonValue(
                     JsonObject{{"schemas", JsonValue(std::move(recipes))},
                                {"trees", serialize(layouts_)},
                                {"entries", JsonValue(std::move(bindings))}})));
    }
    dump_graph(dir);
    aot::write_struct_layout_artifact(ndarray_layouts_, dir);
    aot::write_value_parameter_artifact(value_parameters_, dir);
    aot::write_argument_origin_artifact(argument_origins_, dir);
  }

 private:
  const CompileConfig &config_;
  DeviceCapabilityConfig caps_;
  std::map<std::string, std::string> entries_, units_, helpers_;
  std::map<std::string, std::string> signatures_, entry_signatures_;
  struct BuildObject {
    std::string unit, signature, values, origins;
    std::vector<std::string> imports;
    GraphKernelMetadata effects;
    TI_IO_DEF(unit, signature, values, origins, imports, effects);
  };
  std::map<std::string, std::vector<std::string>> imports_;
  bool cache_captures_{false};
  struct Root {
    std::string unit;
    GraphKernelMetadata effects;
    TI_IO_DEF(unit, effects);
  };
  std::filesystem::path directory_;
  std::unique_ptr<io::VirtualDir> baseline_;
  std::unordered_set<std::string> baseline_members_;
  std::uint64_t referenced_units_{0};
  std::uint64_t emissions_{0}, hits_{0}, restored_{0};
  std::map<std::string, Root> roots_;
  // Private producer staging. The final TCM retains its content-addressed
  // members and deterministic digest order regardless of emission order.
  mutable std::ofstream unit_stream_;
  uint64_t unit_end_{0};
  std::map<std::string, std::pair<uint64_t, uint64_t>> written_units_;
  std::map<int, int> tree_bindings_, node_ordinals_;
  std::map<std::string, int> named_slots_;
  std::map<std::string, std::string> schemas_;
  std::vector<std::string> layouts_;
  std::map<std::string, std::vector<int>> entry_trees_;

  bool read_baseline_member(const std::string &member, const std::string &digest, std::string &bytes) const {
    if (!baseline_ || digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string::npos)
      return false;
    size_t size = 0;
    if (!baseline_->get_file_size(member, size))
      return false;
    bytes.resize(size);
    return baseline_->load_file(member, bytes.data(), size) == size &&
           aot_artifact_sha256(bytes) == digest;
  }

  bool read_baseline_unit(const std::string &digest, std::string &bytes) const {
    return read_baseline_member("code_units/" + digest + ".bc", digest, bytes);
  }

  bool reference_baseline(const std::string &digest) {
    if (!baseline_members_.count("code_units/" + digest + ".bc"))
      return false;
    // A bitcode body is never empty. A zero span denotes a link reference.
    if (written_units_.try_emplace(digest, 0, 0).second)
      ++referenced_units_;
    return true;
  }

  void write_unit(const std::string &digest, const std::string &bytes) {
    if (reference_baseline(digest))
      return;
    if (written_units_.try_emplace(digest, unit_end_, bytes.size()).second) {
      unit_stream_.write(bytes.data(), bytes.size());
      TI_ERROR_IF(!unit_stream_, "Unable to stage unified LLVM unit {}", digest);
      unit_end_ += bytes.size();
    }
  }

  bool available(const std::string &digest) {
    if (directory_.empty() || written_units_.count(digest) || reference_baseline(digest))
      return true;
    std::string bytes;
    if (!read_baseline_unit(digest, bytes))
      return false;
    write_unit(digest, bytes);
    return true;
  }

  void register_tree(const SNode &root) {
    if (!tree_bindings_.count(root.get_snode_tree_id()))
      declare_storage(root, {});
  }

  void add_per_backend(const std::string &name, Kernel *kernel) override {
    TI_ERROR_IF(kernel->get_offload_execution_plan(),
                "Unified LLVM launch policy is not lowered for {}", name);
    // Keep logical parameter types. prepare supplies the concrete layout of
    // the RuntimeContext, packed arguments and return buffer on its device.
    using namespace liong::json;
    std::vector<std::pair<std::vector<int>, CallableBase::Parameter>> nested;
    for (const auto &[path, parameter] : kernel->nested_parameters)
      if (path.size() > 1)
        nested.emplace_back(path, parameter);
    std::sort(nested.begin(), nested.end(), [](const auto &a, const auto &b) {
      return a.first < b.first;
    });
    const auto request = resolve_kernel_compile_request(
        config_, caps_, kernel->get_compile_tier_override(),
        CompilePurpose::aot, compile_options_);
    const auto &cc = request.config();
    auto signature = print(JsonValue(JsonObject{
        {"arguments", serialize(kernel->parameter_list)},
        {"nested", serialize(nested)}, {"returns", serialize(kernel->rets)},
        {"fast_math", serialize(cc.fast_math)}, {"debug", serialize(cc.debug)},
        {"advanced_optimization", serialize(cc.advanced_optimization)},
        {"llvm_opt_level", serialize(cc.llvm_opt_level)},
        {"external_optimization_level", serialize(cc.external_optimization_level)}}));
    auto signature_hash = aot_artifact_sha256(signature);
    signatures_.try_emplace(signature_hash, std::move(signature));
    entry_signatures_.emplace(name, std::move(signature_hash));
    // The frontend dependency recorder includes field references inside Expr
    // nodes. Its ids are only used to resolve this build's resource slots.
    const auto trees = kernel->source_snode_tree_dependencies();
    if (!trees.empty()) {
      std::set<int> slots;
      for (int id : trees) {
        if (!tree_bindings_.count(id))
          register_tree(*kernel->program->get_snode_root(id));
        TI_ERROR_IF(!slots.insert(tree_bindings_.at(id)).second,
                    "Independent storage trees used by one entry require distinct bindings");
      }
      entry_trees_.emplace(name, trees);
    }
    // The default Python fast tier limits target compilation at prepare.
    // Compacting a private, target-neutral archive unit is independent of
    // that budget. Honor explicit optimization settings in either phase.
    const bool frontend_fast_default =
        cc.option_source(CompileOption::compile_tier) == CompileOptionSource::frontend_default &&
        cc.option_source(CompileOption::advanced_optimization) == CompileOptionSource::derived;
    const bool simplify = !cc.debug && cc.llvm_opt_level > 0 &&
                          (cc.advanced_optimization || frontend_fast_default);
    auto key = trees.empty() && kernel->ir_is_ast()
                   ? unified_leaf_identity(*kernel, simplify)
                   : std::string{};
    auto found = roots_.find(key);
    if (found != roots_.end() && available(found->second.unit)) {
      ++hits_;
      entries_.emplace(name, found->second.unit);
      record_kernel_resource_effects(name, found->second.effects);
      return;
    }
    ++emissions_;
    Root root;
    std::vector<std::string> imports;
    auto bytes = emit_unified_bitcode(*kernel, root.effects, units_, helpers_,
                                      trees, node_ordinals_, simplify, cache_captures_ ? &imports : nullptr,
                                      [this](const std::string &digest) { return available(digest); });
    root.unit = aot_artifact_sha256(bytes);
    if (!imports.empty())
      imports_.try_emplace(root.unit, std::move(imports));
    units_.try_emplace(root.unit, std::move(bytes));
    if (!directory_.empty()) {
      for (const auto &[hash, body] : units_)
        write_unit(hash, body);
      units_.clear();
    }
    entries_.emplace(name, root.unit);
    record_kernel_resource_effects(name, root.effects);
    if (!key.empty())
      roots_.insert_or_assign(std::move(key), std::move(root));
  }

  const std::vector<int> &kernel_snode_tree_ids(
      const std::string &name) const override {
    static const std::vector<int> empty;
    TI_ERROR_IF(!entries_.count(name), "Unknown unified LLVM kernel {}", name);
    auto it = entry_trees_.find(name);
    return it == entry_trees_.end() ? empty : it->second;
  }

  void add_field_per_backend(const std::string &,
                             const SNode *node,
                             bool,
                             DataType,
                             std::vector<int>,
                             int,
                             int) override {
    register_tree(*node->get_root());
  }

  void add_per_backend_tmpl(const std::string &name,
                            const std::string &key,
                            Kernel *kernel) override {
    add_per_backend(name + "__tmpl__" + key, kernel);
  }
};
}  // namespace

int declare_unified_storage(AotModuleBuilder &builder, const SNode &root,
                            const std::vector<DenseSNodeSchema::Parameter> &parameters,
                            const std::string &binding_key) {
  auto &unified = dynamic_cast<UnifiedAotModuleBuilder &>(builder);
  return unified.declare_storage(root, parameters, binding_key);
}

std::string unified_build_object(AotModuleBuilder &builder, const std::string &name) {
  auto *unified = dynamic_cast<UnifiedAotModuleBuilder *>(&builder);
  return unified ? unified->build_object(name) : std::string{};
}

bool restore_unified_build_object(AotModuleBuilder &builder, const std::string &name,
                                  const std::string &record) {
  auto *unified = dynamic_cast<UnifiedAotModuleBuilder *>(&builder);
  return unified && unified->restore_object(name, record);
}

void set_unified_build_directory(AotModuleBuilder &builder,
                                 const std::string &directory, bool cache_captures) {
  auto *unified = dynamic_cast<UnifiedAotModuleBuilder *>(&builder);
  TI_ERROR_IF(!unified, "A shared LLVM body directory requires a unified builder");
  unified->set_build_directory(directory, cache_captures);
}

std::unique_ptr<AotModuleBuilder> make_unified_aot_module_builder(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps) {
  return std::make_unique<UnifiedAotModuleBuilder>(config, caps);
}
}  // namespace taichi::lang::LLVM
