#include "offline_cache_util.h"

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/kernel.h"
#include "taichi/rhi/device_capability.h"
#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif
#include "taichi/system/profiler.h"

#include "picosha2.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <type_traits>
#include <vector>

namespace taichi::lang {

std::optional<std::pair<std::uint8_t, int>> get_offline_cache_implicit_target(
    Arch arch) {
#if defined(TI_WITH_CUDA)
  if (arch == Arch::cuda) {
    // The provider and actual LLVM target are implicit inputs in addition to
    // explicit capabilities. Retain both in cached-context validation.
    return std::make_pair(
        static_cast<std::uint8_t>(
            CUDADriver::get_instance_without_context().get_provider()),
        CUDAContext::get_instance().get_codegen_compute_capability());
  }
#endif
  return std::nullopt;
}

namespace {
template <typename T>
bool same_cache_field(const T &a, const T &b) {
  if constexpr (std::is_floating_point_v<T>) {
    // Serialization preserves signed zero and NaN payload bits.
    return std::memcmp(&a, &b, sizeof(T)) == 0;
  } else {
    return a == b;
  }
}
}  // namespace

bool same_offline_cache_compile_config(const CompileConfig &config,
                                       const CompileConfig &other) {
#define CACHE_FIELD(name)                         \
  if (!same_cache_field(config.name, other.name)) \
  return false
#define CACHE_TYPE(name) CACHE_FIELD(name)
#define CACHE_SORTED(name) CACHE_FIELD(name)
#define CACHE_VALUE(value)
#define CACHE_CUDA_TARGET()
#include "taichi/analysis/compile_config_key_fields.inc.h"
#undef CACHE_CUDA_TARGET
#undef CACHE_VALUE
#undef CACHE_SORTED
#undef CACHE_TYPE
#undef CACHE_FIELD
  return true;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_parameter_list(
    const std::vector<CallableBase::Parameter> &parameter_list) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(parameter_list);
  serializer.finalize();
  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_rets(
    const std::vector<CallableBase::Ret> &ret_list) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(ret_list);
  serializer.finalize();
  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_compile_config(
    const CompileConfig &config) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
#define CACHE_FIELD(name) serializer(config.name)
#define CACHE_TYPE(name) serializer(config.name.to_string())
#define CACHE_SORTED(name)                   \
  {                                          \
    auto sorted = config.name;               \
    std::sort(sorted.begin(), sorted.end()); \
    serializer(sorted);                      \
  }
#define CACHE_VALUE(value) serializer(value)
#define CACHE_CUDA_TARGET()                                             \
  {                                                                     \
    const auto target = get_offline_cache_implicit_target(config.arch); \
    if (target.has_value()) {                                           \
      serializer(target->first);                                        \
      serializer(target->second);                                       \
    }                                                                   \
  }
#include "taichi/analysis/compile_config_key_fields.inc.h"
#undef CACHE_CUDA_TARGET
#undef CACHE_VALUE
#undef CACHE_SORTED
#undef CACHE_TYPE
#undef CACHE_FIELD
  serializer.finalize();

  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_device_caps(
    const DeviceCapabilityConfig &caps) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(caps.devcaps);
  serializer.finalize();
  return serializer.data;
}

static void get_offline_cache_key_of_snode_impl(
    const SNode *snode,
    BinaryOutputSerializer &serializer,
    std::unordered_set<int> &visited) {
  if (auto iter = visited.find(snode->id); iter != visited.end()) {
    serializer(snode->id);  // Use snode->id as placeholder to identify a snode
    return;
  }

  visited.insert(snode->id);
  for (auto &c : snode->ch) {
    get_offline_cache_key_of_snode_impl(c.get(), serializer, visited);
  }
  for (int i = 0; i < taichi_max_num_indices; ++i) {
    auto &extractor = snode->extractors[i];
    serializer(extractor.num_elements_from_root);
    serializer(extractor.shape);
    serializer(extractor.acc_shape);
    serializer(extractor.active);
  }
  serializer(snode->index_offsets);
  serializer(snode->num_active_indices);
  serializer(snode->physical_index_position);
  serializer(snode->id);
  serializer(snode->depth);
  serializer(snode->name);
  serializer(snode->num_cells_per_container);
  // C-1 (2026-05): per-pointer-SNode 池容量 hint 影响 SPIR-V layout 偏移，
  // 必须纳入 cache key。-1 默认值哈希稳定，旧 SNodeTree 不受影响。
  serializer(snode->vk_max_active_hint);
  serializer(snode->hash_expected_active_hint);
  serializer(snode->chunk_size);
  serializer(snode->cell_size_bytes);
  serializer(snode->offset_bytes_in_parent_cell);
  serializer(snode->dt->to_string());
  serializer(snode->has_ambient);
  if (!snode->ambient_val.dt->is_primitive(PrimitiveTypeID::unknown)) {
    serializer(snode->ambient_val.stringify());
  }
  if (snode->grad_info && !snode->grad_info->is_primal()) {
    if (auto *adjoint_snode = snode->grad_info->adjoint_snode()) {
      get_offline_cache_key_of_snode_impl(adjoint_snode, serializer, visited);
    }
    if (auto *dual_snode = snode->grad_info->dual_snode()) {
      get_offline_cache_key_of_snode_impl(dual_snode, serializer, visited);
    }
  }
  if (snode->physical_type) {
    serializer(snode->physical_type->to_string());
  }
  serializer(snode->id_in_bit_struct);
  serializer(snode->is_bit_level);
  serializer(snode->is_path_all_dense);
  serializer(snode->node_type_name);
  serializer(snode->type);
  serializer(snode->_morton);
  serializer(snode->get_snode_tree_id());
}

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode) {
  TI_ASSERT(snode);

  BinaryOutputSerializer serializer;
  serializer.initialize();
  {
    std::unordered_set<int> visited;
    get_offline_cache_key_of_snode_impl(snode, serializer, visited);
  }
  serializer.finalize();

  picosha2::hash256_one_by_one hasher;
  hasher.process(serializer.data.begin(), serializer.data.end());
  hasher.finish();

  return picosha2::get_hash_hex_string(hasher);
}

namespace {

std::string get_hashed_offline_cache_key_impl(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel,
    bool include_optimization_spec) {
  TI_AUTO_PROF;
  std::vector<std::uint8_t> kernel_params_string, kernel_rets_string;
  std::string kernel_body_string;
  if (kernel) {  // param_list, rets, body
    kernel_params_string =
        get_offline_cache_key_of_parameter_list(kernel->parameter_list);
    kernel_rets_string = get_offline_cache_key_of_rets(kernel->rets);
    if (kernel->has_cached_offline_cache_body()) {
      kernel_body_string = kernel->get_cached_offline_cache_body();
    } else {
      std::ostringstream oss;
      {
        TI_PROFILER("gen_offline_cache_key.body");
        gen_offline_cache_key(kernel->ir.get(), &oss);
      }
      kernel_body_string = oss.str();
      kernel->set_offline_cache_body(kernel_body_string);
    }
  }

  auto compile_config_key = get_offline_cache_key_of_compile_config(config);
  auto device_caps_key = get_offline_cache_key_of_device_caps(caps);
  std::string autodiff_mode =
      std::to_string(static_cast<std::size_t>(kernel->autodiff_mode));
  const std::string optimization_spec =
      include_optimization_spec && kernel
          ? kernel->optimization_spec_cache_key()
          : "";
  picosha2::hash256_one_by_one hasher;
  std::string schema_tag =
      std::string(include_optimization_spec ? "tcs:" : "tcs-semantic:") +
      std::to_string(kOfflineCacheSchemaVersion);
  hasher.process(schema_tag.begin(), schema_tag.end());
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(device_caps_key.begin(), device_caps_key.end());
  hasher.process(kernel_params_string.begin(), kernel_params_string.end());
  hasher.process(kernel_rets_string.begin(), kernel_rets_string.end());
  hasher.process(kernel_body_string.begin(), kernel_body_string.end());
  hasher.process(autodiff_mode.begin(), autodiff_mode.end());
  hasher.process(optimization_spec.begin(), optimization_spec.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  // Cache keys use T; host-only semantic identities use S. Both must start
  // with a letter because downstream metadata treats them as identifiers.
  res.insert(res.begin(), include_optimization_spec ? 'T' : 'S');
  return res;
}

std::string get_hashed_offline_cache_key_context_impl(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel,
    bool include_optimization_spec) {
  TI_AUTO_PROF;
  std::vector<std::uint8_t> kernel_params_string, kernel_rets_string;
  if (kernel) {
    kernel_params_string =
        get_offline_cache_key_of_parameter_list(kernel->parameter_list);
    kernel_rets_string = get_offline_cache_key_of_rets(kernel->rets);
  }

  auto compile_config_key = get_offline_cache_key_of_compile_config(config);
  auto device_caps_key = get_offline_cache_key_of_device_caps(caps);
  std::string autodiff_mode =
      kernel ? std::to_string(static_cast<std::size_t>(kernel->autodiff_mode))
             : "";
  const std::string optimization_spec =
      include_optimization_spec && kernel
          ? kernel->optimization_spec_cache_key()
          : "";

  picosha2::hash256_one_by_one hasher;
  std::string schema_tag = std::string(include_optimization_spec
                                           ? "tcs-ctx:"
                                           : "tcs-ctx-semantic:") +
                           std::to_string(kOfflineCacheSchemaVersion);
  hasher.process(schema_tag.begin(), schema_tag.end());
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(device_caps_key.begin(), device_caps_key.end());
  hasher.process(kernel_params_string.begin(), kernel_params_string.end());
  hasher.process(kernel_rets_string.begin(), kernel_rets_string.end());
  hasher.process(autodiff_mode.begin(), autodiff_mode.end());
  hasher.process(optimization_spec.begin(), optimization_spec.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), include_optimization_spec ? 'C' : 'S');
  return res;
}

}  // namespace

std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel) {
  return get_hashed_offline_cache_key_impl(
      config, caps, kernel, /*include_optimization_spec=*/true);
}

std::string get_hashed_offline_cache_semantic_key(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel) {
  return get_hashed_offline_cache_key_impl(
      config, caps, kernel, /*include_optimization_spec=*/false);
}

std::string get_hashed_offline_cache_key_context(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel) {
  return get_hashed_offline_cache_key_context_impl(
      config, caps, kernel, /*include_optimization_spec=*/true);
}

std::string get_hashed_offline_cache_semantic_key_context(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel) {
  return get_hashed_offline_cache_key_context_impl(
      config, caps, kernel, /*include_optimization_spec=*/false);
}

}  // namespace taichi::lang
