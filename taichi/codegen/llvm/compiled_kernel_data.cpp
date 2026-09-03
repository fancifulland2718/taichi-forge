#include "taichi/codegen/llvm/compiled_kernel_data.h"

#include "llvm/IR/Verifier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"

namespace taichi::lang {

static std::unique_ptr<CompiledKernelData> new_llvm_compiled_kernel_data() {
  return std::make_unique<LLVM::CompiledKernelData>();
}

CompiledKernelData::Creator *const CompiledKernelData::llvm_creator =
    new_llvm_compiled_kernel_data;

namespace LLVM {

CompiledKernelData::CompiledKernelData(Arch arch, InternalData data)
    : arch_(arch), data_(std::move(data)) {
}

Arch CompiledKernelData::arch() const {
  return arch_;
}

std::unique_ptr<lang::CompiledKernelData> CompiledKernelData::clone() const {
  auto result = std::make_unique<CompiledKernelData>(arch_, data_);
  result->set_kernel_identity(kernel_identity());
  result->set_logical_kernel_identity(logical_kernel_identity());
  result->set_optimization_spec_identity(optimization_spec_identity());
  result->set_snode_relocation_descriptor(snode_relocation_descriptor());
  return result;
}

std::vector<int> CompiledKernelData::snode_tree_ids() const {
  return data_.used_snode_tree_ids;
}

bool CompiledKernelData::has_snode_tree_dependencies() const noexcept {
  return !data_.used_snode_tree_ids.empty();
}

bool CompiledKernelData::may_trigger_hash_overflow() const noexcept {
  return data_.may_trigger_hash_overflow;
}

std::size_t CompiledKernelData::task_count() const {
  return data_.compiled_data.tasks.size();
}

void CompiledKernelData::refresh_task_identities() {
  for (std::size_t index = 0; index < data_.compiled_data.tasks.size();
       ++index) {
    auto &task = data_.compiled_data.tasks[index];
    task.task_id = make_task_identity(index, task.task_type);
  }
}

namespace {

std::optional<std::int64_t> positive_geometry(int value) {
  if (value <= 0) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(value);
}

}  // namespace

std::vector<OffloadedTaskManifest> CompiledKernelData::task_manifest() const {
  std::vector<OffloadedTaskManifest> result;
  result.reserve(data_.compiled_data.tasks.size());
  const bool cpu_execution = arch_is_cpu(arch_);
  const bool cuda_execution = arch_ == Arch::cuda;
  for (std::size_t index = 0; index < data_.compiled_data.tasks.size();
       ++index) {
    const auto &task = data_.compiled_data.tasks[index];
    OffloadedTaskManifest item;
    item.task_id = task.task_id;
    item.logical_task_id = make_logical_task_identity(index, task.task_type);
    item.optimization_spec_id = optimization_spec_identity();
    item.task_name = task.name;
    item.arch = arch_;
    item.task_index = static_cast<std::uint32_t>(index);
    item.task_type = task.task_type;
    item.range_mapping =
        task.task_type == OffloadedTaskType::range_for
            ? (task.external_shared_staged
                   ? (task.external_shared_iteration_shape.size() == 2
                          ? "shared_tiled_2d_one_to_one"
                          : "shared_tiled_one_to_one")
                   : (task.one_to_one
                          ? (cuda_execution ? "device_bounded_grid_stride"
                                            : "one_to_one")
                          : (cpu_execution ? "cpu_scheduler"
                                           : "grid_stride")))
            : "not_applicable";
    if (task.constant_range_size >= 0) {
      item.constant_range_size = task.constant_range_size;
    }
    item.requested_grid_size = positive_geometry(task.requested_grid_dim);
    item.requested_block_size = positive_geometry(task.requested_block_dim);
    item.source_block_size_explicit = task.source_block_dim_explicit;
    item.requested_thread_local_mode =
        task.requested_thread_local_mode == 1
            ? "on"
            : (task.requested_thread_local_mode == 2 ? "off" : "auto");
    item.requested_cuda_min_blocks_per_sm =
        task.requested_cuda_min_blocks_per_sm;
    if (task.requested_cuda_max_registers >= 0) {
      item.requested_cuda_max_registers =
          task.requested_cuda_max_registers;
    }
    if (task.requested_grid_residency_waves > 0) {
      item.requested_grid_residency_waves =
          task.requested_grid_residency_waves;
    }
    item.requested_range_work_per_thread_target =
        task.requested_range_work_per_thread_target;
    item.requested_memory_strategy = task.requested_memory_strategy;
    item.sparse_list_op = task.sparse_list_op;
    item.sparse_list_snode_id = task.sparse_list_snode_id;
    item.sparse_list_parent_snode_id = task.sparse_list_parent_snode_id;
    if (task.sparse_list_parent_grid_bound > 0) {
      item.sparse_list_parent_grid_bound =
          task.sparse_list_parent_grid_bound;
    }
    item.may_mutate_sparse_topology = task.may_mutate_sparse_topology;
    item.sparse_mutation_snode_id = task.sparse_mutation_snode_id;
    if (task.external_shared_staged) {
      item.staged_external_arg_index = task.external_shared_arg_index;
      item.staged_halo_low = task.external_shared_halo_low;
      item.staged_halo_high = task.external_shared_halo_high;
      item.staged_external_arg_indices = task.external_shared_arg_indices;
      item.staged_halo_lows = task.external_shared_halo_lows;
      item.staged_halo_highs = task.external_shared_halo_highs;
      item.staged_byte_offsets = task.external_shared_byte_offsets;
      item.staged_element_bytes = task.external_shared_element_bytes;
      item.staged_scalar_bytes = task.external_shared_scalar_bytes;
      item.staged_element_shapes = task.external_shared_element_shapes;
      item.staged_iteration_shape =
          task.external_shared_iteration_shape;
      item.staged_iteration_origin =
          task.external_shared_iteration_origin;
      item.staged_tile_shape = task.external_shared_tile_shape;
      item.staged_halo_lows_nd = task.external_shared_halo_lows_nd;
      item.staged_halo_highs_nd = task.external_shared_halo_highs_nd;
      item.staged_access_offsets = task.external_shared_access_offsets;
    }
    if (cpu_execution) {
      item.actual_geometry_kind = "cpu_runtime_scheduler";
      item.actual_geometry_reason =
          "CPU tasks use the runtime worker scheduler, not a GPU grid";
    } else {
      item.static_shared_bytes = static_cast<std::uint64_t>(
          std::max(task.static_shared_array_bytes, 0));
      item.dynamic_shared_bytes = static_cast<std::uint64_t>(
          std::max(task.dynamic_shared_array_bytes, 0));
      item.thread_local_bytes = task.thread_local_bytes;
      item.selected_grid_size = positive_geometry(task.grid_dim);
      item.selected_block_size = positive_geometry(task.block_dim);
      item.actual_grid_size = item.selected_grid_size;
      item.actual_block_size = item.selected_block_size;
      if (task.external_shared_staged) {
        if (task.external_shared_iteration_shape.size() == 2) {
          item.actual_geometry_kind = "static_exact_tiled_2d";
          item.actual_geometry_reason =
              "the Graph shared-staged recipe materialized an exact "
              "two-dimensional tile grid";
        } else {
          item.actual_geometry_kind = "static_exact_tiled_1d";
          item.actual_geometry_reason =
              "the Graph shared-staged recipe materialized an exact "
              "one-dimensional tile grid";
        }
      } else if (cuda_execution && task.one_to_one) {
        item.actual_geometry_kind = "cuda_device_bounded_grid_stride";
        item.actual_geometry_reason =
            "the Graph payload uses the saturation-capped static grid and "
            "loads its logical range end from a device extent";
      } else {
        item.actual_geometry_kind = "static_direct";
        item.actual_geometry_reason =
            "ordinary direct launch uses the backend-selected geometry";
      }
    }
    result.push_back(std::move(item));
  }
  return result;
}

CompiledKernelData::Err CompiledKernelData::check() const {
  const auto &compiled_data = data_.compiled_data;
  const auto &tasks = compiled_data.tasks;
  if (!compiled_data.module) {
    return Err::kCompiledKernelDataBroken;
  }
  if (llvm::verifyModule(*compiled_data.module, &llvm::errs())) {
    return Err::kCompiledKernelDataBroken;
  }
  for (const auto &t : tasks) {
    if (compiled_data.module->getFunction(t.name) == nullptr) {
      return Err::kCompiledKernelDataBroken;
    }
  }
  return Err::kNoError;
}

CompiledKernelData::Err CompiledKernelData::load_impl(
    const CompiledKernelDataFile &file) {
  arch_ = file.arch();
  if (!arch_uses_llvm(arch_)) {
    return Err::kArchNotMatched;
  }
  try {
    liong::json::deserialize(liong::json::parse(file.metadata()), data_, true);
  } catch (const liong::json::JsonException &) {
    return Err::kParseMetadataFailed;
  }
  llvm::SMDiagnostic err;
  auto ret = llvm::parseAssemblyString(file.src_code(), err, llvm_ctx_);
  if (!ret) {  // File not found or Parse failed
    TI_DEBUG("Fail to parse llvm::Module from string: {}",
             err.getMessage().str());
    return Err::kParseSrcCodeFailed;
  }
  data_.compiled_data.module = std::move(ret);
  return Err::kNoError;
}

CompiledKernelData::Err CompiledKernelData::dump_impl(
    CompiledKernelDataFile &file) const {
  if (!data_.compiled_data.module) {
    return Err::kCompiledKernelDataBroken;
  }
  file.set_arch(arch_);
  try {
    file.set_metadata(liong::json::print(liong::json::serialize(data_)));
  } catch (const liong::json::JsonException &) {
    return Err::kSerMetadataFailed;
  }
  std::string str;
  llvm::raw_string_ostream oss(str);
  data_.compiled_data.module->print(oss, /*AAW=*/nullptr);
  file.set_src_code(std::move(str));
  return Err::kNoError;
}

}  // namespace LLVM
}  // namespace taichi::lang
