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
  return result;
}

std::vector<int> CompiledKernelData::snode_tree_ids() const {
  return data_.used_snode_tree_ids;
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
    item.task_name = task.name;
    item.arch = arch_;
    item.task_index = static_cast<std::uint32_t>(index);
    item.task_type = task.task_type;
    item.range_mapping =
        task.task_type == OffloadedTaskType::range_for
            ? (task.one_to_one ? (cuda_execution
                                      ? "device_bounded_grid_stride"
                                      : "one_to_one")
                               : (cpu_execution ? "cpu_scheduler"
                                                : "grid_stride"))
            : "not_applicable";
    item.requested_grid_size = positive_geometry(task.requested_grid_dim);
    item.requested_block_size = positive_geometry(task.requested_block_dim);
    if (cpu_execution) {
      item.actual_geometry_kind = "cpu_runtime_scheduler";
      item.actual_geometry_reason =
          "CPU tasks use the runtime worker scheduler, not a GPU grid";
    } else {
      item.static_shared_bytes = static_cast<std::uint64_t>(
          std::max(task.static_shared_array_bytes, 0));
      item.dynamic_shared_bytes = static_cast<std::uint64_t>(
          std::max(task.dynamic_shared_array_bytes, 0));
      item.selected_grid_size = positive_geometry(task.grid_dim);
      item.selected_block_size = positive_geometry(task.block_dim);
      item.actual_grid_size = item.selected_grid_size;
      item.actual_block_size = item.selected_block_size;
      if (cuda_execution && task.one_to_one) {
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
