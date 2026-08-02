#include "taichi/codegen/spirv/compiled_kernel_data.h"

namespace taichi::lang {

static std::unique_ptr<CompiledKernelData> new_spirv_compiled_kernel_data() {
  return std::make_unique<spirv::CompiledKernelData>();
}

CompiledKernelData::Creator *const CompiledKernelData::spriv_creator =
    new_spirv_compiled_kernel_data;

namespace spirv {

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
  return data_.metadata.used_snode_tree_ids;
}

std::size_t CompiledKernelData::task_count() const {
  return data_.metadata.kernel_attribs.tasks_attribs.size();
}

void CompiledKernelData::refresh_task_identities() {
  auto &tasks = data_.metadata.kernel_attribs.tasks_attribs;
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    tasks[index].task_id = make_task_identity(index, tasks[index].task_type);
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
  const auto &tasks = data_.metadata.kernel_attribs.tasks_attribs;
  std::vector<OffloadedTaskManifest> result;
  result.reserve(tasks.size());
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    const auto &task = tasks[index];
    OffloadedTaskManifest item;
    item.task_id = task.task_id;
    item.task_name = task.name;
    item.arch = arch_;
    item.task_index = static_cast<std::uint32_t>(index);
    item.task_type = task.task_type;
    item.requested_grid_size = positive_geometry(task.requested_grid_dim);
    item.requested_block_size = positive_geometry(task.requested_block_dim);
    item.selected_block_size =
        positive_geometry(task.advisory_num_threads_per_group);
    if (task.advisory_total_num_threads > 0 &&
        task.advisory_num_threads_per_group > 0) {
      item.selected_grid_size =
          (static_cast<std::int64_t>(task.advisory_total_num_threads) +
           task.advisory_num_threads_per_group - 1) /
          task.advisory_num_threads_per_group;
    }
    item.actual_grid_size = item.selected_grid_size;
    item.actual_block_size = item.selected_block_size;
    item.actual_geometry_kind = "static_direct";
    item.actual_geometry_reason =
        "ordinary direct launch uses the backend-selected geometry";
    item.static_shared_bytes = static_cast<std::uint64_t>(
        std::max(task.static_shared_array_bytes, 0));
    result.push_back(std::move(item));
  }
  return result;
}

CompiledKernelData::Err CompiledKernelData::load_impl(
    const CompiledKernelDataFile &file) {
  arch_ = file.arch();
  if (!arch_uses_spirv(arch_)) {
    return Err::kArchNotMatched;
  }
  try {
    liong::json::deserialize(liong::json::parse(file.metadata()),
                             data_.metadata, true);
  } catch (const liong::json::JsonException &) {
    return Err::kParseMetadataFailed;
  }
  return str2src(file.src_code(), data_.src);
}

CompiledKernelData::Err CompiledKernelData::dump_impl(
    CompiledKernelDataFile &file) const {
  file.set_arch(arch_);
  try {
    file.set_metadata(
        liong::json::print(liong::json::serialize(data_.metadata)));
  } catch (const liong::json::JsonException &) {
    return Err::kSerMetadataFailed;
  }
  std::string str;
  Err err = src2str(data_.src, str);
  file.set_src_code(std::move(str));
  return err;
}

CompiledKernelData::Err CompiledKernelData::src2str(
    const InternalData::Source &src,
    std::string &result) {
  std::ostringstream oss;
  write_to_binary_stream(src, oss);
  if (oss) {
    result = oss.str();
    return Err::kNoError;
  }
  return Err::kSerSrcCodeFailed;
}

CompiledKernelData::Err CompiledKernelData::str2src(
    const std::string &str,
    InternalData::Source &result) {
  return read_from_binary(result, str.data(), str.size())
             ? Err::kNoError
             : Err::kParseSrcCodeFailed;
}

}  // namespace spirv
}  // namespace taichi::lang
