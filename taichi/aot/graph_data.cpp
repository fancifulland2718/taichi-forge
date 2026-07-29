#include "taichi/aot/graph_data.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/storage_view.h"
#include "taichi/program/texture.h"
#include "taichi/program/kernel.h"
#include "taichi/program/matrix.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/type_factory.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>

#if defined(TI_WITH_LLVM)
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#endif

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

#if defined(TI_WITH_VULKAN)
#include "taichi/runtime/gfx/kernel_launcher.h"
#endif

namespace taichi::lang {
namespace aot {

namespace {

std::vector<SNodeTreeDependency> collect_snode_tree_dependencies(
    const std::vector<CompiledDispatch> &dispatches) {
  std::vector<SNodeTreeDependency> dependencies;
  for (const auto &dispatch : dispatches) {
    dependencies.insert(dependencies.end(),
                        dispatch.snode_tree_dependencies.begin(),
                        dispatch.snode_tree_dependencies.end());
  }
  std::sort(dependencies.begin(), dependencies.end());
  dependencies.erase(
      std::unique(dependencies.begin(), dependencies.end()),
      dependencies.end());
  return dependencies;
}

Program *jit_graph_program(const CompiledGraph &graph) {
  Program *program = nullptr;
  for (const auto &dispatch : graph.dispatches) {
    if (dispatch.ti_kernel == nullptr) {
      continue;
    }
    if (program == nullptr) {
      program = dispatch.ti_kernel->program;
    } else {
      TI_ERROR_IF(program != dispatch.ti_kernel->program,
                  "A JIT Graph cannot dispatch kernels from multiple "
                  "Programs.");
    }
  }
  return program;
}

template <typename T, std::size_t InlineCapacity = 4>
class InlineUniqueViewList {
 public:
  void add(const T *view) {
    const T *const *current = data();
    for (std::size_t i = 0; i < size(); ++i) {
      if (current[i] == view) {
        return;
      }
    }
    if (overflow_.empty() && inline_count_ < InlineCapacity) {
      inline_[inline_count_++] = view;
      return;
    }
    if (overflow_.empty()) {
      overflow_.reserve(InlineCapacity * 2);
      overflow_.insert(overflow_.end(), inline_.begin(),
                       inline_.begin() + inline_count_);
    }
    overflow_.push_back(view);
  }

  bool empty() const noexcept {
    return inline_count_ == 0 && overflow_.empty();
  }

  std::size_t size() const noexcept {
    return overflow_.empty() ? inline_count_ : overflow_.size();
  }

  const T *const *data() const noexcept {
    return overflow_.empty() ? inline_.data() : overflow_.data();
  }

 private:
  std::array<const T *, InlineCapacity> inline_{};
  std::size_t inline_count_{0};
  std::vector<const T *> overflow_;
};

struct GraphRuntimeResourceViews {
  InlineUniqueViewList<Ndarray> ndarrays;
  InlineUniqueViewList<storage::RuntimeStorageArgument> runtime_storage;
  InlineUniqueViewList<Texture> textures;

  bool empty() const noexcept {
    return ndarrays.empty() && runtime_storage.empty() && textures.empty();
  }
};

GraphRuntimeResourceViews graph_runtime_resource_views(
    const std::unordered_map<std::string, IValue> &args,
    Program *expected_program) {
  GraphRuntimeResourceViews views;
  for (const auto &[name, value] : args) {
    (void)name;
    if (value.tag == ArgKind::kNdarray) {
      if (value.runtime_storage != nullptr) {
        views.runtime_storage.add(value.runtime_storage);
        continue;
      }
      auto *view = reinterpret_cast<const Ndarray *>(value.val);
      TI_ERROR_IF(view == nullptr, "Graph received a null Ndarray runtime arg");
      Program *owner = view->owning_program();
      if (owner == nullptr) {
        // AOT DeviceAllocation-backed views preserve their existing external
        // ownership contract and never enter a Program registry.
        continue;
      }
      TI_ERROR_IF(expected_program != nullptr && owner != expected_program,
                  "Graph Ndarray runtime arguments must belong to the Graph's "
                  "Program");
      views.ndarrays.add(view);
      continue;
    }
    if (value.tag != ArgKind::kTexture) {
      continue;
    }
    auto *view = reinterpret_cast<const Texture *>(value.val);
    TI_ERROR_IF(view == nullptr, "Graph received a null Texture runtime arg");
    Program *owner = view->owning_program();
    if (owner == nullptr) {
      continue;
    }
    TI_ERROR_IF(expected_program != nullptr && owner != expected_program,
                "Graph Texture runtime arguments must belong to the Graph's "
                "Program");
    views.textures.add(view);
  }
  return views;
}

const CompiledKernelData *get_or_compile_cached_kernel(
    const CompiledDispatch &dispatch,
    const CompileConfig &compile_config,
    CompiledGraphJITCachedKernel &cached,
    bool cache_compiled_kernel_data) {
  auto *prog = dispatch.ti_kernel->program;
  const CompiledKernelData *compiled_kernel_data =
      cache_compiled_kernel_data ? cached.compiled_kernel_data : nullptr;
  if (compiled_kernel_data == nullptr && !cached.kernel_key.empty()) {
    compiled_kernel_data = prog->find_cached_kernel(
        compile_config, cached.kernel_key, *dispatch.ti_kernel);
    if (cache_compiled_kernel_data) {
      cached.compiled_kernel_data = compiled_kernel_data;
    }
  }
  if (compiled_kernel_data == nullptr) {
    compiled_kernel_data = &prog->compile_kernel(
        compile_config, prog->get_device_caps(), *dispatch.ti_kernel);
    if (cached.kernel_key.empty()) {
      cached.kernel_key = dispatch.ti_kernel->get_cached_kernel_key();
    }
    if (cache_compiled_kernel_data) {
      cached.compiled_kernel_data = compiled_kernel_data;
    }
  }
  if (cached.task_count == std::numeric_limits<std::uint32_t>::max()) {
    const std::size_t task_count = compiled_kernel_data->task_count();
    TI_ASSERT(task_count < std::numeric_limits<std::uint32_t>::max());
    cached.task_count = static_cast<std::uint32_t>(task_count);
  }
  return compiled_kernel_data;
}

std::vector<int> append_arg_index(const std::vector<int> &arg_id, int index) {
  std::vector<int> result = arg_id;
  result.push_back(index);
  return result;
}

DataType get_primitive_dtype(DataType dtype) {
  if (dtype->is<TensorType>()) {
    return dtype->cast<TensorType>()->get_element_type();
  }
  return dtype;
}

PrimitiveTypeID get_primitive_type_id(DataType dtype) {
  return get_primitive_dtype(dtype)->as<PrimitiveType>()->type;
}

void validate_graph_runtime_storage_argument(
    const storage::RuntimeStorageArgument &argument,
    const std::string &name,
    PrimitiveTypeID expected_dtype_id,
    std::size_t expected_index_rank,
    const std::vector<int> &expected_element_shape) {
  const auto &qualification = argument.qualification();
  TI_ERROR_IF(!qualification.capabilities.bindable ||
                  !qualification.capabilities.replayable ||
                  !qualification.capabilities.zero_copy_qualified,
              "Graph runtime storage argument {} is not replayable: {}", name,
              storage::to_string(qualification.reason));
  const auto &descriptor = argument.descriptor();
  const char *actual_storage =
      descriptor.source_kind() == storage::StorageSourceKind::kNdarray
          ? "an ndarray"
          : "dense storage";
  TI_ERROR_IF(descriptor.index_rank() != expected_index_rank,
              "Dispatch node is compiled for argument {} with field_dim={} "
              "but got {} with field_dim={}",
              name, expected_index_rank, actual_storage,
              descriptor.index_rank());
  const auto element_shape = descriptor.element_shape();
  TI_ERROR_IF(element_shape.size() != expected_element_shape.size(),
              "Mismatched element rank for Graph argument {}", name);
  for (std::size_t i = 0; i < element_shape.size(); ++i) {
    TI_ERROR_IF(element_shape[i] != expected_element_shape[i],
                "Mismatched element shape for Graph argument {}", name);
  }
  const PrimitiveTypeID actual_dtype_id =
      get_primitive_type_id(descriptor.scalar_type());
  TI_ERROR_IF(actual_dtype_id != expected_dtype_id,
              "Dispatch node is compiled for argument {} with dtype={} but "
              "got {} with dtype={}",
              name, PrimitiveType::get(expected_dtype_id).to_string(),
              actual_storage,
              PrimitiveType::get(actual_dtype_id).to_string());
}
template <typename T>
void write_arg_buffer(char *arg_buffer, int offset, uint64 value) {
  T typed_value = taichi_union_cast_with_different_sizes<T>(value);
  std::memcpy(arg_buffer + offset, &typed_value, sizeof(T));
}

void write_scalar_arg(char *arg_buffer,
                      const CompiledGraphRuntimeArgPlan &arg_plan,
                      uint64 value) {
  switch (arg_plan.type_size) {
    case 1:
      write_arg_buffer<int8>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    case 2:
      write_arg_buffer<int16>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    case 4:
      write_arg_buffer<int32>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    case 8:
      write_arg_buffer<int64>(arg_buffer, arg_plan.arg_buffer_offset, value);
      break;
    default:
      TI_ERROR("Unsupported type size {}", arg_plan.type_size);
  }
}

CompiledGraphDispatchRuntimePlan build_cpu_runtime_arg_plan(
    const CompiledDispatch &dispatch) {
  CompiledGraphDispatchRuntimePlan plan;
  plan.cpu_fast_path = true;
  plan.args.reserve(dispatch.symbolic_args.size());
  auto *args_type = dispatch.ti_kernel->args_type;
  for (int i = 0; i < dispatch.symbolic_args.size(); ++i) {
    const auto &symbolic_arg = dispatch.symbolic_args[i];
    CompiledGraphRuntimeArgPlan arg_plan;
    arg_plan.tag = symbolic_arg.tag;
    arg_plan.name = symbolic_arg.name;
    arg_plan.arg_id = {i};
    arg_plan.arg_buffer_offset =
        args_type->get_element_offset(arg_plan.arg_id);
    arg_plan.dtype_id = get_primitive_type_id(symbolic_arg.dtype());
    arg_plan.field_dim = symbolic_arg.field_dim;
    arg_plan.element_shape = symbolic_arg.element_shape;

    if (symbolic_arg.tag == ArgKind::kScalar) {
      arg_plan.type_size = data_type_size(symbolic_arg.dtype());
      if (arg_plan.type_size != 1 && arg_plan.type_size != 2 &&
          arg_plan.type_size != 4 && arg_plan.type_size != 8) {
        plan.cpu_fast_path = false;
        plan.args.clear();
        return plan;
      }
    } else if (symbolic_arg.tag == ArgKind::kNdarray) {
      arg_plan.ndarray_data_ptr_key = append_arg_index(
          arg_plan.arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      arg_plan.ndarray_grad_ptr_key = append_arg_index(
          arg_plan.arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
      arg_plan.ndarray_shape_offsets.reserve(symbolic_arg.field_dim);
      for (int j = 0; j < symbolic_arg.field_dim; ++j) {
        arg_plan.ndarray_shape_offsets.push_back(
            args_type->get_element_offset({i, 0, j}));
      }
    } else {
      plan.cpu_fast_path = false;
      plan.args.clear();
      return plan;
    }
    plan.args.push_back(std::move(arg_plan));
  }
  return plan;
}

void init_runtime_context_from_plan(
    const CompiledGraphDispatchRuntimePlan &plan,
    const std::unordered_map<std::string, IValue> &args,
    LaunchContextBuilder &ctx) {
  TI_COMPILE_PROFILER("compiled_graph_init_runtime_context");
  char *arg_buffer = ctx.get_context().arg_buffer;
  for (const auto &arg_plan : plan.args) {
    auto found = args.find(arg_plan.name);
    TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                arg_plan.name);
    const IValue &ival = found->second;
    if (arg_plan.tag == ArgKind::kScalar) {
      TI_ASSERT(ival.tag == ArgKind::kScalar);
      write_scalar_arg(arg_buffer, arg_plan, ival.val);
      continue;
    }

    TI_ASSERT(arg_plan.tag == ArgKind::kNdarray);
    TI_ASSERT(ival.tag == ArgKind::kNdarray);
    if (ival.runtime_storage != nullptr) {
      validate_graph_runtime_storage_argument(
          *ival.runtime_storage, arg_plan.name, arg_plan.dtype_id,
          arg_plan.field_dim, arg_plan.element_shape);
      ctx.set_arg_runtime_storage(arg_plan.arg_id, *ival.runtime_storage);
      continue;
    }
    Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
    TI_ERROR_IF(arr == nullptr, "Graph received a null Ndarray runtime arg");
    TI_ERROR_IF(arr->get_element_shape() != arg_plan.element_shape,
                "Mismatched shape information for argument {}",
                arg_plan.name);
    TI_ERROR_IF(arr->shape.size() != arg_plan.field_dim,
                "Dispatch node is compiled for argument {} with field_dim={} "
                "but got an ndarray with field_dim={}",
                arg_plan.name, arg_plan.field_dim, arr->shape.size());

    DataType arr_primitive_dtype = get_primitive_dtype(arr->dtype);
    PrimitiveTypeID arr_dtype_id =
        arr_primitive_dtype->as<PrimitiveType>()->type;
    TI_ERROR_IF(arr_dtype_id != arg_plan.dtype_id,
                "Dispatch node is compiled for argument {} with dtype={} but "
                "got an ndarray with dtype={}",
                arg_plan.name, PrimitiveType::get(arg_plan.dtype_id).to_string(),
                arr_primitive_dtype.to_string());


    intptr_t ptr = arr->get_device_allocation_ptr_as_int();
    ctx.array_ptrs[arg_plan.ndarray_data_ptr_key] = (void *)ptr;
    if (ptr != 0) {
      ctx.array_ptrs[arg_plan.ndarray_grad_ptr_key] = nullptr;
    }
    ctx.device_allocation_type[arg_plan.arg_id] =
        LaunchContextBuilder::DevAllocType::kNdarray;
    if (Program *owner = arr->owning_program()) {
      LaunchContextBuilder::NdarrayResourceRef ref;
      ref.arg_offset = arg_plan.arg_buffer_offset;
      ref.owner = owner;
      ref.data = arr;
      ref.data_handle = arr->runtime_resource_handle();
      TI_ERROR_IF(!ref.data_handle,
                  "Graph received an unregistered Ndarray runtime resource");
      ctx.ndarray_ptrs.push_back(std::move(ref));
    }
    size_t total_size = 1;
    for (int j = 0; j < arr->shape.size(); ++j) {
      int32 shape = (int32)arr->shape[j];
      std::memcpy(arg_buffer + arg_plan.ndarray_shape_offsets[j], &shape,
                  sizeof(shape));
      total_size *= arr->shape[j];
    }
    ctx.array_runtime_sizes[arg_plan.arg_id] = total_size;
  }
}

#if defined(TI_WITH_LLVM)
bool try_launch_cached_llvm_kernel(Program *prog,
                                   const CompiledKernelData &compiled,
                                   CompiledGraphJITCachedKernel &cached,
                                   LaunchContextBuilder &launch_ctx) {
  auto *launcher =
      dynamic_cast<LLVM::KernelLauncher *>(&prog->get_kernel_launcher());
  auto *llvm_compiled =
      dynamic_cast<const LLVM::CompiledKernelData *>(&compiled);
  if (launcher == nullptr || llvm_compiled == nullptr) {
    return false;
  }
  if (cached.llvm_launch_id < 0) {
    auto handle = launcher->register_llvm_kernel(*llvm_compiled);
    cached.llvm_launch_id = handle.get_launch_id();
  }
  KernelLaunchHandle handle;
  handle.set_launch_id(cached.llvm_launch_id);
  TI_ASSERT(launch_ctx.argpack_ptrs.empty());
  if (!launch_ctx.ndarray_ptrs.empty()) {
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
  }
  if (!launch_ctx.dense_storage_ptrs.empty()) {
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
  }
  if (!launch_ctx.texture_ptrs.empty()) {
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
  }
  // The surrounding Graph transaction already owns both the SNodeTree read
  // guard and, when needed, the runtime-resource submission guard.
  {
    TI_PROFILER("launch_llvm_kernel");
    launcher->launch_llvm_kernel(handle, launch_ctx);
  }
  prog->mark_runtime_submission();
  prog->check_runtime_error_after_kernel_launch(compiled);
  return true;
}
#endif

}  // namespace

CompiledGraph::CompiledGraph(
    std::vector<CompiledDispatch> compiled_dispatches)
    : dispatches(std::move(compiled_dispatches)),
      snode_tree_dependencies(
          collect_snode_tree_dependencies(dispatches)) {
}

CompiledGraph::CompiledGraph(
    std::vector<CompiledDispatch> compiled_dispatches,
    std::unordered_map<std::string, aot::Arg> graph_args)
    : dispatches(std::move(compiled_dispatches)),
      args(std::move(graph_args)),
      snode_tree_dependencies(
          collect_snode_tree_dependencies(dispatches)) {
}

#if defined(TI_WITH_CUDA)
namespace {

struct CudaGraphArgSignatureEntry {
  std::string name;
  ArgKind tag{ArgKind::kUnknown};
  Device *device{nullptr};
  DeviceAllocationId alloc_id{0};
  uint64_t byte_offset{0};
  uint64_t byte_size{0};
  uint64_t runtime_signature{0};
  PrimitiveTypeID dtype_id{PrimitiveTypeID::unknown};
  ExternalArrayLayout layout{ExternalArrayLayout::kNull};
  std::vector<int> shape;
  std::vector<int> element_shape;
  uint64 value{0};
  std::vector<uint8_t> value_bytes;

  bool operator==(const CudaGraphArgSignatureEntry &other) const {
    return structurally_equals(other) && alloc_id == other.alloc_id &&
           runtime_signature == other.runtime_signature &&
           value == other.value && value_bytes == other.value_bytes;
  }

  bool structurally_equals(const CudaGraphArgSignatureEntry &other) const {
    return name == other.name && tag == other.tag && device == other.device &&
           byte_size == other.byte_size && dtype_id == other.dtype_id &&
           layout == other.layout && shape == other.shape &&
           element_shape == other.element_shape;
  }
};

struct CudaGraphSignatureCandidate {
  std::vector<CudaGraphArgSignatureEntry> entries;
  std::vector<DeviceAllocation> allocations;
};

bool cuda_graph_signatures_are_structurally_compatible(
    const std::vector<CudaGraphArgSignatureEntry> &lhs,
    const std::vector<CudaGraphArgSignatureEntry> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!lhs[i].structurally_equals(rhs[i])) {
      return false;
    }
  }
  return true;
}

class CudaGraphCaptureStream {
 public:
  CudaGraphCaptureStream() {
    CUDADriver::get_instance().stream_create(&stream_, CU_STREAM_NON_BLOCKING);
  }

  ~CudaGraphCaptureStream() {
    if (stream_ != nullptr) {
      CUDAContext::get_instance().make_current();
      CUDADriver::get_instance().stream_destroy(stream_);
    }
  }

  void *get() const {
    return stream_;
  }

 private:
  void *stream_{nullptr};
};

class CudaGraphHandle {
 public:
  CudaGraphHandle() = default;
  ~CudaGraphHandle() {
    reset();
  }
  CudaGraphHandle(const CudaGraphHandle &) = delete;
  CudaGraphHandle &operator=(const CudaGraphHandle &) = delete;

  CUgraph *put() {
    reset();
    return &graph_;
  }

  CUgraph get() const {
    return graph_;
  }

  explicit operator bool() const {
    return graph_ != nullptr;
  }

  void reset() {
    if (graph_ != nullptr) {
      CUDAContext::get_instance().make_current();
      CUDADriver::get_instance().graph_destroy(graph_);
      graph_ = nullptr;
    }
  }

 private:
  CUgraph graph_{nullptr};
};

class CudaGraphExecHandle {
 public:
  CudaGraphExecHandle() = default;
  ~CudaGraphExecHandle() {
    reset();
  }
  CudaGraphExecHandle(const CudaGraphExecHandle &) = delete;
  CudaGraphExecHandle &operator=(const CudaGraphExecHandle &) = delete;

  CUgraphExec *put() {
    reset();
    return &graph_exec_;
  }

  CUgraphExec get() const {
    return graph_exec_;
  }

  explicit operator bool() const {
    return graph_exec_ != nullptr;
  }

  void reset() {
    if (graph_exec_ != nullptr) {
      auto &context = CUDAContext::get_instance();
      context.make_current();
      // Replay stays on the default stream to preserve the ordering contract
      // visible to ordinary CUDA launches.
      auto &driver = CUDADriver::get_instance();
      driver.stream_synchronize(nullptr);
      driver.graph_exec_destroy(graph_exec_);
      graph_exec_ = nullptr;
    }
  }

 private:
  CUgraphExec graph_exec_{nullptr};
};

class CudaStreamCaptureGuard {
 public:
  explicit CudaStreamCaptureGuard(void *stream) : stream_(stream) {
  }
  ~CudaStreamCaptureGuard() {
    abort();
  }
  CudaStreamCaptureGuard(const CudaStreamCaptureGuard &) = delete;
  CudaStreamCaptureGuard &operator=(const CudaStreamCaptureGuard &) = delete;

  uint32_t end(CUgraph *graph) {
    active_ = false;
    return CUDADriver::get_instance().stream_end_capture.call(stream_, graph);
  }

  void abort() noexcept {
    if (!active_) {
      return;
    }
    active_ = false;
    CUgraph graph = nullptr;
    auto &driver = CUDADriver::get_instance();
    const uint32_t result =
        driver.stream_end_capture.call(stream_, &graph);
    if (result == CUDA_SUCCESS && graph != nullptr) {
      driver.graph_destroy.call(graph);
    }
  }

 private:
  void *stream_{nullptr};
  bool active_{true};
};

class CudaGraphCapturePacket {
 public:
  explicit CudaGraphCapturePacket(void *stream) : stream_(stream) {
  }
  ~CudaGraphCapturePacket() {
    release_and_wait();
  }
  CudaGraphCapturePacket(const CudaGraphCapturePacket &) = delete;
  CudaGraphCapturePacket &operator=(const CudaGraphCapturePacket &) = delete;
  CudaGraphCapturePacket(CudaGraphCapturePacket &&other) noexcept
      : launcher(std::exchange(other.launcher, nullptr)),
        packet(std::move(other.packet)),
        stream_(std::exchange(other.stream_, nullptr)) {
    other.packet.device_arg_buffer = nullptr;
  }
  CudaGraphCapturePacket &operator=(
      CudaGraphCapturePacket &&other) noexcept {
    if (this != &other) {
      release_and_wait();
      launcher = std::exchange(other.launcher, nullptr);
      packet = std::move(other.packet);
      stream_ = std::exchange(other.stream_, nullptr);
      other.packet.device_arg_buffer = nullptr;
    }
    return *this;
  }

  bool retire_argument_buffer() {
    if (packet.device_arg_buffer == nullptr) {
      return false;
    }
    auto &context = CUDAContext::get_instance();
    context.make_current();
    auto &driver = CUDADriver::get_instance();
    bool stream_ordered_free = false;
    if (context.supports_mem_pool()) {
      driver.mem_free_async_impl(packet.device_arg_buffer, stream_);
      stream_ordered_free = true;
    } else {
      driver.mem_free(packet.device_arg_buffer);
    }
    packet.device_arg_buffer = nullptr;
    return stream_ordered_free;
  }

  cuda::KernelLauncher *launcher{nullptr};
  cuda::KernelLauncher::GraphLaunchPacket packet;

 private:
  void release_and_wait() {
    if (retire_argument_buffer()) {
      CUDADriver::get_instance().stream_synchronize(stream_);
    }
  }

  void *stream_{nullptr};
};

class CudaEventHandle {
 public:
  CudaEventHandle() {
    CUDADriver::get_instance().event_create(&event_,
                                            CU_EVENT_DISABLE_TIMING);
  }
  ~CudaEventHandle() {
    reset();
  }
  CudaEventHandle(const CudaEventHandle &) = delete;
  CudaEventHandle &operator=(const CudaEventHandle &) = delete;
  CudaEventHandle(CudaEventHandle &&other) noexcept
      : event_(std::exchange(other.event_, nullptr)),
        recorded_(std::exchange(other.recorded_, false)) {
  }
  CudaEventHandle &operator=(CudaEventHandle &&other) noexcept {
    if (this != &other) {
      reset();
      event_ = std::exchange(other.event_, nullptr);
      recorded_ = std::exchange(other.recorded_, false);
    }
    return *this;
  }

  void record(void *stream) {
    CUDADriver::get_instance().event_record(event_, stream);
    recorded_ = true;
  }

  bool ready() const {
    if (!recorded_) {
      return true;
    }
    auto &driver = CUDADriver::get_instance();
    const uint32_t result = driver.event_query.call(event_);
    if (result == CUDA_SUCCESS) {
      return true;
    }
    if (result == CUDA_ERROR_NOT_READY) {
      return false;
    }
    TI_ERROR("{}", driver.event_query.get_error_message(result));
  }

  void wait() {
    if (recorded_) {
      CUDADriver::get_instance().event_synchronize(event_);
      recorded_ = false;
    }
  }

 private:
  void reset() {
    if (event_ != nullptr) {
      wait();
      CUDADriver::get_instance().event_destroy(event_);
      event_ = nullptr;
    }
  }

  void *event_{nullptr};
  bool recorded_{false};
};

struct CudaDeferredReplayResources {
  CudaDeferredReplayResources(
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases,
      std::vector<std::vector<uint8_t>> host_buffers)
      : allocation_leases(std::move(leases)),
        host_arg_buffers(std::move(host_buffers)) {
    ready_event.record(nullptr);
  }
  CudaDeferredReplayResources(
      CudaEventHandle event,
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases,
      std::vector<std::vector<uint8_t>> host_buffers)
      : ready_event(std::move(event)),
        allocation_leases(std::move(leases)),
        host_arg_buffers(std::move(host_buffers)) {
    ready_event.record(nullptr);
  }
  ~CudaDeferredReplayResources() {
    // Host patch buffers and old allocation leases remain valid until every
    // preceding default-stream replay and the patch that replaced them has
    // completed.
    ready_event.wait();
  }
  CudaDeferredReplayResources(const CudaDeferredReplayResources &) = delete;
  CudaDeferredReplayResources &operator=(
      const CudaDeferredReplayResources &) = delete;
  CudaDeferredReplayResources(CudaDeferredReplayResources &&) noexcept =
      default;
  CudaDeferredReplayResources &operator=(
      CudaDeferredReplayResources &&) noexcept = default;

  CudaEventHandle ready_event;
  std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
      allocation_leases;
  std::vector<std::vector<uint8_t>> host_arg_buffers;
};

}  // namespace

struct CompiledGraphCudaState {
  std::vector<CudaGraphArgSignatureEntry> signature;
  std::vector<DeviceAllocation> allocations;
  std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
      allocation_leases;
  std::unique_ptr<CudaGraphCaptureStream> capture_stream;
  std::vector<CudaGraphCapturePacket> packets;
  CudaGraphExecHandle graph_exec;
  cuda::CudaGraphConditionalControl *conditional_control{nullptr};
  std::uint64_t conditional_handle{0};
  bool conditional_mode{false};
  std::vector<CudaDeferredReplayResources> deferred_resources;
  std::vector<CudaEventHandle> reusable_events;
  CompiledGraphCaptureRetryState retry;
  CompiledGraphStats stats;
  bool diagnostics_enabled{false};
  bool has_captured_once{false};

  static constexpr std::size_t kMaxDeferredReplayBatches = 2;

  ~CompiledGraphCudaState() {
    retire();
  }

  void *ensure_capture_stream() {
    if (capture_stream == nullptr) {
      capture_stream = std::make_unique<CudaGraphCaptureStream>();
    }
    return capture_stream->get();
  }

  void recycle_front_deferred_resources() {
    deferred_resources.front().ready_event.wait();
    reusable_events.push_back(
        std::move(deferred_resources.front().ready_event));
    deferred_resources.erase(deferred_resources.begin());
  }

  void collect_ready_deferred_resources() {
    while (!deferred_resources.empty() &&
           deferred_resources.front().ready_event.ready()) {
      recycle_front_deferred_resources();
    }
  }

  void defer_replay_resources(
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases,
      std::vector<std::vector<uint8_t>> host_buffers) {
    collect_ready_deferred_resources();
    if (deferred_resources.size() >= kMaxDeferredReplayBatches) {
      recycle_front_deferred_resources();
    }
    if (reusable_events.empty()) {
      deferred_resources.emplace_back(std::move(leases),
                                      std::move(host_buffers));
    } else {
      CudaEventHandle event = std::move(reusable_events.back());
      reusable_events.pop_back();
      deferred_resources.emplace_back(std::move(event), std::move(leases),
                                      std::move(host_buffers));
    }
  }

  void retire() {
    auto &context = CUDAContext::get_instance();
    context.make_current();
    graph_exec.reset();
    if (conditional_control != nullptr) {
      CUDADriver::get_instance().mem_free(conditional_control);
      conditional_control = nullptr;
    }
    conditional_handle = 0;
    conditional_mode = false;
    void *stream = capture_stream == nullptr ? nullptr : capture_stream->get();
    bool stream_ordered_free = false;
    for (auto &packet : packets) {
      stream_ordered_free =
          packet.retire_argument_buffer() || stream_ordered_free;
    }
    if (stream_ordered_free) {
      CUDADriver::get_instance().stream_synchronize(stream);
    }
    packets.clear();
    // graph_exec.reset() synchronized the default stream, so all deferred
    // events are ready. Recycle their handles for a later recapture/patch.
    while (!deferred_resources.empty()) {
      recycle_front_deferred_resources();
    }
    signature.clear();
    allocations.clear();
    // A lease may be the last owner preventing an Ndarray allocation retired
    // by Python GC from being released. Drop it only after all replay and
    // capture-owned argument buffers that contain the address are retired.
    allocation_leases.clear();
  }

  uint64_t known_persistent_argument_bytes() const {
    uint64_t bytes = conditional_control == nullptr
                         ? 0
                         : sizeof(cuda::CudaGraphConditionalControl);
    for (const auto &packet : packets) {
      bytes += packet.packet.arg_buffer_size;
    }
    for (const auto &batch : deferred_resources) {
      for (const auto &buffer : batch.host_arg_buffers) {
        bytes += buffer.size();
      }
    }
    return bytes;
  }
};

void CompiledGraphCudaStateDeleter::operator()(
    CompiledGraphCudaState *state) const noexcept {
  delete state;
}

namespace {

CompiledGraphCudaState *get_cuda_graph_state(CompiledGraphJITCache &cache) {
  if (!cache.cuda_graph_state) {
    cache.cuda_graph_state.reset(new CompiledGraphCudaState());
  }
  if (cache.graph_diagnostics_enabled) {
    cache.cuda_graph_state->diagnostics_enabled = true;
  }
  return cache.cuda_graph_state.get();
}

std::optional<CudaGraphSignatureCandidate>
make_cuda_graph_signature(const CompiledGraph &graph,
                          Program &program,
                          const std::unordered_map<std::string, IValue> &args) {
  CudaGraphSignatureCandidate signature;
  signature.entries.reserve(args.size());
  signature.allocations.reserve(args.size());
  for (const auto &kv : args) {
    const auto declared_it = graph.args.find(kv.first);
    if (declared_it == graph.args.end() ||
        declared_it->second.tag != kv.second.tag) {
      return std::nullopt;
    }

    CudaGraphArgSignatureEntry entry;
    entry.name = kv.first;
    entry.tag = kv.second.tag;
    entry.dtype_id = declared_it->second.dtype_id;
    entry.element_shape = declared_it->second.element_shape;

    if (kv.second.tag == ArgKind::kNdarray) {
      DeviceAllocation allocation = kDeviceNullAllocation;
      if (kv.second.runtime_storage != nullptr) {
        const auto &argument = *kv.second.runtime_storage;
        const auto &qualification = argument.qualification();
        const auto &descriptor = argument.descriptor();
        const auto owner_kind = descriptor.owner().kind;
        if (!qualification.capabilities.capturable ||
            (owner_kind != storage::StorageOwnerKind::kProgramNdarray &&
             owner_kind != storage::StorageOwnerKind::kSNodePayload) ||
            argument.synchronization_domain_identity() != 0) {
          return std::nullopt;
        }
        const auto binding =
            program.resolve_runtime_storage_argument_under_graph_guard(
                argument);
        if (!binding.valid || binding.allocation == kDeviceNullAllocation) {
          return std::nullopt;
        }
        allocation = binding.allocation;
        entry.device = allocation.device;
        entry.alloc_id = allocation.alloc_id;
        entry.byte_offset = binding.byte_offset;
        entry.byte_size = binding.byte_size;
        entry.runtime_signature = binding.runtime_signature;
        entry.dtype_id =
            descriptor.scalar_type()->as<PrimitiveType>()->type;
        entry.shape.reserve(descriptor.index_rank());
        for (std::size_t axis = 0; axis < descriptor.index_rank(); ++axis) {
          const std::int64_t extent = descriptor.index_extent(axis);
          if (extent < 0 || extent > (std::numeric_limits<int>::max)()) {
            return std::nullopt;
          }
          entry.shape.push_back(static_cast<int>(extent));
        }
        entry.element_shape.reserve(descriptor.element_rank());
        for (std::size_t axis = 0; axis < descriptor.element_rank(); ++axis) {
          const std::int64_t extent = descriptor.element_extent(axis);
          if (extent < 0 || extent > (std::numeric_limits<int>::max)()) {
            return std::nullopt;
          }
          entry.element_shape.push_back(static_cast<int>(extent));
        }
        switch (descriptor.properties().array_layout) {
          case storage::StorageArrayLayout::kScalar:
            entry.layout = ExternalArrayLayout::kNull;
            break;
          case storage::StorageArrayLayout::kAos:
            entry.layout = ExternalArrayLayout::kAOS;
            break;
          case storage::StorageArrayLayout::kSoa:
            entry.layout = ExternalArrayLayout::kSOA;
            break;
          case storage::StorageArrayLayout::kNone:
            return std::nullopt;
        }
      } else {
        auto *arr = reinterpret_cast<Ndarray *>(kv.second.val);
        if (arr == nullptr) {
          return std::nullopt;
        }
        allocation = arr->get_device_allocation();
        entry.device = allocation.device;
        entry.alloc_id = allocation.alloc_id;
        entry.byte_offset = 0;
        entry.byte_size = arr->get_nelement() * arr->get_element_size();
        entry.dtype_id = arr->get_element_data_type()
                             ->as<PrimitiveType>()
                             ->type;
        entry.layout = arr->layout;
        entry.shape = arr->shape;
        entry.element_shape = arr->get_element_shape();
      }
      auto *device = dynamic_cast<cuda::CudaDevice *>(allocation.device);
      if (device == nullptr) {
        return std::nullopt;
      }
      const bool already_listed =
          std::find(signature.allocations.begin(),
                    signature.allocations.end(),
                    allocation) != signature.allocations.end();
      if (!already_listed) {
        signature.allocations.push_back(allocation);
      }
    } else if (kv.second.tag == ArgKind::kScalar) {
      entry.byte_size = data_type_size(declared_it->second.dtype());
      entry.value = kv.second.val;
    } else if (kv.second.tag == ArgKind::kMatrix) {
      auto *matrix = reinterpret_cast<Matrix *>(kv.second.val);
      entry.byte_size = matrix->length() * data_type_size(matrix->dtype());
      if (entry.byte_size > 128) {
        return std::nullopt;
      }
      entry.value_bytes.resize(entry.byte_size);
      std::memcpy(entry.value_bytes.data(),
                  reinterpret_cast<const void *>(matrix->data()),
                  entry.byte_size);
    } else {
      // Texture handles require a backend-specific lifetime owner and are not
      // eligible for CUDA argument-buffer patching yet.
      return std::nullopt;
    }
    signature.entries.push_back(std::move(entry));
  }
  std::sort(signature.entries.begin(), signature.entries.end(),
            [](const CudaGraphArgSignatureEntry &lhs,
               const CudaGraphArgSignatureEntry &rhs) {
              return lhs.name < rhs.name;
            });
  std::sort(signature.allocations.begin(), signature.allocations.end(),
            [](const DeviceAllocation &lhs, const DeviceAllocation &rhs) {
              const auto lhs_device =
                  reinterpret_cast<std::uintptr_t>(lhs.device);
              const auto rhs_device =
                  reinterpret_cast<std::uintptr_t>(rhs.device);
              return lhs_device == rhs_device ? lhs.alloc_id < rhs.alloc_id
                                              : lhs_device < rhs_device;
            });
  return signature;
}
std::optional<
    std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>>
acquire_cuda_graph_allocation_leases(
    const std::vector<DeviceAllocation> &allocations) {
  std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>> leases;
  leases.reserve(allocations.size());
  for (const DeviceAllocation &allocation : allocations) {
    auto *device = dynamic_cast<cuda::CudaDevice *>(allocation.device);
    if (device == nullptr) {
      return std::nullopt;
    }
    auto lease = device->acquire_allocation_lease(allocation);
    if (lease == nullptr) {
      return std::nullopt;
    }
    leases.push_back(std::move(lease));
  }
  return leases;
}

bool patch_cuda_graph_arguments(
    const CompiledGraph &graph,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphCudaState &state,
    std::vector<std::vector<uint8_t>> &host_arg_buffers) {
  if (state.packets.size() != graph.dispatches.size()) {
    return false;
  }
  host_arg_buffers.clear();
  host_arg_buffers.reserve(graph.dispatches.size());
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr || launcher != state.packets[i].launcher) {
      return false;
    }

    LaunchContextBuilder launch_ctx(dispatch.ti_kernel);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    host_arg_buffers.emplace_back();
    if (!launcher->update_cuda_graph_launch(
            state.packets[i].packet, launch_ctx, host_arg_buffers.back(),
            /*stream=*/nullptr)) {
      return false;
    }
  }
  return true;
}

bool is_cuda_graph_structural_driver_error(uint32_t error) {
  return error == CUDA_ERROR_NOT_SUPPORTED ||
         error == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
}

void mark_cuda_graph_fallback(
    CompiledGraphCudaState &state,
    CompiledGraphFallbackReason reason,
    bool structural = false) {
  state.stats.last_path = CompiledGraphExecutionPath::ordinary_fallback;
  state.stats.last_fallback_reason = reason;
  if (state.diagnostics_enabled) {
    ++state.stats.ordinary_fallbacks;
    if (structural) {
      ++state.stats.structural_fallbacks;
    }
  }
}

bool handle_cuda_graph_driver_failure(CompiledGraphCudaState &state,
                                      uint32_t error,
                                      const char *stage) {
  if (state.diagnostics_enabled) {
    state.stats.last_driver_error = error;
  }
  if (classify_cuda_driver_error(error) ==
      BackendErrorClassification::kFatal) {
    state.stats.last_path = CompiledGraphExecutionPath::none;
    state.stats.last_fallback_reason =
        CompiledGraphFallbackReason::fatal_driver_failure;
    state.stats.last_driver_error = error;
    throw BackendRuntimeError(
        Arch::cuda, error, fmt::format("cuda_graph_{}", stage),
        fmt::format("CUDA graph {} failed with a context-fatal error: {}",
                    stage, get_cuda_error_message(error)));
  }

  if (is_cuda_graph_structural_driver_error(error)) {
    state.retry.record_structural_failure();
    mark_cuda_graph_fallback(
        state, CompiledGraphFallbackReason::structural_unsupported, true);
  } else {
    state.retry.record_transient_failure();
    if (state.diagnostics_enabled) {
      ++state.stats.transient_failures;
    }
    mark_cuda_graph_fallback(
        state, CompiledGraphFallbackReason::transient_driver_failure);
  }
  state.retire();
  return false;
}

bool try_run_cuda_graph(const CompiledGraph &graph,
                        const CompileConfig &compile_config,
                        const std::unordered_map<std::string, IValue> &args,
                        CompiledGraphJITCache &cache,
                        Program &program,
                        RuntimeStatistics *statistics) {
  auto *state = get_cuda_graph_state(cache);
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  if (compile_config.debug) {
    mark_cuda_graph_fallback(*state,
                             CompiledGraphFallbackReason::debug_mode);
    return false;
  }
  if (graph.dispatches.empty()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::insufficient_dispatches, true);
    return false;
  }
  std::optional<CudaGraphSignatureCandidate> signature;
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    // A runtime Field may have been destroyed after this executable was last
    // used. Retire the cached graph before surfacing the stale-generation
    // error so no later cleanup keeps an executable with a dead root address.
    state->retire();
    throw;
  }
  if (!signature.has_value()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::unsupported_arguments, true);
    return false;
  }
  CUDAContext::get_instance().make_current();
  if (state->retry.structurally_disabled()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    return false;
  }
  state->collect_ready_deferred_resources();
  if (state->graph_exec &&
      state->signature == signature->entries) {
    CUDADriver::get_instance().graph_launch(state->graph_exec.get(), nullptr);
    state->stats.last_path = CompiledGraphExecutionPath::cuda_exact_replay;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
    if (state->diagnostics_enabled) {
      ++state->stats.exact_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }

  if (!state->graph_exec && !state->retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  const bool structurally_compatible =
      state->graph_exec && cuda_graph_signatures_are_structurally_compatible(
                               state->signature, signature->entries);
  if (structurally_compatible) {
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, *state, host_arg_buffers)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      // The default stream orders this event after the previous replay and
      // every argument-buffer upload above. At that point old allocations are
      // no longer referenced and the host staging buffers have been consumed.
      // Record before the new replay so bounded retirement does not wait for
      // one more graph execution than correctness requires.
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers));
      CUDADriver::get_instance().graph_launch(state->graph_exec.get(),
                                              nullptr);
      state->stats.last_path =
          CompiledGraphExecutionPath::cuda_patched_replay;
      state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
      if (state->diagnostics_enabled) {
        ++state->stats.patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }
  // Any partial argument-buffer update is ordered on the default stream.
  // Retiring the old executable synchronizes that stream before recapture.
  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (state->has_captured_once) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }

  auto &driver = CUDADriver::get_instance();
  void *capture_stream = state->ensure_capture_stream();

  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      state->retry.record_structural_failure();
      mark_cuda_graph_fallback(
          *state, CompiledGraphFallbackReason::structural_unsupported, true);
      state->retire();
      return false;
    }

    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    auto handle = launcher->register_llvm_kernel(llvm_ckd);

    LaunchContextBuilder launch_ctx(dispatch.ti_kernel);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    if (!launcher->prepare_cuda_graph_launch(
            handle, launch_ctx, capture_packet.packet, capture_stream)) {
      driver.stream_synchronize(capture_stream);
      state->retry.record_structural_failure();
      mark_cuda_graph_fallback(
          *state, CompiledGraphFallbackReason::structural_unsupported, true);
      state->retire();
      return false;
    }
    state->packets.push_back(std::move(capture_packet));
  }

  state->stats.zero_arg_eligible =
      !state->packets.empty() &&
      std::all_of(state->packets.begin(), state->packets.end(),
                  [](const CudaGraphCapturePacket &packet) {
                    return packet.packet.arg_buffer_size == 0 &&
                           packet.packet.device_arg_buffer == nullptr;
                  });

  driver.stream_synchronize(capture_stream);
  auto capture_lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  auto begin_err = driver.stream_begin_capture.call(
      capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_err != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(*state, begin_err,
                                            "stream begin capture");
  }
  CudaStreamCaptureGuard capture_guard(capture_stream);
  try {
    for (const auto &packet : state->packets) {
      packet.launcher->capture_cuda_graph_launch(packet.packet,
                                                 capture_stream);
    }
  } catch (...) {
    if (state->diagnostics_enabled) {
      ++state->stats.capture_exceptions;
    }
    capture_guard.abort();
    throw;
  }
  CudaGraphHandle captured_graph;
  auto end_err = capture_guard.end(captured_graph.put());
  if (end_err != CUDA_SUCCESS || !captured_graph) {
    return handle_cuda_graph_driver_failure(*state, end_err,
                                            "stream end capture");
  }
  auto instantiate_err = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), captured_graph.get(), 0);
  captured_graph.reset();
  if (instantiate_err != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(*state, instantiate_err,
                                            "instantiate");
  }
  state->retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path = CompiledGraphExecutionPath::cuda_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
    if (state->stats.zero_arg_eligible) {
      ++state->stats.zero_arg_captures;
    }
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  driver.graph_launch(state->graph_exec.get(), nullptr);
  return true;
}

bool try_run_cuda_bounded_graph(
    const CompiledGraph &graph,
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Program &program,
    Ndarray &predicate,
    int max_iterations,
    bool continue_while_nonzero,
    RuntimeStatistics *statistics) {
  auto *state = get_cuda_graph_state(cache);
  if (state->diagnostics_enabled) {
    state->stats.backend = CompiledGraphBackend::cuda;
    ++state->stats.attempts;
    state->stats.last_driver_error = 0;
  }
  auto structural_fallback = [&]() {
    state->retry.record_structural_failure();
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::structural_unsupported, true);
    state->retire();
    return false;
  };
  if (compile_config.debug || graph.dispatches.empty() ||
      max_iterations <= 0 || predicate.get_nelement() != 1 ||
      predicate.get_element_data_type() != PrimitiveType::i32 ||
      predicate.owning_program() != &program) {
    return structural_fallback();
  }

  const DeviceAllocation predicate_allocation =
      predicate.get_device_allocation();
  auto *predicate_device = dynamic_cast<cuda::CudaDevice *>(
      predicate_allocation.device);
  if (predicate_device == nullptr) {
    return structural_fallback();
  }
  auto &driver = CUDADriver::get_instance();
  const bool conditional_symbols =
      driver.stream_begin_capture_to_graph.available() &&
      driver.stream_end_capture.available() &&
      driver.graph_create.available() &&
      driver.graph_conditional_handle_create.available() &&
      driver.graph_add_node.available() &&
      driver.graph_instantiate_with_flags.available() &&
      driver.graph_launch.available() && driver.graph_destroy.available() &&
      driver.graph_exec_destroy.available();
  if (!conditional_symbols ||
      !cuda::driver_graph_conditional_setter_compiled()) {
    return structural_fallback();
  }

  std::optional<CudaGraphSignatureCandidate> signature;
  try {
    signature = make_cuda_graph_signature(graph, program, args);
  } catch (...) {
    state->retire();
    throw;
  }
  if (!signature.has_value()) {
    return structural_fallback();
  }
  CUDAContext::get_instance().make_current();
  state->collect_ready_deferred_resources();

  auto update_control = [&]() {
    TI_ASSERT(state->conditional_control != nullptr);
    cuda::CudaGraphConditionalControl control;
    control.predicate = reinterpret_cast<std::uintptr_t>(
        predicate_device->get_alloc_info(predicate_allocation).ptr);
    control.max_iterations = static_cast<std::uint32_t>(max_iterations);
    control.continue_while_nonzero = continue_while_nonzero ? 1u : 0u;
    driver.memcpy_host_to_device(state->conditional_control, &control,
                                 sizeof(control));
  };
  auto launch = [&](CompiledGraphExecutionPath path) {
    update_control();
    driver.graph_launch(state->graph_exec.get(), nullptr);
    state->stats.last_path = path;
    state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  };

  if (state->conditional_mode && state->graph_exec &&
      state->signature == signature->entries) {
    launch(CompiledGraphExecutionPath::cuda_exact_replay);
    if (state->diagnostics_enabled) {
      ++state->stats.exact_replays;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    return true;
  }
  if (!state->graph_exec && !state->retry.should_attempt()) {
    if (state->diagnostics_enabled) {
      ++state->stats.retry_backoff_fallbacks;
    }
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::retry_backoff);
    return false;
  }

  auto allocation_leases =
      acquire_cuda_graph_allocation_leases(signature->allocations);
  if (!allocation_leases.has_value()) {
    mark_cuda_graph_fallback(
        *state, CompiledGraphFallbackReason::resource_unavailable);
    return false;
  }
  const bool structurally_compatible =
      state->conditional_mode && state->graph_exec &&
      cuda_graph_signatures_are_structurally_compatible(
          state->signature, signature->entries);
  if (structurally_compatible) {
    std::vector<std::vector<uint8_t>> host_arg_buffers;
    if (patch_cuda_graph_arguments(graph, args, *state, host_arg_buffers)) {
      std::vector<std::unique_ptr<cuda::CudaDevice::AllocationLease>>
          old_allocation_leases;
      if (state->allocations != signature->allocations) {
        old_allocation_leases = std::move(state->allocation_leases);
        state->allocation_leases = std::move(*allocation_leases);
        state->allocations = signature->allocations;
      }
      state->signature = std::move(signature->entries);
      state->defer_replay_resources(std::move(old_allocation_leases),
                                    std::move(host_arg_buffers));
      launch(CompiledGraphExecutionPath::cuda_patched_replay);
      if (state->diagnostics_enabled) {
        ++state->stats.patched_replays;
      }
      if (statistics != nullptr) {
        statistics->record_graph_replay();
      }
      return true;
    }
  }

  const bool is_recapture = state->has_captured_once;
  state->retire();
  state->signature = std::move(signature->entries);
  state->allocations = signature->allocations;
  state->allocation_leases = std::move(*allocation_leases);
  if (state->diagnostics_enabled) {
    ++state->stats.capture_attempts;
    if (is_recapture) {
      ++state->stats.recaptures;
    }
  }
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }

  void *capture_stream = state->ensure_capture_stream();
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
        &prog->get_program_impl()->get_kernel_launcher());
    if (launcher == nullptr) {
      return structural_fallback();
    }
    auto *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/true);
    const auto &llvm_ckd =
        dynamic_cast<const LLVM::CompiledKernelData &>(*compiled_kernel_data);
    auto handle = launcher->register_llvm_kernel(llvm_ckd);
    LaunchContextBuilder launch_ctx(dispatch.ti_kernel);
    graph.init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    prog->resolve_ndarray_launch_context_under_guard(launch_ctx);
    prog->resolve_runtime_storage_launch_context_under_guard(launch_ctx);
    prog->resolve_texture_launch_context_under_guard(launch_ctx);
    CudaGraphCapturePacket capture_packet(capture_stream);
    capture_packet.launcher = launcher;
    if (!launcher->prepare_cuda_graph_launch(
            handle, launch_ctx, capture_packet.packet, capture_stream)) {
      driver.stream_synchronize(capture_stream);
      return structural_fallback();
    }
    state->packets.push_back(std::move(capture_packet));
  }
  state->stats.zero_arg_eligible =
      !state->packets.empty() &&
      std::all_of(state->packets.begin(), state->packets.end(),
                  [](const CudaGraphCapturePacket &packet) {
                    return packet.packet.arg_buffer_size == 0 &&
                           packet.packet.device_arg_buffer == nullptr;
                  });

  try {
    cuda::driver_graph_prepare_conditional_setter();
  } catch (...) {
    return structural_fallback();
  }
  driver.malloc(reinterpret_cast<void **>(&state->conditional_control),
                sizeof(cuda::CudaGraphConditionalControl));
  CudaGraphHandle parent_graph;
  const auto create_error = driver.graph_create.call(parent_graph.put(), 0);
  if (create_error != CUDA_SUCCESS || !parent_graph) {
    return handle_cuda_graph_driver_failure(
        *state, create_error, "conditional graph create");
  }
  void *current_context = nullptr;
  const auto context_error =
      driver.context_get_current.call(&current_context);
  if (context_error != CUDA_SUCCESS || current_context == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, context_error, "conditional context query");
  }

  constexpr unsigned int kAssignDefaultValue = 1;
  std::uint64_t conditional_handle = 0;
  const auto handle_error = driver.graph_conditional_handle_create.call(
      &conditional_handle, parent_graph.get(), current_context, 1,
      kAssignDefaultValue);
  if (handle_error != CUDA_SUCCESS || conditional_handle == 0) {
    return handle_cuda_graph_driver_failure(
        *state, handle_error, "conditional handle create");
  }
  TaichiCudaGraphNodeParams node_params{};
  constexpr std::uint32_t kConditionalNodeType = 13;
  constexpr std::uint32_t kWhileConditionalType = 1;
  node_params.type = kConditionalNodeType;
  node_params.parameters.conditional.handle = conditional_handle;
  node_params.parameters.conditional.type = kWhileConditionalType;
  node_params.parameters.conditional.size = 1;
  node_params.parameters.conditional.ph_graph_out = nullptr;
  node_params.parameters.conditional.context = current_context;
  void *conditional_node = nullptr;
  const auto add_error = driver.graph_add_node.call(
      &conditional_node, parent_graph.get(), nullptr, 0, &node_params);
  if (add_error != CUDA_SUCCESS || conditional_node == nullptr ||
      node_params.parameters.conditional.ph_graph_out == nullptr ||
      node_params.parameters.conditional.ph_graph_out[0] == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, add_error, "conditional node create");
  }

  driver.stream_synchronize(capture_stream);
  auto capture_lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  const auto begin_error = driver.stream_begin_capture_to_graph.call(
      capture_stream, node_params.parameters.conditional.ph_graph_out[0],
      nullptr, nullptr, 0, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_error != CUDA_SUCCESS) {
    return handle_cuda_graph_driver_failure(
        *state, begin_error, "conditional body capture begin");
  }
  CudaStreamCaptureGuard capture_guard(capture_stream);
  try {
    for (const auto &packet : state->packets) {
      packet.launcher->capture_cuda_graph_launch(packet.packet,
                                                 capture_stream);
    }
    cuda::driver_graph_set_conditional(
        state->conditional_control, conditional_handle, capture_stream);
  } catch (...) {
    if (state->diagnostics_enabled) {
      ++state->stats.capture_exceptions;
    }
    capture_guard.abort();
    state->retire();
    throw;
  }
  CUgraph captured_child = nullptr;
  const auto end_error = capture_guard.end(&captured_child);
  if (end_error != CUDA_SUCCESS || captured_child == nullptr) {
    return handle_cuda_graph_driver_failure(
        *state, end_error, "conditional body capture end");
  }
  const auto instantiate_error = driver.graph_instantiate_with_flags.call(
      state->graph_exec.put(), parent_graph.get(), 0);
  if (instantiate_error != CUDA_SUCCESS || !state->graph_exec) {
    return handle_cuda_graph_driver_failure(
        *state, instantiate_error, "conditional graph instantiate");
  }

  state->conditional_handle = conditional_handle;
  state->conditional_mode = true;
  state->retry.record_success();
  state->has_captured_once = true;
  state->stats.last_path = CompiledGraphExecutionPath::cuda_capture;
  state->stats.last_fallback_reason = CompiledGraphFallbackReason::none;
  if (state->diagnostics_enabled) {
    ++state->stats.captures;
    if (state->stats.zero_arg_eligible) {
      ++state->stats.zero_arg_captures;
    }
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  launch(CompiledGraphExecutionPath::cuda_capture);
  return true;
}

}  // namespace

#endif

#if !defined(TI_WITH_CUDA)
void CompiledGraphCudaStateDeleter::operator()(
    CompiledGraphCudaState *state) const noexcept {
  TI_ASSERT(state == nullptr);
}
#endif

#if defined(TI_WITH_VULKAN)
struct CompiledGraphVulkanState {
  std::unique_ptr<gfx::GraphReplayRegistration> registration;
  bool diagnostics_enabled{false};
};

void CompiledGraphVulkanStateDeleter::operator()(
    CompiledGraphVulkanState *state) const noexcept {
  delete state;
}

namespace {

CompiledGraphVulkanState *get_vulkan_graph_state(
    CompiledGraphJITCache &cache,
    gfx::GfxRuntime *runtime) {
  if (cache.vulkan_graph_state &&
      !runtime->owns_graph_replay_registration(
          *cache.vulkan_graph_state->registration)) {
    cache.vulkan_graph_state.reset();
  }
  if (!cache.vulkan_graph_state) {
    auto state = std::make_unique<CompiledGraphVulkanState>();
    state->registration =
        runtime->register_graph_replay(cache.graph_replay_token());
    cache.vulkan_graph_state.reset(state.release());
  }
  if (cache.graph_diagnostics_enabled &&
      !cache.vulkan_graph_state->diagnostics_enabled) {
    // A report may be requested before the first Vulkan run, before a runtime
    // replay state exists. Touching the registration here enables detailed
    // counters before this very launch instead of dropping the first sample.
    cache.vulkan_graph_state->registration->debug_stats();
    cache.vulkan_graph_state->diagnostics_enabled = true;
  }
  return cache.vulkan_graph_state.get();
}

bool try_run_vulkan_graph(const CompiledGraph &graph,
                          const CompileConfig &compile_config,
                          const std::unordered_map<std::string, IValue> &args,
                          CompiledGraphJITCache &cache,
                          RuntimeStatistics *statistics) {
  if (compile_config.debug) {
    auto &stats = cache.vulkan_inline_stats;
    stats.backend = CompiledGraphBackend::vulkan;
    stats.last_path = CompiledGraphExecutionPath::ordinary_fallback;
    stats.last_fallback_reason = CompiledGraphFallbackReason::debug_mode;
    if (cache.graph_diagnostics_enabled) {
      ++stats.attempts;
      ++stats.ordinary_fallbacks;
    }
    return false;
  }
  if (graph.dispatches.size() <= 1) {
    auto &stats = cache.vulkan_inline_stats;
    stats.backend = CompiledGraphBackend::vulkan;
    stats.last_path = CompiledGraphExecutionPath::ordinary_fallback;
    stats.last_fallback_reason =
        CompiledGraphFallbackReason::insufficient_dispatches;
    if (cache.graph_diagnostics_enabled) {
      ++stats.attempts;
      ++stats.ordinary_fallbacks;
    }
    return false;
  }
  if (cache.kernels.size() != graph.dispatches.size()) {
    cache.kernels.assign(graph.dispatches.size(), {});
  }

  std::vector<std::unique_ptr<LaunchContextBuilder>> launch_contexts;
  std::vector<gfx::GfxRuntime::GraphDispatch> gfx_dispatches;
  launch_contexts.reserve(graph.dispatches.size());
  gfx_dispatches.reserve(graph.dispatches.size());

  gfx::KernelLauncher *gfx_launcher = nullptr;
  for (std::size_t i = 0; i < graph.dispatches.size(); ++i) {
    const auto &dispatch = graph.dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    auto *prog = dispatch.ti_kernel->program;
    auto *launcher = dynamic_cast<gfx::KernelLauncher *>(
        &prog->get_kernel_launcher());
    if (launcher == nullptr) {
      return false;
    }
    if (gfx_launcher == nullptr) {
      gfx_launcher = launcher;
    } else if (gfx_launcher != launcher) {
      return false;
    }

    const CompiledKernelData *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cache.kernels[i],
                                     /*cache_compiled_kernel_data=*/false);
    auto handle = launcher->get_or_register_kernel(*compiled_kernel_data);
    launch_contexts.push_back(
        std::make_unique<LaunchContextBuilder>(dispatch.ti_kernel));
    graph.init_runtime_context(dispatch.symbolic_args, args,
                               *launch_contexts.back());
    prog->resolve_ndarray_launch_context_under_guard(*launch_contexts.back());
    prog->resolve_runtime_storage_launch_context_under_guard(
        *launch_contexts.back());
    prog->resolve_texture_launch_context_under_guard(*launch_contexts.back());
    gfx_dispatches.push_back({handle, launch_contexts.back().get()});
  }

  if (gfx_launcher == nullptr) {
    return false;
  }
  auto *runtime = gfx_launcher->runtime();
  auto *state = get_vulkan_graph_state(cache, runtime);
  return runtime->try_launch_graph(
      gfx_dispatches, state->registration->replay_key(), statistics);
}

}  // namespace
#endif

#if !defined(TI_WITH_VULKAN)
void CompiledGraphVulkanStateDeleter::operator()(
    CompiledGraphVulkanState *state) const noexcept {
  TI_ASSERT(state == nullptr);
}
#endif

CompiledGraphDebugSnapshot CompiledGraphJITCache::debug_graph_stats() {
  std::lock_guard<std::mutex> lock(run_mutex);
  const bool diagnostics_previously_enabled = graph_diagnostics_enabled;
  graph_diagnostics_enabled = true;
  auto finalize = [&](CompiledGraphStats result) {
    CompiledGraphDebugSnapshot snapshot;
    snapshot.stats = result;
    snapshot.diagnostics_previously_enabled =
        diagnostics_previously_enabled;
    snapshot.diagnostics_counters_complete =
        graph_diagnostics_counters_complete;
    for (const auto &kernel : kernels) {
      if (kernel.task_count ==
          std::numeric_limits<std::uint32_t>::max()) {
        continue;
      }
      ++snapshot.known_compiled_dispatches;
      snapshot.known_compiled_tasks += kernel.task_count;
    }
    return snapshot;
  };
#if defined(TI_WITH_CUDA)
  if (cuda_graph_state) {
    cuda_graph_state->diagnostics_enabled = true;
    cuda_graph_state->stats.backend = CompiledGraphBackend::cuda;
    CompiledGraphStats result = cuda_graph_state->stats;
    if (!diagnostics_previously_enabled &&
        (result.last_path != CompiledGraphExecutionPath::none ||
         result.last_fallback_reason != CompiledGraphFallbackReason::none)) {
      graph_diagnostics_counters_complete = false;
    }
    result.known_persistent_argument_bytes =
        cuda_graph_state->known_persistent_argument_bytes();
    result.retry_backoff_remaining =
        cuda_graph_state->retry.retry_backoff_remaining();
    result.consecutive_transient_failures =
        cuda_graph_state->retry.consecutive_transient_failures();
    return finalize(result);
  }
#endif
#if defined(TI_WITH_VULKAN)
  if (vulkan_graph_state && vulkan_graph_state->registration) {
    const auto source = vulkan_graph_state->registration->debug_stats();
    vulkan_graph_state->diagnostics_enabled = true;
    CompiledGraphStats result;
    if (!diagnostics_previously_enabled &&
        source.last_path != gfx::GraphReplayLastPath::none) {
      graph_diagnostics_counters_complete = false;
    }
    result.backend = CompiledGraphBackend::vulkan;
    result.attempts = source.attempts;
    result.ordinary_fallbacks = source.fallbacks;
    result.records = source.recorded;
    result.replays = source.replayed;
    result.structural_fallbacks = source.structural_fallbacks;
    result.replay_slot_saturation_fallbacks =
        source.slot_saturation_fallbacks;
    result.known_persistent_argument_bytes =
        source.known_persistent_argument_bytes;
    result.effect_reads = source.effect_reads;
    result.effect_writes = source.effect_writes;
    result.dependency_barriers = source.dependency_barriers;
    result.exit_barriers = source.exit_barriers;
    result.barrier_deferrals = source.barrier_deferrals;
    result.rar_elisions = source.rar_elisions;
    switch (source.last_path) {
      case gfx::GraphReplayLastPath::fallback:
        result.last_path = CompiledGraphExecutionPath::ordinary_fallback;
        break;
      case gfx::GraphReplayLastPath::record:
        result.last_path = CompiledGraphExecutionPath::vulkan_record;
        break;
      case gfx::GraphReplayLastPath::replay:
        result.last_path = CompiledGraphExecutionPath::vulkan_replay;
        break;
      case gfx::GraphReplayLastPath::none:
        break;
    }
    switch (source.last_fallback_reason) {
      case gfx::GraphReplayFallbackReason::runtime_mode:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::runtime_mode;
        break;
      case gfx::GraphReplayFallbackReason::insufficient_tasks:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::insufficient_dispatches;
        break;
      case gfx::GraphReplayFallbackReason::structural_unsupported:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::structural_unsupported;
        break;
      case gfx::GraphReplayFallbackReason::slot_saturated:
        result.last_fallback_reason =
            CompiledGraphFallbackReason::replay_slot_saturated;
        break;
      case gfx::GraphReplayFallbackReason::none:
        break;
    }
    return finalize(result);
  }
#endif
  if (vulkan_inline_stats.attempts > 0) {
    if (!diagnostics_previously_enabled) {
      graph_diagnostics_counters_complete = false;
    }
    return finalize(vulkan_inline_stats);
  }
  if (vulkan_inline_stats.backend != CompiledGraphBackend::none) {
    if (!diagnostics_previously_enabled) {
      graph_diagnostics_counters_complete = false;
    }
    return finalize(vulkan_inline_stats);
  }
  return finalize({});
}

void CompiledGraphJITCache::clear_runtime_state() {
#if defined(TI_WITH_CUDA)
  // Match jit_run_cached() lock ordering so reset/destruction cannot retire a
  // graph executable while another CUDA submission transaction is active.
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
#endif
  std::lock_guard<std::mutex> lock(run_mutex);
  cuda_graph_state.reset();
  vulkan_graph_state.reset();
  vulkan_inline_stats = {};
  graph_diagnostics_counters_complete = true;
  kernels.clear();
  runtime_arg_plans.clear();
  validated_snode_tree_program = nullptr;
  validated_snode_tree_epoch = 0;
}

CompiledGraphJITCache::~CompiledGraphJITCache() {
  clear_runtime_state();
}

void CompiledGraph::run(
    const std::unordered_map<std::string, IValue> &args) const {
  for (const auto &dispatch : dispatches) {
    TI_ASSERT(dispatch.compiled_kernel);
    LaunchContextBuilder launch_ctx(dispatch.compiled_kernel);
    init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    TI_ERROR_IF(
        !launch_ctx.dense_storage_ptrs.empty(),
        "AOT Graph does not support runtime dense storage arguments; use an "
        "owning Ndarray or a JIT Graph");
    Program *program = nullptr;
    std::vector<const Ndarray *> ndarray_views;
    std::vector<const Texture *> texture_views;
    for (const auto &ref : launch_ctx.ndarray_ptrs) {
      Program *owner = ref.owner;
      TI_ASSERT(owner != nullptr);
      TI_ERROR_IF(program != nullptr && owner != program,
                  "AOT Graph Ndarray arguments span multiple Programs");
      program = owner;
      for (const Ndarray *view : {ref.data, ref.grad}) {
        if (view == nullptr) {
          continue;
        }
        if (std::find(ndarray_views.begin(), ndarray_views.end(), view) ==
            ndarray_views.end()) {
          ndarray_views.push_back(view);
        }
      }
    }
    for (const auto &ref : launch_ctx.texture_ptrs) {
      Program *owner = ref.owner;
      TI_ASSERT(owner != nullptr);
      TI_ERROR_IF(program != nullptr && owner != program,
                  "AOT Graph runtime resources span multiple Programs");
      program = owner;
      if (std::find(texture_views.begin(), texture_views.end(), ref.texture) ==
          texture_views.end()) {
        texture_views.push_back(ref.texture);
      }
    }
    if (program != nullptr) {
      program->ensure_runtime_submission_allowed("AOT Graph launch");
    }
    std::optional<Program::RuntimeResourceSubmissionGuard> resource_guard;
    if (program != nullptr) {
      resource_guard.emplace(
          program->acquire_runtime_resource_submission_guard());
      program->retain_ndarrays_for_external_submission(ndarray_views);
      program->retain_textures_for_external_submission(texture_views);
      program->resolve_ndarray_launch_context(launch_ctx);
      program->resolve_texture_launch_context(launch_ctx);
    }
    // Run cgraph loaded from AOT module
    dispatch.compiled_kernel->launch(launch_ctx);
  }
}

void CompiledGraph::jit_run(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args) const try {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->ensure_runtime_submission_allowed("Graph launch");
  }
  std::optional<Program::SNodeTreeLifecycleReadGuard> tree_lifecycle_guard;
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  std::optional<Program::RuntimeSubmissionScope> completion_scope;
  GraphRuntimeResourceViews resource_views;
  if (program != nullptr) {
    tree_lifecycle_guard.emplace(
        program->acquire_snode_tree_lifecycle_read_guard());
    program->validate_snode_tree_dependencies(snode_tree_dependencies);
    resource_views = graph_runtime_resource_views(args, program);
    if (!resource_views.empty()) {
      resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
      if (!resource_views.ndarrays.empty()) {
        program->retain_ndarrays_for_external_submission(
            resource_views.ndarrays.data(), resource_views.ndarrays.size());
      }
      if (!resource_views.runtime_storage.empty()) {
        program->retain_runtime_storage_for_graph_submission(
            resource_views.runtime_storage.data(),
            resource_views.runtime_storage.size());
      }
      if (!resource_views.textures.empty()) {
        program->retain_textures_for_external_submission(
            resource_views.textures.data(), resource_views.textures.size());
      }
    }
    completion_scope.emplace(program->acquire_runtime_submission_scope());
  }
#if defined(TI_WITH_CUDA)
  std::unique_lock<std::recursive_mutex> cuda_submission_lock;
  if (compile_config.arch == Arch::cuda) {
    cuda_submission_lock =
        CUDAContext::get_instance().get_submission_lock_guard();
  }
#endif
  for (const auto &dispatch : dispatches) {
    TI_ASSERT(dispatch.ti_kernel);
    LaunchContextBuilder launch_ctx(dispatch.ti_kernel);
    init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    // Compile & Run (JIT): The compilation result will be cached, so don't
    // worry that the kernels dispatched by this cgraph will be compiled
    // repeatedly.
    auto *prog = dispatch.ti_kernel->program;
    const auto &compiled_kernel_data = prog->compile_kernel(
        compile_config, prog->get_device_caps(), *dispatch.ti_kernel);
    prog->launch_kernel(compiled_kernel_data, launch_ctx);
  }
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

void CompiledGraph::jit_run_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache) const try {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->ensure_runtime_submission_allowed("cached Graph launch");
  }
  std::optional<Program::SNodeTreeLifecycleReadGuard> tree_lifecycle_guard;
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  std::optional<Program::RuntimeSubmissionScope> completion_scope;
  GraphRuntimeResourceViews resource_views;
  if (program != nullptr) {
    tree_lifecycle_guard.emplace(
        program->acquire_snode_tree_lifecycle_read_guard());
    resource_views = graph_runtime_resource_views(args, program);
    if (!resource_views.empty()) {
      resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
      if (!resource_views.ndarrays.empty()) {
        program->retain_ndarrays_for_external_submission(
            resource_views.ndarrays.data(), resource_views.ndarrays.size());
      }
      if (!resource_views.runtime_storage.empty()) {
        program->retain_runtime_storage_for_graph_submission(
            resource_views.runtime_storage.data(),
            resource_views.runtime_storage.size());
      }
      if (!resource_views.textures.empty()) {
        program->retain_textures_for_external_submission(
            resource_views.textures.data(), resource_views.textures.size());
      }
    }
    completion_scope.emplace(program->acquire_runtime_submission_scope());
  }
#if defined(TI_WITH_CUDA)
  // A graph is one submission transaction. This is required not only while
  // capturing: replaying a CUDA graph concurrently with an ordinary kernel on
  // the shared legacy default stream exposed invalid runtime state to both
  // callers once Python graph execution started releasing the GIL.
  std::unique_lock<std::recursive_mutex> cuda_submission_lock;
  if (compile_config.arch == Arch::cuda) {
    cuda_submission_lock =
        CUDAContext::get_instance().get_submission_lock_guard();
  }
#endif
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (program != nullptr &&
      (cache.validated_snode_tree_program != program ||
       cache.validated_snode_tree_epoch != tree_lifecycle_guard->epoch())) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      // The dependency is stale. Retire replay/cached launch state before
      // surfacing the rebuild requirement so no backend object keeps an
      // executable containing the old root binding.
      cache.cuda_graph_state.reset();
      cache.vulkan_graph_state.reset();
      cache.vulkan_inline_stats = {};
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard->epoch();
  }
#if defined(TI_WITH_CUDA)
  if (compile_config.arch == Arch::cuda) {
    TI_ASSERT(program != nullptr);
    if (try_run_cuda_graph(*this, compile_config, args, cache, *program,
                           &program->runtime_statistics())) {
      program->mark_runtime_submission(
          RuntimeSubmissionKind::kGraphBackendSubmission);
      return;
    }
    if (program != nullptr) {
      program->runtime_statistics().record_graph_ordinary_fallback();
    }
  }
#endif
#if defined(TI_WITH_VULKAN)
  if (compile_config.arch == Arch::vulkan) {
    if (try_run_vulkan_graph(*this, compile_config, args, cache,
                             program != nullptr
                                 ? &program->runtime_statistics()
                                 : nullptr)) {
      TI_ASSERT(program != nullptr);
      program->mark_runtime_submission(
          RuntimeSubmissionKind::kGraphBackendSubmission);
      return;
    }
    if (program != nullptr) {
      program->runtime_statistics().record_graph_ordinary_fallback();
    }
  }
#endif
  if (cache.kernels.size() != dispatches.size()) {
    cache.kernels.assign(dispatches.size(), {});
    cache.runtime_arg_plans.clear();
  }
  const bool use_cpu_runtime_arg_plan = arch_is_cpu(compile_config.arch);
  if (use_cpu_runtime_arg_plan &&
      cache.runtime_arg_plans.size() != dispatches.size()) {
    cache.runtime_arg_plans.clear();
    cache.runtime_arg_plans.reserve(dispatches.size());
    for (const auto &dispatch : dispatches) {
      cache.runtime_arg_plans.push_back(build_cpu_runtime_arg_plan(dispatch));
    }
  }
  const bool cache_compiled_kernel_data =
      arch_is_cpu(compile_config.arch) || compile_config.arch == Arch::cuda;
  for (std::size_t i = 0; i < dispatches.size(); ++i) {
    const auto &dispatch = dispatches[i];
    TI_ASSERT(dispatch.ti_kernel);
    LaunchContextBuilder launch_ctx(dispatch.ti_kernel);
    if (use_cpu_runtime_arg_plan && cache.runtime_arg_plans[i].cpu_fast_path) {
      init_runtime_context_from_plan(cache.runtime_arg_plans[i], args,
                                     launch_ctx);
    } else {
      init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    }
    auto *prog = dispatch.ti_kernel->program;
    auto &cached = cache.kernels[i];
    const CompiledKernelData *compiled_kernel_data =
        get_or_compile_cached_kernel(dispatch, compile_config, cached,
                                     cache_compiled_kernel_data);
#if defined(TI_WITH_LLVM)
    if (arch_is_cpu(compile_config.arch) &&
        try_launch_cached_llvm_kernel(prog, *compiled_kernel_data, cached,
                                      launch_ctx)) {
      continue;
    }
#endif
    prog->launch_kernel(*compiled_kernel_data, launch_ctx);
  }
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

bool CompiledGraph::jit_run_bounded_cuda_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache,
    Ndarray *predicate,
    int max_iterations,
    bool continue_while_nonzero) const try {
#if defined(TI_WITH_CUDA)
  if (compile_config.arch != Arch::cuda || predicate == nullptr) {
    return false;
  }
  Program *program = jit_graph_program(*this);
  if (program == nullptr) {
    return false;
  }
  program->ensure_runtime_submission_allowed(
      "bounded CUDA Graph launch");
  auto tree_lifecycle_guard =
      program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_views = graph_runtime_resource_views(args, program);
  std::optional<Program::RuntimeResourceGraphScope> resource_guard;
  if (!resource_views.empty()) {
    resource_guard.emplace(program->acquire_runtime_resource_graph_scope());
    if (!resource_views.ndarrays.empty()) {
      program->retain_ndarrays_for_external_submission(
          resource_views.ndarrays.data(), resource_views.ndarrays.size());
    }
    if (!resource_views.runtime_storage.empty()) {
      program->retain_runtime_storage_for_graph_submission(
          resource_views.runtime_storage.data(),
          resource_views.runtime_storage.size());
    }
    if (!resource_views.textures.empty()) {
      program->retain_textures_for_external_submission(
          resource_views.textures.data(), resource_views.textures.size());
    }
  }
  auto completion_scope = program->acquire_runtime_submission_scope();
  auto cuda_submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(cache.run_mutex);
  if (cache.validated_snode_tree_program != program ||
      cache.validated_snode_tree_epoch != tree_lifecycle_guard.epoch()) {
    try {
      program->validate_snode_tree_dependencies(snode_tree_dependencies);
    } catch (...) {
      cache.cuda_graph_state.reset();
      cache.kernels.clear();
      cache.runtime_arg_plans.clear();
      cache.validated_snode_tree_program = nullptr;
      cache.validated_snode_tree_epoch = 0;
      throw;
    }
    cache.validated_snode_tree_program = program;
    cache.validated_snode_tree_epoch = tree_lifecycle_guard.epoch();
  }
  if (!try_run_cuda_bounded_graph(
          *this, compile_config, args, cache, *program, *predicate,
          max_iterations, continue_while_nonzero,
          &program->runtime_statistics())) {
    return false;
  }
  program->mark_runtime_submission(
      RuntimeSubmissionKind::kGraphBackendSubmission);
  return true;
#else
  return false;
#endif
} catch (const BackendRuntimeError &error) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
    program->report_backend_runtime_error(error);
  }
  throw;
} catch (...) {
  Program *program = jit_graph_program(*this);
  if (program != nullptr) {
    program->record_runtime_submission_failure();
  }
  throw;
}

// static
void CompiledGraph::init_runtime_context(
    const std::vector<Arg> &paramter_list,
    const std::unordered_map<std::string, IValue> &args,
    LaunchContextBuilder &ctx) {
  TI_COMPILE_PROFILER("compiled_graph_init_runtime_context");
  for (int i = 0; i < paramter_list.size(); ++i) {
    auto &symbolic_arg = paramter_list[i];
    const std::vector<int> arg_id{i};
    auto found = args.find(symbolic_arg.name);
    TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                symbolic_arg.name);
    const aot::IValue &ival = found->second;
    if (symbolic_arg.tag == aot::ArgKind::kNdarray) {
      TI_ASSERT(ival.tag == aot::ArgKind::kNdarray);
      if (ival.runtime_storage != nullptr) {
        validate_graph_runtime_storage_argument(
            *ival.runtime_storage, symbolic_arg.name, symbolic_arg.dtype_id,
            symbolic_arg.field_dim, symbolic_arg.element_shape);
        ctx.set_arg_runtime_storage(arg_id, *ival.runtime_storage);
        continue;
      }
      Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
      TI_ERROR_IF(arr == nullptr, "Graph received a null Ndarray runtime arg");

      TI_ERROR_IF(arr->get_element_shape() != symbolic_arg.element_shape,
                  "Mismatched shape information for argument {}",
                  symbolic_arg.name);
      TI_ERROR_IF(arr->shape.size() != symbolic_arg.field_dim,
                  "Dispatch node is compiled for argument {} with "
                  "field_dim={} but got an ndarray with field_dim={}",
                  symbolic_arg.name, symbolic_arg.field_dim, arr->shape.size());

      DataType symbolic_arg_element_dtype = symbolic_arg.element_dtype();
      DataType symbolic_arg_primitive_dtype = symbolic_arg_element_dtype;
      if (symbolic_arg_element_dtype->is<TensorType>()) {
        symbolic_arg_primitive_dtype =
            symbolic_arg_element_dtype->cast<TensorType>()->get_element_type();
      }

      DataType arr_primitive_dtype = arr->dtype;
      if (arr->dtype->is<TensorType>()) {
        arr_primitive_dtype =
            arr->dtype->cast<TensorType>()->get_element_type();
      }

      TI_ERROR_IF(arr_primitive_dtype != symbolic_arg_primitive_dtype,
                  "Dispatch node is compiled for argument {} with "
                  "dtype={} but got an ndarray with dtype={}",
                  symbolic_arg.name, symbolic_arg_primitive_dtype.to_string(),
                  arr_primitive_dtype.to_string());
      ctx.set_arg_ndarray(arg_id, *arr);
    } else if (symbolic_arg.tag == aot::ArgKind::kScalar) {
      TI_ASSERT(ival.tag == aot::ArgKind::kScalar);
      // Matrix args are flattened so they're same as scalars.
      int type_size = data_type_size(symbolic_arg.dtype());
      switch (type_size) {
        case 1:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int8>(ival.val));
          break;
        case 2:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int16>(ival.val));
          break;
        case 4:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int32>(ival.val));
          break;
        case 8:
          ctx.set_arg(arg_id,
                      taichi_union_cast_with_different_sizes<int64>(ival.val));
          break;
        default:
          TI_ERROR("Unsupported type size {}", type_size);
      }
    } else if (symbolic_arg.tag == aot::ArgKind::kTexture) {
      TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
      Texture *tex = reinterpret_cast<Texture *>(ival.val);
      ctx.set_arg_texture(arg_id, *tex);
    } else if (symbolic_arg.tag == aot::ArgKind::kRWTexture) {
      TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
      Texture *tex = reinterpret_cast<Texture *>(ival.val);
      ctx.set_arg_rw_texture(arg_id, *tex);
    } else if (symbolic_arg.tag == aot::ArgKind::kMatrix) {
      TI_ASSERT(ival.tag == aot::ArgKind::kMatrix);
      Matrix *mat = reinterpret_cast<Matrix *>(ival.val);

      TI_ERROR_IF(symbolic_arg.element_shape.empty() ||
                      symbolic_arg.element_shape.size() > 2,
                  "Matrix argument {} has unsupported element shape {}",
                  symbolic_arg.name, symbolic_arg.element_shape);
      uint64_t symbolic_arg_size = 1;
      for (int dimension : symbolic_arg.element_shape) {
        TI_ERROR_IF(dimension <= 0,
                    "Matrix argument {} has invalid element shape {}",
                    symbolic_arg.name, symbolic_arg.element_shape);
        symbolic_arg_size *= static_cast<uint64_t>(dimension);
      }
      TI_ERROR_IF(symbolic_arg_size != mat->length(),
                  "Dispatch node is compiled for argument {} with "
                  "size={} but got a matrix with size={}",
                  symbolic_arg.name, symbolic_arg_size, mat->length());
      if (mat->ndim() != 0) {
        bool shape_matches = mat->ndim() == symbolic_arg.element_shape.size();
        if (shape_matches) {
          for (uint32_t axis = 0; axis < mat->ndim(); ++axis) {
            shape_matches &=
                mat->shape(axis) == symbolic_arg.element_shape[axis];
          }
        }
        // AOT graphs produced before the native vector descriptor encoded a
        // vector as [N, 1]. Keep that 0.5.x representation readable while new
        // Graph Args use the canonical rank-1 [N] tensor shape.
        const bool legacy_vector_shape =
            mat->ndim() == 1 && symbolic_arg.element_shape.size() == 2 &&
            symbolic_arg.element_shape[1] == 1 &&
            mat->shape(0) == symbolic_arg.element_shape[0];
        // Matrix values historically accepted a flat ti.Matrix([...]) whose
        // length matched an NxM annotation. Keep that source-compatible form,
        // but reject a genuinely rank-2 value with transposed dimensions.
        const bool legacy_flat_matrix =
            mat->ndim() == 1 && symbolic_arg.element_shape.size() == 2 &&
            mat->shape(0) == symbolic_arg_size;
        TI_ERROR_IF(!shape_matches && !legacy_vector_shape &&
                        !legacy_flat_matrix,
                    "Dispatch node is compiled for Matrix argument {} with "
                    "shape={} but got rank {} runtime data",
                    symbolic_arg.name, symbolic_arg.element_shape, mat->ndim());
      }
      TI_ERROR_IF(mat->length() * data_type_size(mat->dtype()) > 128,
                  "Matrix size={} is out of bound",
                  mat->length() * data_type_size(mat->dtype()));
      ctx.set_arg_matrix(i, *mat);
    } else {
      TI_ERROR("Error in compiled graph: unknown tag {}", int(ival.tag));
    }
  }
}

}  // namespace aot
}  // namespace taichi::lang
