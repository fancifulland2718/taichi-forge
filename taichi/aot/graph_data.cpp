#include "taichi/aot/graph_data.h"
#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/program/kernel.h"
#include "taichi/program/matrix.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/type_factory.h"

#include <cstring>
#include <memory>

#if defined(TI_WITH_LLVM)
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#endif

#if defined(TI_WITH_CUDA)
#include <algorithm>
#include <optional>

#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

#if defined(TI_WITH_VULKAN)
#include "taichi/runtime/gfx/kernel_launcher.h"
#endif

namespace taichi::lang {
namespace aot {

namespace {

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
      arg_plan.arg_buffer_offset =
          args_type->get_element_offset(arg_plan.arg_id);
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
    Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
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
  {
    TI_PROFILER("launch_llvm_kernel");
    launcher->launch_llvm_kernel(handle, launch_ctx);
  }
  prog->check_runtime_error_after_kernel_launch(compiled);
  return true;
}
#endif

}  // namespace

#if defined(TI_WITH_CUDA)
namespace {

struct CudaGraphArgSignatureEntry {
  std::string name;
  uint64 value{0};
  ArgKind tag{ArgKind::kScalar};

  bool operator==(const CudaGraphArgSignatureEntry &other) const {
    return name == other.name && value == other.value && tag == other.tag;
  }
};

struct CudaGraphCapturePacket {
  cuda::KernelLauncher *launcher{nullptr};
  cuda::KernelLauncher::GraphLaunchPacket packet;
};

class CudaGraphCaptureStream {
 public:
  CudaGraphCaptureStream() {
    CUDADriver::get_instance().stream_create(&stream_, CU_STREAM_NON_BLOCKING);
  }

  ~CudaGraphCaptureStream() {
    if (stream_ != nullptr) {
      CUDADriver::get_instance().stream_destroy(stream_);
    }
  }

  void *get() const {
    return stream_;
  }

 private:
  void *stream_{nullptr};
};

struct CudaGraphState {
  std::vector<CudaGraphArgSignatureEntry> signature;
  std::vector<CudaGraphCapturePacket> packets;
  std::unique_ptr<CudaGraphCaptureStream> capture_stream;
  CUgraphExec graph_exec{nullptr};
  bool disabled{false};

  ~CudaGraphState() {
    destroy();
  }

  void *ensure_capture_stream() {
    if (capture_stream == nullptr) {
      capture_stream = std::make_unique<CudaGraphCaptureStream>();
    }
    return capture_stream->get();
  }

  void destroy() {
    auto &context = CUDAContext::get_instance();
    auto &driver = CUDADriver::get_instance();
    context.make_current();
    if (graph_exec != nullptr) {
      // Replay stays on the default stream to preserve the ordering contract
      // visible to ordinary CUDA launches.
      driver.stream_synchronize(nullptr);
      driver.graph_exec_destroy(graph_exec);
      graph_exec = nullptr;
    }
    void *stream = capture_stream == nullptr ? nullptr : capture_stream->get();
    bool stream_ordered_free = false;
    for (auto &packet : packets) {
      if (packet.packet.device_arg_buffer != nullptr) {
        if (context.supports_mem_pool()) {
          driver.mem_free_async_impl(packet.packet.device_arg_buffer,
                                     stream);
          stream_ordered_free = true;
        } else {
          driver.mem_free(packet.packet.device_arg_buffer);
        }
        packet.packet.device_arg_buffer = nullptr;
      }
    }
    if (stream_ordered_free) {
      driver.stream_synchronize(stream);
    }
    packets.clear();
    signature.clear();
  }
};

CudaGraphState *get_cuda_graph_state(CompiledGraphJITCache &cache) {
  if (cache.cuda_graph_state == nullptr) {
    cache.cuda_graph_state = new CudaGraphState();
  }
  return static_cast<CudaGraphState *>(cache.cuda_graph_state);
}

std::optional<std::vector<CudaGraphArgSignatureEntry>>
make_cuda_graph_signature(const std::unordered_map<std::string, IValue> &args) {
  std::vector<CudaGraphArgSignatureEntry> signature;
  signature.reserve(args.size());
  for (const auto &kv : args) {
    if (kv.second.tag != ArgKind::kNdarray) {
      return std::nullopt;
    }
    signature.push_back({kv.first, kv.second.val, kv.second.tag});
  }
  std::sort(signature.begin(), signature.end(),
            [](const CudaGraphArgSignatureEntry &lhs,
               const CudaGraphArgSignatureEntry &rhs) {
              return lhs.name < rhs.name;
            });
  return signature;
}

bool try_run_cuda_graph(const CompiledGraph &graph,
                        const CompileConfig &compile_config,
                        const std::unordered_map<std::string, IValue> &args,
                        CompiledGraphJITCache &cache) {
  if (compile_config.debug) {
    return false;
  }
  if (graph.dispatches.empty()) {
    return false;
  }
  auto signature = make_cuda_graph_signature(args);
  if (!signature.has_value()) {
    return false;
  }
  CUDAContext::get_instance().make_current();
  auto state = get_cuda_graph_state(cache);
  if (state->disabled) {
    return false;
  }
  if (state->graph_exec != nullptr && state->signature == *signature) {
    CUDADriver::get_instance().graph_launch(state->graph_exec, nullptr);
    return true;
  }

  state->destroy();
  state->signature = *signature;
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
      state->destroy();
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
    CudaGraphCapturePacket capture_packet;
    capture_packet.launcher = launcher;
    if (!launcher->prepare_cuda_graph_launch(
            handle, launch_ctx, capture_packet.packet, capture_stream)) {
      driver.stream_synchronize(capture_stream);
      state->destroy();
      return false;
    }
    state->packets.push_back(std::move(capture_packet));
  }

  driver.stream_synchronize(capture_stream);
  auto capture_lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  auto begin_err = driver.stream_begin_capture.call_with_warning(
      capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_err != CUDA_SUCCESS) {
    state->disabled = true;
    state->destroy();
    return false;
  }
  for (const auto &packet : state->packets) {
    packet.launcher->capture_cuda_graph_launch(packet.packet, capture_stream);
  }
  CUgraph captured_graph{nullptr};
  auto end_err = driver.stream_end_capture.call_with_warning(
      capture_stream, &captured_graph);
  if (end_err != CUDA_SUCCESS || captured_graph == nullptr) {
    state->disabled = true;
    state->destroy();
    return false;
  }
  auto instantiate_err = driver.graph_instantiate_with_flags.call_with_warning(
      &state->graph_exec, captured_graph, 0);
  driver.graph_destroy(captured_graph);
  if (instantiate_err != CUDA_SUCCESS || state->graph_exec == nullptr) {
    state->disabled = true;
    state->destroy();
    return false;
  }
  driver.graph_launch(state->graph_exec, nullptr);
  return true;
}

}  // namespace
#endif

#if defined(TI_WITH_VULKAN)
namespace {

bool try_run_vulkan_graph(const CompiledGraph &graph,
                          const CompileConfig &compile_config,
                          const std::unordered_map<std::string, IValue> &args,
                          CompiledGraphJITCache &cache) {
  if (compile_config.debug || graph.dispatches.size() <= 1) {
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
    gfx_dispatches.push_back({handle, launch_contexts.back().get()});
  }

  if (gfx_launcher == nullptr) {
    return false;
  }
  return gfx_launcher->runtime()->try_launch_graph(
      gfx_dispatches, static_cast<const void *>(&cache));
}

}  // namespace
#endif

CompiledGraphJITCache::~CompiledGraphJITCache() {
#if defined(TI_WITH_CUDA)
  delete static_cast<CudaGraphState *>(cuda_graph_state);
#endif
  cuda_graph_state = nullptr;
}

void CompiledGraph::run(
    const std::unordered_map<std::string, IValue> &args) const {
  for (const auto &dispatch : dispatches) {
    TI_ASSERT(dispatch.compiled_kernel);
    LaunchContextBuilder launch_ctx(dispatch.compiled_kernel);
    init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    // Run cgraph loaded from AOT module
    dispatch.compiled_kernel->launch(launch_ctx);
  }
}

void CompiledGraph::jit_run(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args) const {
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
}

void CompiledGraph::jit_run_cached(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args,
    CompiledGraphJITCache &cache) const {
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
#if defined(TI_WITH_CUDA)
  if (compile_config.arch == Arch::cuda &&
      try_run_cuda_graph(*this, compile_config, args, cache)) {
    return;
  }
#endif
#if defined(TI_WITH_VULKAN)
  if (compile_config.arch == Arch::vulkan &&
      try_run_vulkan_graph(*this, compile_config, args, cache)) {
    return;
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
      Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);

      TI_ERROR_IF(arr->get_element_shape() != symbolic_arg.element_shape,
                  "Mismatched shape information for argument {}",
                  symbolic_arg.name);
      TI_ERROR_IF(arr->shape.size() != symbolic_arg.field_dim,
                  "Dispatch node is compiled for argument {} with "
                  "field_dim={} but got an ndarray with field_dim={}",
                  symbolic_arg.name, symbolic_arg.field_dim, arr->shape.size());

      // CGraph uses aot::Arg as symbolic argument, which represents
      // TensorType via combination of element_shape and PrimitiveTypeID
      // Therefore we only check for element_type for now.
      //
      // TODO(zhanlue): Replace all "element_shape + PrimitiveType" use cases
      // with direct use of "TensorType",
      //                In the end, "element_shape" should only appear inside
      //                TensorType and nowhere else.
      //
      //                This refactor includes aot::Arg, kernel::Arg,
      //                MetalDataType, and more...
      DataType symbolic_arg_primitive_dtype = symbolic_arg.dtype();
      if (symbolic_arg.dtype()->is<TensorType>()) {
        symbolic_arg_primitive_dtype =
            symbolic_arg.dtype()->cast<TensorType>()->get_element_type();
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

      uint32_t symbolic_arg_size = (uint32_t)(symbolic_arg.element_shape[0] *
                                              symbolic_arg.element_shape[1]);
      TI_ERROR_IF(symbolic_arg_size != mat->length(),
                  "Dispatch node is compiled for argument {} with "
                  "size={} but got a matrix with size={}",
                  symbolic_arg.name, symbolic_arg_size, mat->length());
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
