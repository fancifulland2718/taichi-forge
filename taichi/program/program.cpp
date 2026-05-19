// Program, context for Taichi program execution

#include "program.h"

#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/struct/struct.h"
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/system/timeline.h"
#include "taichi/system/threading.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/math/arithmetic.h"
#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/program/parallel_executor.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_sort.h"
#endif

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#endif

#ifdef TI_WITH_VULKAN
#include "taichi/runtime/program_impls/vulkan/vulkan_program.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif
#ifdef TI_WITH_OPENGL
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/rhi/opengl/opengl_api.h"
#endif
#ifdef TI_WITH_DX11
#include "taichi/runtime/program_impls/dx/dx_program.h"
#include "taichi/rhi/dx/dx_api.h"
#endif
#ifdef TI_WITH_DX12
#include "taichi/runtime/program_impls/dx12/dx12_program.h"
#include "taichi/rhi/dx12/dx12_api.h"
#endif
#ifdef TI_WITH_METAL
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/rhi/metal/metal_api.h"
#endif  // TI_WITH_METAL

#if defined(_M_X64) || defined(__x86_64)
// For _MM_SET_FLUSH_ZERO_MODE
#include <xmmintrin.h>
#endif  // defined(_M_X64) || defined(__x86_64)

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace taichi::lang {
std::atomic<int> Program::num_instances_;

namespace {
struct CpuHistogramTaskContext {
  const int32_t *values{nullptr};
  int32_t *partial{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  int num_threads{1};
};

struct CpuReduceI32TaskContext {
  const int32_t *values{nullptr};
  int64_t *partial{nullptr};
  std::size_t n{0};
  int num_threads{1};
  int op{0};
};

struct CpuReduceF32TaskContext {
  const float *values{nullptr};
  float *partial{nullptr};
  std::size_t n{0};
  int num_threads{1};
  int op{0};
};

taichi::ThreadPool &get_cpu_primitive_thread_pool(int max_threads) {
  static std::mutex mutex;
  static std::unique_ptr<taichi::ThreadPool> pool;
  static int pool_threads = 0;
  std::lock_guard<std::mutex> lock(mutex);
  if (!pool || pool_threads < max_threads) {
    pool = std::make_unique<taichi::ThreadPool>(max_threads);
    pool_threads = max_threads;
  }
  return *pool;
}

int64_t cpu_reduce_i32_identity(int op) {
  if (op == 1) {
    return std::numeric_limits<int32_t>::max();
  }
  if (op == 2) {
    return std::numeric_limits<int32_t>::min();
  }
  return 0;
}

int64_t cpu_reduce_i32_combine(int64_t a, int64_t b, int op) {
  if (op == 1) {
    return std::min<int64_t>(a, b);
  }
  if (op == 2) {
    return std::max<int64_t>(a, b);
  }
  return a + b;
}

float cpu_reduce_f32_identity(int op) {
  if (op == 1) {
    return std::numeric_limits<float>::infinity();
  }
  if (op == 2) {
    return -std::numeric_limits<float>::infinity();
  }
  return 0.0f;
}

float cpu_reduce_f32_combine(float a, float b, int op) {
  if (op == 1) {
    return std::min(a, b);
  }
  if (op == 2) {
    return std::max(a, b);
  }
  return a + b;
}

void cpu_reduce_i32_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuReduceI32TaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  int64_t acc = cpu_reduce_i32_identity(ctx->op);
  for (std::size_t i = begin; i < end; ++i) {
    acc = cpu_reduce_i32_combine(acc, ctx->values[i], ctx->op);
  }
  ctx->partial[tid] = acc;
}

void cpu_reduce_f32_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuReduceF32TaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  float acc = cpu_reduce_f32_identity(ctx->op);
  for (std::size_t i = begin; i < end; ++i) {
    acc = cpu_reduce_f32_combine(acc, ctx->values[i], ctx->op);
  }
  ctx->partial[tid] = acc;
}

void store_i32_wrapped_from_i64(int32_t *output, int64_t value) {
  uint32_t wrapped = static_cast<uint32_t>(value);
  std::memcpy(output, &wrapped, sizeof(wrapped));
}

bool snode_tree_contains_hash(const SNode *snode) {
  if (snode == nullptr) {
    return false;
  }
  if (snode->type == SNodeType::hash) {
    return true;
  }
  for (const auto &child : snode->ch) {
    if (snode_tree_contains_hash(child.get())) {
      return true;
    }
  }
  return false;
}
}  // namespace

Program::Program(Arch desired_arch) : snode_rw_accessors_bank_(this) {
  TI_TRACE("Program initializing...");

  // For performance considerations and correctness of QuantFloatType
  // operations, we force floating-point operations to flush to zero on all
  // backends (including CPUs).
#if defined(_M_X64) || defined(__x86_64)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif  // defined(_M_X64) || defined(__x86_64)
#if defined(__arm64__) || defined(__aarch64__)
  // Enforce flush to zero on arm64 CPUs
  // https://developer.arm.com/documentation/100403/0201/register-descriptions/advanced-simd-and-floating-point-registers/aarch64-register-descriptions/fpcr--floating-point-control-register?lang=en
  std::uint64_t fpcr;
  __asm__ __volatile__("");
  __asm__ __volatile__("MRS %0, FPCR" : "=r"(fpcr));
  __asm__ __volatile__("");
  __asm__ __volatile__("MSR FPCR, %0"
                       :
                       : "ri"(fpcr | (1 << 24)));  // Bit 24 is FZ
  __asm__ __volatile__("");
#endif  // defined(__arm64__) || defined(__aarch64__)
  auto &config = compile_config_;
  config = default_compile_config;
  config.arch = desired_arch;
  config.fit();

  profiler = make_profiler(config.arch, config.kernel_profiler);
  if (arch_uses_llvm(config.arch)) {
#ifdef TI_WITH_LLVM
    if (config.arch != Arch::dx12) {
      program_impl_ = std::make_unique<LlvmProgramImpl>(config, profiler.get());
    } else {
      // NOTE: use Dx12ProgramImpl to avoid using LlvmRuntimeExecutor for dx12.
#ifdef TI_WITH_DX12
      TI_ASSERT(directx12::is_dx12_api_available());
      program_impl_ = std::make_unique<Dx12ProgramImpl>(config);
#else
      TI_ERROR("This taichi is not compiled with DX12");
#endif
    }
#else
    TI_ERROR("This taichi is not compiled with LLVM");
#endif
  } else if (config.arch == Arch::metal) {
#ifdef TI_WITH_METAL
    TI_ASSERT(metal::is_metal_api_available());
    program_impl_ = std::make_unique<MetalProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Metal")
#endif
  } else if (config.arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    TI_ASSERT(vulkan::is_vulkan_api_available());
    program_impl_ = std::make_unique<VulkanProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Vulkan")
#endif
  } else if (config.arch == Arch::dx11) {
#ifdef TI_WITH_DX11
    TI_ASSERT(directx11::is_dx_api_available());
    program_impl_ = std::make_unique<Dx11ProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with DX11");
#endif
  } else if (config.arch == Arch::opengl) {
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(false));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else if (config.arch == Arch::gles) {
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(true));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }

  // program_impl_ should be set in the if-else branch above
  TI_ASSERT(program_impl_);

  // Phase 1c-D: propagate the user's vulkan_sparse_experimental opt-in to
  // the process-global extension table BEFORE any is_extension_supported()
  // query (the very next block already calls it for Extension::assertion).
  // Sticky once true; OR'd with the legacy TI_VULKAN_SPARSE env var.
  set_vulkan_sparse_experimental(config.vulkan_sparse_experimental);
  // §13 (2026-05-02): default for vulkan_sparse_experimental is now true.
  // Emit a one-shot informational warning whenever the experimental sparse
  // path is exercised on Arch::vulkan, so users can correlate any unexpected
  // behaviour with this opt-in path. Skipped on cpu/cuda (which never read
  // the flag) and skipped when the user explicitly disables it.
  if (config.arch == Arch::vulkan && config.vulkan_sparse_experimental) {
    static bool sparse_warn_emitted = false;
    if (!sparse_warn_emitted) {
      sparse_warn_emitted = true;
      TI_WARN(
          "Vulkan sparse SNode support is experimental and enabled by default "
          "as of taichi-forge 0.3.x; pass ti.init(vulkan_sparse_experimental="
          "False) to disable if you observe regressions vs. cuda/cpu.");
    }
  }
  // G9.1: same propagation pattern for quant_array / bit_struct on Vulkan.
  set_vulkan_quant_experimental(config.vulkan_quant_experimental);

  Device *compute_device = nullptr;
  compute_device = program_impl_->get_compute_device();
  // Must have handled all the arch fallback logic by this point.
  TI_ASSERT_INFO(num_instances_ == 0, "Only one instance at a time");
  total_compilation_time_ = 0;
  num_instances_ += 1;
  SNode::counter = 0;

  result_buffer = nullptr;
  finalized_ = false;

  if (!is_extension_supported(config.arch, Extension::assertion)) {
    if (config.check_out_of_bound) {
      TI_WARN("Out-of-bound access checking is not supported on arch={}",
              arch_name(config.arch));
      config.check_out_of_bound = false;
    }
  }

  Timelines::get_instance().set_enabled(config.timeline);

  TI_TRACE("Program ({}) arch={} initialized.", fmt::ptr(this),
           arch_name(config.arch));
}

TypeFactory &Program::get_type_factory() {
  TI_WARN(
      "Program::get_type_factory() will be deprecated, Please use "
      "TypeFactory::get_instance()");
  return TypeFactory::get_instance();
}

Function *Program::create_function(const FunctionKey &func_key) {
  TI_TRACE("Creating function {}...", func_key.get_full_name());
  functions_.emplace_back(std::make_unique<Function>(this, func_key));
  TI_ASSERT(function_map_.count(func_key) == 0);
  function_map_[func_key] = functions_.back().get();
  return functions_.back().get();
}

const CompiledKernelData &Program::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  auto &mgr = program_impl_->get_kernel_compilation_manager();
  // P-Compile-6: apply per-kernel compile_tier override (if set) by passing
  // an effective CompileConfig copy down. CompileConfig::compile_tier is
  // already part of the offline cache key (offline_cache_util.cpp), so cache
  // entries for the same kernel under different tiers are automatically
  // segregated.
  const auto &override = kernel_def.get_compile_tier_override();
  if (override.has_value() && *override != compile_config.compile_tier) {
    CompileConfig effective_config = compile_config;
    effective_config.compile_tier = *override;
    const auto &ckd = mgr.load_or_compile(effective_config, caps, kernel_def);
    total_compilation_time_ += Time::get_time() - start_t;
    return ckd;
  }
  const auto &ckd = mgr.load_or_compile(compile_config, caps, kernel_def);
  total_compilation_time_ += Time::get_time() - start_t;
  return ckd;
}

// P5.b — batch / parallel kernel compilation.
//
// Design:
// 1. Compilation is dispatched to a ParallelExecutor with
//    `compile_config.num_compile_threads` workers.
// 2. All heavy lifting (IR passes, LLVM opt, SPIR-V codegen, GPU module load)
//    runs on worker threads. The main thread only submits + flushes.
// 3. Ordering: kernel compilation is order-independent at the C++ level —
//    @ti.func inlining and template specialization are resolved in Python
//    before a `Kernel` object ever reaches C++. Each kernel is compiled as a
//    self-contained unit.
// 4. Thread-safety:
//    - `KernelCompilationManager::load_or_compile` is guarded by its own
//      cache_mutex_ (P5.a) so concurrent cache hits/inserts are safe.
//    - LLVM: TaichiLLVMContext maintains per-thread_id state under
//      thread_map_mut_; first-touch on a worker lazily clones the runtime
//      module + struct_modules from the main thread (which is already
//      quiescent after materialize_runtime).
//    - CUDA: `cuModuleLoadDataEx` is serialized by CUDAContext::get_lock_guard
//      inside JITSessionCUDA; all optimization runs in parallel.
//    - Vulkan: SPIR-V codegen touches no shared state.
// 5. Error propagation: the first exception from any worker is captured and
//    rethrown on the calling thread after flush(); remaining workers still
//    finish their in-flight tasks so we never leave the executor in a bad
//    state.
//
// V7 (2026-04-26) — thread-local depth counter that tracks whether the
// current thread is acting as a compile_kernels outer worker. The LLVM
// codegen path consults this via Program::in_compile_kernels_worker() to
// avoid double-pool oversubscription. Only incremented when
// compile_config.compile_dag_scheduler is true.
namespace {
thread_local int g_compile_kernels_worker_depth = 0;
}  // namespace

bool Program::in_compile_kernels_worker() {
  return g_compile_kernels_worker_depth > 0;
}

// Caller contract: do NOT destroy SNode trees concurrently with this call.
void Program::compile_kernels(
    const CompileConfig &compile_config,
    const std::vector<const Kernel *> &kernels) {
  if (kernels.empty()) {
    return;
  }
  auto start_t = Time::get_time();
  const auto caps = get_device_caps();

  const int n_compile_threads =
      std::max(1, compile_config.num_compile_threads);
  int n_workers = std::min<int>(n_compile_threads, (int)kernels.size());

  auto &mgr = program_impl_->get_kernel_compilation_manager();
  const bool dag_mode = compile_config.compile_dag_scheduler;
  // V8.a (2026-04-26): when dag_mode is on and there are fewer kernels than
  // compile threads, skip the outer ParallelExecutor entirely. The serial
  // outer loop lets each kernel's inner offload pool (LLVM
  // compilation_workers / SPIR-V V2 std::async) consume the full T-wide
  // worker budget on its own. With V7 enabled the previous behaviour would
  // create only N outer workers and force inner-serial, leaving (T-N) cores
  // idle. See compile_doc/优化总规划.md §3.5.
  const bool prefer_inner_parallelism =
      dag_mode && (int)kernels.size() < n_compile_threads;
  if (n_workers <= 1 || prefer_inner_parallelism) {
    // Fast path: honour the same serial path as compile_kernel.
    for (auto *k : kernels) {
      mgr.load_or_compile(compile_config, caps, *k);
    }
    total_compilation_time_ += Time::get_time() - start_t;
    return;
  }

  std::mutex err_mu;
  std::exception_ptr first_error;

  {
    ParallelExecutor exec("compile_kernels", n_workers);
    for (auto *k : kernels) {
      exec.enqueue([&, k]() {
        // V7: mark this worker so the LLVM inner pool stays serial.
        if (dag_mode) {
          ++g_compile_kernels_worker_depth;
        }
        try {
          mgr.load_or_compile(compile_config, caps, *k);
        } catch (...) {
          std::lock_guard<std::mutex> g(err_mu);
          if (!first_error) {
            first_error = std::current_exception();
          }
        }
        if (dag_mode) {
          --g_compile_kernels_worker_depth;
        }
      });
    }
    // ~ParallelExecutor runs flush() implicitly via its destructor.
    exec.flush();
  }

  total_compilation_time_ += Time::get_time() - start_t;
  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

void Program::launch_kernel(const CompiledKernelData &compiled_kernel_data,
                            LaunchContextBuilder &ctx) {
  program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data, ctx);
  const bool check_runtime_error =
      compile_config().debug || hash_snode_tree_count_ > 0;
  if (check_runtime_error && arch_uses_llvm(compiled_kernel_data.arch())) {
    program_impl_->check_runtime_error(result_buffer);
  }
}

void Program::materialize_runtime() {
  program_impl_->materialize_runtime(profiler.get(), &result_buffer);
}

static void remove_rw_accessor_cache(
    SNode *parent_snode,
    SNodeRwAccessorsBank *snode_rw_accessors_bank) {
  for (int i = 0; i < (int)parent_snode->ch.size(); i++) {
    auto child_snode = parent_snode->ch[i].get();
    if (child_snode->type == SNodeType::place) {
      snode_rw_accessors_bank->remove_cached_kernels(child_snode);
    }
    remove_rw_accessor_cache(child_snode, snode_rw_accessors_bank);
  }
}

void Program::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_ASSERT(arch_uses_llvm(compile_config().arch) ||
            compile_config().arch == Arch::vulkan ||
            compile_config().arch == Arch::dx11 ||
            compile_config().arch == Arch::dx12);

  // When accessing a ti.field at Python scope, SNodeRwAccessorsBank creates
  // a Taichi Kernel to read/write the field in a JIT manner, which caches the
  // compiled JIT Kernel so as to avoid recompilation when accessing the same
  // field.

  // This cache uses the place-SNode's address (SNode*) as the key,
  // which becomes unsafe once the SNodeTree gets destroyed and that
  // place-SNode's address gets reused by another SNode. We have to remove all
  // cached kernels upon SNodeTree destruction.
  SNode *root = snode_tree->root();
  const bool contains_hash = snode_tree_contains_hash(root);

  // Traverse SNodeTree to remove all cached RWAccessor kernels
  remove_rw_accessor_cache(root, &snode_rw_accessors_bank_);

  program_impl_->destroy_snode_tree(snode_tree);
  if (contains_hash) {
    --hash_snode_tree_count_;
  }
  free_snode_tree_ids_.push(snode_tree->id());
}

SNodeTree *Program::add_snode_tree(std::unique_ptr<SNode> root,
                                   bool compile_only) {
  const int id = allocate_snode_tree_id();
  auto tree = std::make_unique<SNodeTree>(id, std::move(root));
  tree->root()->set_snode_tree_id(id);
  const bool contains_hash = snode_tree_contains_hash(tree->root());
  if (compile_only) {
    program_impl_->compile_snode_tree_types(tree.get());
  } else {
    program_impl_->materialize_snode_tree(tree.get(), result_buffer);
  }
  if (contains_hash) {
    ++hash_snode_tree_count_;
  }
  if (id < snode_trees_.size()) {
    snode_trees_[id] = std::move(tree);
  } else {
    TI_ASSERT(id == snode_trees_.size());
    snode_trees_.push_back(std::move(tree));
  }
  return snode_trees_[id].get();
}

SNode *Program::get_snode_root(int tree_id) {
  return snode_trees_[tree_id]->root();
}

void Program::synchronize() {
  program_impl_->synchronize();
}

StreamSemaphore Program::flush() {
  return program_impl_->flush();
}

int Program::get_snode_tree_size() {
  return snode_trees_.size();
}

Kernel &Program::get_snode_reader(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_reader_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto ret = Stmt::make<FrontendReturnStmt>(ExprGroup(
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices)));
    builder.insert(std::move(ret));
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_ret(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

Kernel &Program::get_snode_writer(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_writer_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto expr =
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices);
    expr.type_check(&this->compile_config());
    auto argload_expr = Expr::make<ArgLoadExpression>(
        std::vector<int>{snode->num_active_indices},
        snode->dt->get_compute_type());
    argload_expr->type_check(&this->compile_config());
    builder.insert_assignment(expr, argload_expr, expr->dbg_info);
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_scalar_param(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

uint64 Program::fetch_result_uint64(int i) {
  return program_impl_->fetch_result_uint64(i, result_buffer);
}

void Program::finalize() {
  if (finalized_) {
    return;
  }

  synchronize();
  TI_TRACE("Program finalizing...");

  synchronize();
  if (compile_config().arch == Arch::vulkan) {
    vulkan_radix_sort_clear_workspace();
    vulkan_scan_clear_workspace();
    vulkan_compact_clear_workspace();
    vulkan_histogram_clear_workspace();
    vulkan_reduce_clear_workspace();
  }
  textures_.clear();
  argpacks_.clear();
  ndarrays_.clear();
  if (arch_uses_llvm(compile_config().arch) ||
      compile_config().arch == Arch::vulkan) {
    program_impl_->finalize();
  }

  Stmt::reset_counter();

  finalized_ = true;
  num_instances_ -= 1;
  program_impl_->dump_cache_data_to_disk();
  compile_config_ = default_compile_config;
  TI_TRACE("Program ({}) finalized_.", fmt::ptr(this));

  // Reset memory pool
  HostMemoryPool::get_instance().reset();
}

int Program::default_block_dim(const CompileConfig &config) {
  if (arch_is_cpu(config.arch)) {
    return config.default_cpu_block_dim;
  } else {
    return config.default_gpu_block_dim;
  }
}

void Program::print_memory_profiler_info() {
  program_impl_->print_memory_profiler_info(snode_trees_, result_buffer);
}

std::size_t Program::get_snode_num_dynamically_allocated(SNode *snode) {
  return program_impl_->get_snode_num_dynamically_allocated(snode,
                                                            result_buffer);
}

void Program::reset_hash_snode_probe_stats() {
  program_impl_->reset_hash_snode_probe_stats(result_buffer);
}

std::vector<int64> Program::get_hash_snode_probe_stats() {
  return program_impl_->get_hash_snode_probe_stats(result_buffer);
}

Ndarray *Program::create_ndarray(const DataType type,
                                 const std::vector<int> &shape,
                                 ExternalArrayLayout layout,
                                 bool zero_fill,
                                 const DebugInfo &dbg_info) {
  auto arr = std::make_unique<Ndarray>(this, type, shape, layout, dbg_info);
  if (zero_fill) {
    Arch arch = compile_config().arch;
    if (arch_is_cpu(arch) || arch == Arch::cuda || arch == Arch::amdgpu) {
      fill_ndarray_fast_u32(arr.get(), /*data=*/0);
    } else if (arch != Arch::dx12) {
      // Device api support for dx12 backend are not complete yet
      Stream *stream =
          program_impl_->get_compute_device()->get_compute_stream();
      auto [cmdlist, res] = stream->new_command_list_unique();
      TI_ASSERT(res == RhiResult::success);
      cmdlist->buffer_fill(arr->ndarray_alloc_.get_ptr(0),
                           arr->get_element_size() * arr->get_nelement(),
                           /*data=*/0);
      stream->submit_synced(cmdlist.get());
    }
  }
  auto arr_ptr = arr.get();
  ndarrays_.insert({arr_ptr, std::move(arr)});
  return arr_ptr;
}

ArgPack *Program::create_argpack(const DataType dt) {
  auto pack = std::make_unique<ArgPack>(this, dt);
  auto pack_ptr = pack.get();
  argpacks_.insert({pack_ptr, std::move(pack)});
  return pack_ptr;
}

void Program::delete_ndarray(Ndarray *ndarray) {
  // [Note] Ndarray memory deallocation
  // Ndarray's memory allocation is managed by Taichi and Python can control
  // this via Taichi indirectly. For example, when an ndarray is GC-ed in
  // Python, it signals Taichi to free its memory allocation. But Taichi will
  // make sure **no pending kernels to be executed needs the ndarray** before it
  // actually frees the memory. When `ti.reset()` is called, all ndarrays
  // allocated in this program should be gone and no longer valid in Python.
  // This isn't the best implementation, ndarrays should be managed by taichi
  // runtime instead of this giant program and it should be freed when:
  // - Python GC signals taichi that it's no longer useful
  // - All kernels using it are executed.
  if (ndarrays_.count(ndarray) &&
      !program_impl_->used_in_kernel(ndarray->ndarray_alloc_.alloc_id)) {
    ndarrays_.erase(ndarray);
  }
}

void Program::delete_argpack(ArgPack *argpack) {
  // [Note] Argpack memory deallocation
  // Argpack's memory allocation is managed by Taichi and Python can control
  // this via Taichi indirectly. For example, when an argpack is GC-ed in
  // Python, it signals Taichi to free its memory allocation. But Taichi will
  // make sure **no pending kernels to be executed needs the argpack** before it
  // actually frees the memory. When `ti.reset()` is called, all argpack
  // allocated in this program should be gone and no longer valid in Python.
  // This isn't the best implementation, argpacks should be managed by taichi
  // runtime instead of this giant program and it should be freed when:
  // - Python GC signals taichi that it's no longer useful
  // - All kernels using it are executed.
  if (argpacks_.count(argpack) &&
      !program_impl_->used_in_kernel(argpack->argpack_alloc_.alloc_id)) {
    argpacks_.erase(argpack);
  }
}

Texture *Program::create_texture(BufferFormat buffer_format,
                                 const std::vector<int> &shape) {
  if (shape.size() == 1) {
    textures_.push_back(
        std::make_unique<Texture>(this, buffer_format, shape[0], 1, 1));
  } else if (shape.size() == 2) {
    textures_.push_back(
        std::make_unique<Texture>(this, buffer_format, shape[0], shape[1], 1));
  } else if (shape.size() == 3) {
    textures_.push_back(std::make_unique<Texture>(this, buffer_format, shape[0],
                                                  shape[1], shape[2]));
  } else {
    TI_ERROR("Texture shape invalid");
  }
  return textures_.back().get();
}

intptr_t Program::get_ndarray_data_ptr_as_int(const Ndarray *ndarray) {
  uint64_t *data_ptr{nullptr};
  if (arch_is_cpu(compile_config().arch) ||
      compile_config().arch == Arch::cuda ||
      compile_config().arch == Arch::amdgpu) {
    // For the LLVM backends, device allocation is a physical pointer.
    data_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
  }

  return reinterpret_cast<intptr_t>(data_ptr);
}

void Program::fill_ndarray_fast_u32(Ndarray *ndarray, uint32_t val) {
  // This is a temporary solution to bypass device api.
  // Should be moved to CommandList once available in CUDA.
  program_impl_->fill_ndarray(
      ndarray->ndarray_alloc_,
      ndarray->get_nelement() * ndarray->get_element_size() / sizeof(uint32_t),
      val);
}

bool Program::cuda_cub_radix_sort_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_radix_sort_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_radix_sort_ndarray(Ndarray *keys,
                                                 Ndarray *values,
                                                 int key_type,
                                                 int mode,
                                                 int nan_policy) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB sort is only available on the CUDA backend.");
  TI_ERROR_IF(!keys, "CUDA CUB sort received null keys ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1,
              "CUDA CUB sort currently expects a 1D ndarray.");
  const bool has_values = values != nullptr;
  if (has_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "CUDA CUB sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "CUDA CUB sort keys and values must have the same length.");
    TI_ERROR_IF(values->get_element_size() != sizeof(int32_t),
                "CUDA CUB sort currently expects i32 payload values.");
  }
#ifdef TI_WITH_CUDA
  std::size_t expected_key_size = 0;
  TI_ERROR_IF(mode < 0 || mode > 1,
              "CUDA CUB sort received an unsupported sort mode.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CUDA CUB sort received an unsupported NaN policy.");
  const auto cub_key_type = static_cast<cuda::CubSortKeyType>(key_type);
  const auto cub_mode = static_cast<cuda::CubSortMode>(mode);
  const auto cub_nan_policy =
      static_cast<cuda::CubSortNanPolicy>(nan_policy);
  switch (cub_key_type) {
    case cuda::CubSortKeyType::u32:
    case cuda::CubSortKeyType::i32:
    case cuda::CubSortKeyType::f32:
      expected_key_size = 4;
      break;
    case cuda::CubSortKeyType::u64:
    case cuda::CubSortKeyType::i64:
    case cuda::CubSortKeyType::f64:
      expected_key_size = 8;
      break;
  }
  TI_ERROR_IF(expected_key_size == 0,
              "CUDA CUB sort received an unsupported key type.");
  TI_ERROR_IF(keys->get_element_size() != expected_key_size,
              "CUDA CUB sort key dtype does not match the requested key type.");
  if (cub_mode == cuda::CubSortMode::split32) {
    TI_ERROR_IF(cub_key_type != cuda::CubSortKeyType::u64 &&
                    cub_key_type != cuda::CubSortKeyType::i64 &&
                    cub_key_type != cuda::CubSortKeyType::f64,
                "CUDA CUB split32 sort supports only u64/i64/f64 keys.");
  }
#endif
  auto key_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys));
  auto value_ptr = has_values
                       ? reinterpret_cast<void *>(
                             get_ndarray_data_ptr_as_int(values))
                       : nullptr;
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_radix_sort(
      key_ptr, value_ptr, static_cast<int>(keys->get_nelement()),
      cub_key_type, cub_mode, cub_nan_policy, has_values, stream, this);
#else
  TI_ERROR(
      "CUDA CUB sort requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_radix_sort_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_radix_sort_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_radix_sort_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_radix_sort_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cuda_cub_scan_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_inclusive_scan_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_inclusive_scan_ndarray(Ndarray *data,
                                                     int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB scan is only available on the CUDA backend.");
  TI_ERROR_IF(!data, "CUDA CUB scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CUDA CUB scan currently expects a 1D ndarray.");
#ifdef TI_WITH_CUDA
  const auto cub_value_type = static_cast<cuda::CubScanValueType>(value_type);
  std::size_t expected_value_size = 0;
  switch (cub_value_type) {
    case cuda::CubScanValueType::i32:
      expected_value_size = sizeof(int32_t);
      break;
  }
  TI_ERROR_IF(expected_value_size == 0,
              "CUDA CUB scan received an unsupported value type.");
  TI_ERROR_IF(data->get_element_size() != expected_value_size,
              "CUDA CUB scan dtype does not match the requested value type.");
  auto data_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_inclusive_scan(
      data_ptr, static_cast<int>(data->get_nelement()), cub_value_type, stream,
      this);
#else
  TI_ERROR(
      "CUDA CUB scan requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_scan_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_inclusive_scan_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_scan_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_inclusive_scan_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cuda_cub_select_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_select_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_select_i32_ndarray(Ndarray *values,
                                                 Ndarray *flags,
                                                 Ndarray *output,
                                                 Ndarray *count) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB select is only available on the CUDA backend.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CUDA CUB select received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "CUDA CUB select currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement() ||
                  values->get_nelement() > output->get_nelement(),
              "CUDA CUB select expects values/flags to have the same length "
              "and output to have at least that many elements.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "CUDA CUB select count ndarray must have at least one element.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t) ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  output->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "CUDA CUB select currently supports only i32 values, flags, "
              "output, and count.");
#ifdef TI_WITH_CUDA
  auto values_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto flags_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(flags));
  auto output_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  auto count_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(count));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_select_flagged(
      values_ptr, flags_ptr, output_ptr, count_ptr,
      static_cast<int>(values->get_nelement()), cuda::CubSelectValueType::i32,
      stream, this);
#else
  TI_ERROR(
      "CUDA CUB select requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_select_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_select_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_select_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_select_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cuda_cub_histogram_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_histogram_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_histogram_i32_ndarray(Ndarray *values,
                                                    Ndarray *bins) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB histogram is only available on CUDA.");
  TI_ERROR_IF(!values || !bins,
              "CUDA CUB histogram received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "CUDA CUB histogram currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t) ||
                  bins->get_element_size() != sizeof(int32_t),
              "CUDA CUB histogram currently expects i32 values and bins.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "CUDA CUB histogram expects at least one bin.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto bins_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(bins));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_histogram_even(
      values_ptr, bins_ptr, static_cast<int>(values->get_nelement()),
      static_cast<int>(bins->get_nelement()), cuda::CubHistogramValueType::i32,
      stream, this);
#else
  TI_ERROR(
      "CUDA CUB histogram requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_histogram_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_histogram_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_histogram_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_histogram_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cuda_cub_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_reduce_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_reduce_ndarray(Ndarray *values,
                                             Ndarray *output,
                                             int value_type,
                                             int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB reduce currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA CUB reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB reduce output ndarray must have at least one item.");
  TI_ERROR_IF(values->get_element_size() != output->get_element_size(),
              "CUDA CUB reduce expects matching input/output element sizes.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t),
              "CUDA CUB reduce currently expects 32-bit values.");
  TI_ERROR_IF(value_type < 0 || value_type > 1,
              "CUDA CUB reduce received an unsupported value type.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA CUB reduce received an unsupported op.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_reduce(values_ptr, output_ptr,
                          static_cast<int>(values->get_nelement()),
                          static_cast<cuda::CubReduceValueType>(value_type),
                          static_cast<cuda::CubReduceOp>(op), stream, this);
#else
  TI_ERROR(
      "CUDA CUB reduce requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_reduce_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_reduce_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_reduce_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_reduce_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cpu_scan_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_inclusive_scan_ndarray(Ndarray *data,
                                                int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scan is only available on CPU backends.");
  TI_ERROR_IF(!data, "CPU native scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CPU native scan currently expects a 1D ndarray.");
  TI_ERROR_IF(value_type != 0,
              "CPU native scan currently supports only i32 values.");
  TI_ERROR_IF(data->get_element_size() != sizeof(int32_t),
              "CPU native scan currently expects i32 data.");

  uint32_t *ptr =
      reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(data));
  TI_ERROR_IF(!ptr, "CPU native scan received a null data pointer.");
  uint32_t prefix = 0;
  const std::size_t n = data->get_nelement();
  for (std::size_t i = 0; i < n; ++i) {
    prefix += ptr[i];
    ptr[i] = prefix;
  }
  return 0;
}

std::size_t Program::cpu_scan_workspace_bytes() const {
  return 0;
}

bool Program::cpu_compact_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_compact_i32_ndarray(Ndarray *values,
                                             Ndarray *flags,
                                             Ndarray *output,
                                             Ndarray *count) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native compact is only available on CPU backends.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CPU native compact received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "CPU native compact expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement(),
              "CPU native compact values and flags must have the same length.");
  TI_ERROR_IF(output->get_nelement() < values->get_nelement(),
              "CPU native compact output must have at least input length.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "CPU native compact count must contain at least one item.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t) ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  output->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "CPU native compact currently expects i32 values, flags, "
              "output, and count.");

  auto *values_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(values));
  auto *flags_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(flags));
  auto *output_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output));
  auto *count_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(count));
  TI_ERROR_IF(!values_ptr || !flags_ptr || !output_ptr || !count_ptr,
              "CPU native compact received a null data pointer.");

  std::size_t written = 0;
  const std::size_t n = values->get_nelement();
  for (std::size_t i = 0; i < n; ++i) {
    if (flags_ptr[i] != 0) {
      output_ptr[written++] = values_ptr[i];
    }
  }
  TI_ERROR_IF(written > static_cast<std::size_t>(
                            std::numeric_limits<int32_t>::max()),
              "CPU native compact output count exceeds i32 range.");
  count_ptr[0] = static_cast<int32_t>(written);
  return 0;
}

std::size_t Program::cpu_compact_workspace_bytes() const {
  return 0;
}

bool Program::cpu_histogram_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_histogram_i32_ndarray(Ndarray *values,
                                               Ndarray *bins) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native histogram is only available on CPU backends.");
  TI_ERROR_IF(!values || !bins,
              "CPU native histogram received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "CPU native histogram expects 1D ndarrays.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t) ||
                  bins->get_element_size() != sizeof(int32_t),
              "CPU native histogram currently expects i32 values and bins.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "CPU native histogram expects at least one bin.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(
                      std::numeric_limits<int32_t>::max()),
              "CPU native histogram input is too large for i32 bin counts.");

  auto *values_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(values));
  auto *bins_ptr = reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(bins));
  TI_ERROR_IF(!values_ptr || !bins_ptr,
              "CPU native histogram received a null data pointer.");

  const std::size_t n = values->get_nelement();
  const std::size_t num_bins = bins->get_nelement();
  for (std::size_t i = 0; i < num_bins; ++i) {
    bins_ptr[i] = 0;
  }

  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && num_bins <= 4096 &&
                            target_threads > 1;
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<int32_t> partial(
        static_cast<std::size_t>(num_threads) * num_bins, 0);
    CpuHistogramTaskContext ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.num_bins = num_bins;
    ctx.num_threads = num_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(num_threads, num_threads, &ctx,
             [](void *raw_ctx, int /*thread_id*/, int task_id) {
               auto *ctx = static_cast<CpuHistogramTaskContext *>(raw_ctx);
               const int tid = task_id;
               const std::size_t begin =
                   ctx->n * static_cast<std::size_t>(tid) /
                   static_cast<std::size_t>(ctx->num_threads);
               const std::size_t end =
                   ctx->n * static_cast<std::size_t>(tid + 1) /
                   static_cast<std::size_t>(ctx->num_threads);
               int32_t *local =
                   ctx->partial +
                   static_cast<std::size_t>(tid) * ctx->num_bins;
               for (std::size_t i = begin; i < end; ++i) {
                 int32_t bin = ctx->values[i];
                 if (bin >= 0 &&
                     static_cast<std::size_t>(bin) < ctx->num_bins) {
                   local[bin] += 1;
                 }
               }
             });
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      int32_t total = 0;
      for (int tid = 0; tid < num_threads; ++tid) {
        total += partial[static_cast<std::size_t>(tid) * num_bins + bin];
      }
      bins_ptr[bin] = total;
    }
    return partial.size() * sizeof(int32_t);
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t bin = values_ptr[i];
    if (bin >= 0 && static_cast<std::size_t>(bin) < num_bins) {
      bins_ptr[bin] += 1;
    }
  }
  return 0;
}

std::size_t Program::cpu_histogram_workspace_bytes() const {
  return 0;
}

bool Program::cpu_reduce_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_reduce_ndarray(Ndarray *values,
                                        Ndarray *output,
                                        int value_type,
                                        int op) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native reduce is only available on CPU backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CPU native reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CPU native reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CPU native reduce output must contain at least one item.");
  TI_ERROR_IF(values->get_element_size() != output->get_element_size(),
              "CPU native reduce expects matching input/output element sizes.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t),
              "CPU native reduce currently expects 32-bit values.");
  TI_ERROR_IF(value_type < 0 || value_type > 1,
              "CPU native reduce received an unsupported value type.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CPU native reduce received an unsupported op.");

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && target_threads > 1;

  if (value_type == 0) {
    auto *values_ptr =
        reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(values));
    auto *output_ptr =
        reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output));
    TI_ERROR_IF(!values_ptr || !output_ptr,
                "CPU native reduce received a null data pointer.");

    int64_t result = cpu_reduce_i32_identity(op);
    if (use_parallel) {
      const int num_threads = target_threads;
      std::vector<int64_t> partial(num_threads);
      CpuReduceI32TaskContext ctx;
      ctx.values = values_ptr;
      ctx.partial = partial.data();
      ctx.n = n;
      ctx.num_threads = num_threads;
      ctx.op = op;
      auto &pool = get_cpu_primitive_thread_pool(max_threads);
      pool.run(num_threads, num_threads, &ctx, cpu_reduce_i32_task);
      for (int tid = 0; tid < num_threads; ++tid) {
        result = cpu_reduce_i32_combine(result, partial[tid], op);
      }
      if (op == 0) {
        store_i32_wrapped_from_i64(output_ptr, result);
      } else {
        output_ptr[0] = static_cast<int32_t>(result);
      }
      return partial.size() * sizeof(int64_t);
    }

    for (std::size_t i = 0; i < n; ++i) {
      result = cpu_reduce_i32_combine(result, values_ptr[i], op);
    }
    if (op == 0) {
      store_i32_wrapped_from_i64(output_ptr, result);
    } else {
      output_ptr[0] = static_cast<int32_t>(result);
    }
    return 0;
  }

  auto *values_ptr =
      reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(values));
  auto *output_ptr =
      reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native reduce received a null data pointer.");

  float result = cpu_reduce_f32_identity(op);
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<float> partial(num_threads);
    CpuReduceF32TaskContext ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.num_threads = num_threads;
    ctx.op = op;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(num_threads, num_threads, &ctx, cpu_reduce_f32_task);
    for (int tid = 0; tid < num_threads; ++tid) {
      result = cpu_reduce_f32_combine(result, partial[tid], op);
    }
    output_ptr[0] = result;
    return partial.size() * sizeof(float);
  }

  for (std::size_t i = 0; i < n; ++i) {
    result = cpu_reduce_f32_combine(result, values_ptr[i], op);
  }
  output_ptr[0] = result;
  return 0;
}

std::size_t Program::cpu_reduce_workspace_bytes() const {
  return 0;
}

std::pair<const ArgPackType *, size_t>
Program::get_argpack_type_with_data_layout(const ArgPackType *old_ty,
                                           const std::string &layout) {
  // Convert to StructType
  auto *struct_type_old =
      TypeFactory::get_instance()
          .get_struct_type(old_ty->elements(), old_ty->get_layout())
          ->as<StructType>();
  // Call get_struct_type_with_data_layout
  auto [struct_type, size] = program_impl_->get_struct_type_with_data_layout(
      const_cast<StructType *>(struct_type_old), layout);
  // Convert back to ArgPackType
  auto *new_ty =
      TypeFactory::get_instance()
          .get_argpack_type(struct_type->elements(), struct_type->get_layout())
          ->as<ArgPackType>();
  return {new_ty, size};
}

std::pair<const StructType *, size_t> Program::get_struct_type_with_data_layout(
    const StructType *old_ty,
    const std::string &layout) {
  return program_impl_->get_struct_type_with_data_layout(old_ty, layout);
}

Program::~Program() {
  finalize();
}

DeviceCapabilityConfig translate_devcaps(const std::vector<std::string> &caps) {
  // Each device capability assignment is named like this:
  // - `spirv_version=1.3`
  // - `spirv_has_int8`
  DeviceCapabilityConfig cfg{};
  for (const std::string &cap : caps) {
    std::string_view key;
    uint32_t value;
    size_t ieq = cap.find('=');
    if (ieq == std::string::npos) {
      key = cap;
      value = 1;
    } else {
      key = std::string_view(cap.c_str(), ieq);
      value = (uint32_t)std::atol(cap.c_str() + ieq + 1);
    }
    DeviceCapability devcap = str2devcap(key);
    cfg.set(devcap, value);
  }

  // Assign default caps (that always present).
  if (!cfg.contains(DeviceCapability::spirv_version)) {
    cfg.set(DeviceCapability::spirv_version, 0x10300);
  }
  return cfg;
}

std::unique_ptr<AotModuleBuilder> Program::make_aot_module_builder(
    Arch arch,
    const std::vector<std::string> &caps) {
  DeviceCapabilityConfig cfg = translate_devcaps(caps);
  // FIXME: This couples the runtime backend with the target AOT backend. E.g.
  // If we want to build a Metal AOT module, we have to be on the macOS
  // platform. Consider decoupling this part
  if (arch_uses_llvm(compile_config().arch) ||
      compile_config().arch == Arch::metal ||
      compile_config().arch == Arch::vulkan ||
      compile_config().arch == Arch::opengl ||
      compile_config().arch == Arch::gles ||
      compile_config().arch == Arch::dx12) {
    return program_impl_->make_aot_module_builder(cfg);
  }
  return nullptr;
}

int Program::allocate_snode_tree_id() {
  if (free_snode_tree_ids_.empty()) {
    return snode_trees_.size();
  } else {
    int id = free_snode_tree_ids_.top();
    free_snode_tree_ids_.pop();
    return id;
  }
}

void Program::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  program_impl_->enqueue_compute_op_lambda(op, image_refs);
}

}  // namespace taichi::lang
