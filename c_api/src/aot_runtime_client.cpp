#include "taichi/runtime/aot_runtime_client.h"

#include <atomic>
#include <exception>
#include <limits>
#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#include "taichi/runtime/native_image.h"
#include "taichi/runtime/texture_storage.h"
#include "native_resource_registry.h"
#include "jit_module.h"
#include "taichi_core_impl.h"
#include "execution_domain.h"
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/execution_domain.h"
#endif
#ifdef TI_WITH_LLVM
#include "taichi_llvm_impl.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi_vulkan_impl.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {
class BorrowedStorageRetirement final : public RuntimeClient {
 public:
  explicit BorrowedStorageRetirement(::Runtime &runtime) : runtime_(runtime) {
  }
  void retire() override {
    // Same order as JIT code revocation: Program already owns its resource
    // guard. Never enter the borrower's state mutex from this callback.
    // A global retirement snapshot may invoke us after the borrower has
    // destroyed Runtime. The owner disarms this callback before detaching it.
    if (!active.load(std::memory_order_acquire))
      return;
    std::lock_guard<std::recursive_mutex> lock(runtime_.submission_mutex());
    if (auto live = storage.lock())
      runtime_.retire_borrowed_memory(live);
  }
  std::weak_ptr<::Runtime::BorrowedMemory> storage;
  std::atomic<bool> active{true};

 private:
  ::Runtime &runtime_;
};

struct BorrowedStorageOwner {
  Program &program;
  std::shared_ptr<BorrowedStorageRetirement> retirement;
  std::shared_ptr<void> lease;
  ~BorrowedStorageOwner() {
    retirement->active.store(false, std::memory_order_release);
    program.detach_runtime_client(retirement.get());
  }
};

class BorrowedSampledImage final : public NativeSampledImage {
 public:
  BorrowedSampledImage(ImageParams parameters,
                       Program::TextureResourceLease lease,
                       Arch arch,
                       Device &device,
                       CudaTextureStorage *cuda_storage)
      : NativeSampledImage(parameters, true),
        lease_(std::move(lease)),
        arch_(arch),
        device_(device),
        cuda_storage_(cuda_storage) {
    bytes_ = image_logical_bytes(parameters);
  }
  Arch arch() const override { return arch_; }
  Device &device() const override { return device_; }
  std::size_t workspace_bytes() const override { return 0; }
  std::size_t logical_bytes() const override { return bytes_; }
  std::uint64_t cuda_sampled_object() const override {
    return lease_->get_cuda_texture_object();
  }
  void upload_cuda(std::uint64_t source, void *stream) override {
#ifdef TI_WITH_CUDA
    if (!cuda_storage_)
      throw std::invalid_argument("Borrowed image has no CUDA storage");
    cuda_storage_->upload_device_async(source, stream);
#else
    throw std::invalid_argument("CUDA image upload is unavailable");
#endif
  }
  DeviceAllocation gfx_sampled_allocation() const override {
    return lease_->get_device_allocation();
  }

 private:
  Program::TextureResourceLease lease_;
  Arch arch_;
  Device &device_;
  CudaTextureStorage *cuda_storage_;
  std::size_t bytes_;
};

#ifdef TI_WITH_VULKAN
class BorrowedVulkanRuntime final : public ::VulkanRuntime {
 public:
  explicit BorrowedVulkanRuntime(gfx::GfxRuntime &runtime) : runtime_(runtime) {
  }
  Device &get() override {
    return *runtime_.get_ti_device();
  }
  gfx::GfxRuntime &get_gfx_runtime() override {
    return runtime_;
  }

 private:
  gfx::GfxRuntime &runtime_;
};
#endif

std::unique_ptr<::Runtime> borrow_backend(Program &program) {
  const auto arch = program.compile_config().arch;
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch) || arch == Arch::cuda) {
    auto *impl = static_cast<LlvmProgramImpl *>(program.get_program_impl());
    return std::make_unique<capi::LlvmRuntime>(*impl->get_runtime_executor(),
                                               program.result_buffer);
  }
#endif
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan) {
    auto *runtime = program.get_gfx_runtime_for_borrowing();
    if (!runtime)
      throw std::runtime_error("The Vulkan runtime is not materialized");
    return std::make_unique<BorrowedVulkanRuntime>(*runtime);
  }
#endif
  throw std::invalid_argument(
      "AOT runtime borrowing requires CPU, CUDA or Vulkan");
}
}  // namespace

struct AotRuntimeClient::State {
  State(Program &program, std::unique_ptr<::Runtime> runtime)
      : program(&program),
        runtime(std::move(runtime)),
        arch(program.compile_config().arch),
        device_id(reinterpret_cast<std::uintptr_t>(&this->runtime->get())) {
    if (&this->runtime->get() != program.get_compute_device())
      throw std::logic_error("AOT borrower did not share its Program device");
    this->runtime->borrow_runtime_fault_domain(
        program.runtime_fault_domain(),
        arch == Arch::cuda ? program.cuda_execution_domains() : nullptr);
  }
  std::recursive_mutex mutex;
  Program *program;
  std::unique_ptr<::Runtime> runtime;
  const Arch arch;
  const std::uintptr_t device_id;
  std::atomic<bool> closed{false};
  std::vector<Program::RuntimeResourceSubmissionGuard> scopes;

  void finish(bool synchronize) {
    closed.store(true, std::memory_order_release);
    auto released = std::move(runtime);
    program = nullptr;
    if (released) {
      try {
        released->retire_resources(synchronize);
      } catch (...) {
        // The failed drain retains a finite code/storage closure. Its members
        // reference Runtime, so quarantining only recordings would leave
        // dangling resource destructors after this borrowed adapter is freed.
        (void)released.release();
        throw;
      }
    }
  }
};

AotRuntimeClient::AotRuntimeClient(std::unique_ptr<State> state)
    : state_(std::move(state)) {
}
AotRuntimeClient::~AotRuntimeClient() = default;

std::shared_ptr<AotRuntimeClient> AotRuntimeClient::create(Program &program) {
  auto guard = program.acquire_runtime_resource_submission_guard();
  program.ensure_runtime_submission_allowed("borrow AOT runtime");
  auto result = std::shared_ptr<AotRuntimeClient>(new AotRuntimeClient(
      std::make_unique<State>(program, borrow_backend(program))));
  program.retain_runtime_client(result);
  return result;
}

void AotRuntimeClient::enter() {
  auto &state = *state_;
  state.mutex.lock();
  try {
    // Preserve one empty scope for closed wrappers so late close/GC can clear
    // their Python registrations without calling a dead backend.
    Program::RuntimeResourceSubmissionGuard guard;
    if (!state.closed.load(std::memory_order_acquire)) {
      try {
        guard = state.program->acquire_runtime_resource_submission_guard();
        state.program->ensure_runtime_submission_allowed("borrowed AOT call");
        if (!state.program->runtime_submission_allowed()) {
          state.closed.store(true, std::memory_order_release);
          guard.unlock();
        }
      } catch (const TaichiRuntimeError &) {
        // The retained registration transfers cleanup to the owner finalizer.
        state.closed.store(true, std::memory_order_release);
        if (guard.owns_lock())
          guard.unlock();
      }
    }
    state.scopes.emplace_back(std::move(guard));
  } catch (...) {
    state.mutex.unlock();
    throw;
  }
}

void AotRuntimeClient::leave() {
  auto &state = *state_;
  if (state.scopes.empty())
    throw std::logic_error("Unbalanced borrowed AOT scope");
  if (state.program && state.scopes.back().owns_lock())
    state.program->mark_runtime_submission_pending();
  state.scopes.pop_back();
  state.mutex.unlock();
}

void AotRuntimeClient::close() {
  enter();
  auto &state = *state_;
  std::exception_ptr failure;
  if (state.runtime && state.scopes.back().owns_lock()) {
    auto *program = state.program;
    try {
      state.finish(true);
    } catch (...) {
      failure = std::current_exception();
    }
    program->detach_runtime_client(this);
  }
  leave();
  if (failure)
    std::rethrow_exception(failure);
}

void AotRuntimeClient::retire() {
  std::lock_guard<std::recursive_mutex> lock(state_->mutex);
  state_->finish(false);
}

bool AotRuntimeClient::closed() const {
  return state_->closed.load(std::memory_order_acquire);
}
std::uintptr_t AotRuntimeClient::handle() const {
  std::lock_guard<std::recursive_mutex> lock(state_->mutex);
  return reinterpret_cast<std::uintptr_t>(state_->runtime.get());
}
std::uintptr_t AotRuntimeClient::device_identity() const {
  return state_->device_id;
}
Arch AotRuntimeClient::arch() const {
  return state_->arch;
}

std::shared_ptr<cuda::ExecutionDomain> AotRuntimeClient::execution_domain(
    std::uintptr_t handle) {
#ifdef TI_WITH_CUDA
  auto &state = *state_;
  std::lock_guard<std::recursive_mutex> lock(state.mutex);
  if (state.closed || !state.runtime || state.arch != Arch::cuda)
    throw std::runtime_error("Canvas domain requires a live CUDA borrower");
  auto domain = capi::get_execution_domain(
      reinterpret_cast<TiCudaExecutionDomain>(handle));
  if (!domain->belongs_to(state.runtime->runtime_fault_domain()) ||
      !domain->snapshot().accepting)
    throw std::invalid_argument(
        "Canvas domain belongs to another or closed runtime");
  return domain;
#else
  throw std::runtime_error("CUDA execution domains are unavailable");
#endif
}

RuntimeCompletion borrow_aot_completion(std::uintptr_t handle) {
#ifdef TI_WITH_CUDA
  return capi::get_completion(reinterpret_cast<TiCompletion>(handle));
#else
  throw std::runtime_error("CUDA completion bridge is unavailable");
#endif
}

std::uintptr_t AotRuntimeClient::import_jit_kernels(
    const std::vector<JitKernelImport> &kernels) {
  auto &state = *state_;
  // Hold the borrower alive, then follow the Program's tree-before-resource
  // order. The registered tree callback never reacquires this state mutex.
  std::lock_guard<std::recursive_mutex> lock(state.mutex);
  if (state.closed || !state.program || !state.runtime)
    throw std::runtime_error("AOT runtime is closed");
  if (!state.scopes.empty())
    throw std::logic_error(
        "Borrowing JIT code must run outside a borrowed submission scope");
  auto tree_guard = state.program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_guard =
      state.program->acquire_runtime_resource_submission_guard();
  state.program->ensure_runtime_submission_allowed("borrow selected JIT code");
  std::lock_guard<std::recursive_mutex> runtime_lock(
      state.runtime->submission_mutex());
  return capi::import_jit_module(*state.program, *state.runtime, kernels);
}

std::uintptr_t AotRuntimeClient::import_texture(const Texture *texture) {
  enter();
  try {
    auto &state = *state_;
    if (state.closed || !state.runtime || !state.scopes.back().owns_lock())
      throw std::runtime_error("AOT runtime is closed");
    if (state.arch != Arch::cuda && state.arch != Arch::vulkan)
      throw std::invalid_argument("AOT texture import requires CUDA or Vulkan");
    if (!texture)
      throw std::invalid_argument("AOT import requires a live JIT Texture");
    // Validate the owner's live view before dereferencing the foreign pointer.
    state.program->validate_textures_for_external_submission(&texture, 1);
    auto lease = state.program->acquire_texture_external_lease(texture);
    if ((state.arch == Arch::cuda &&
         (lease->get_mip_levels() != 1 || lease->sampler_config_.has_extended_sampling())) ||
        !is_float_sampled_texture_format(lease->get_buffer_format()))
      throw std::invalid_argument(
          "AOT texture import requires the qualified sampled format/sampler");
    ImageParams params{};
    const auto size = lease->get_size();
    params.x = size[0];
    params.y = size[1];
    params.z = size[2];
    params.dimension = lease->get_dimension();
    params.format = lease->get_buffer_format();
    params.sampler_config = lease->sampler_config_;
    params.mip_levels = lease->get_mip_levels();
    params.usage = ImageAllocUsage::Sampled;
    auto *cuda_storage = lease->cuda_texture_.get();
    if ((state.arch == Arch::cuda && !cuda_storage) ||
        (state.arch == Arch::vulkan &&
         lease->get_device_allocation().device != &state.runtime->get()))
      throw std::invalid_argument("AOT texture storage belongs to another backend/device");
#ifdef TI_WITH_VULKAN
    if (state.arch == Arch::vulkan) {
      auto *device = dynamic_cast<vulkan::VulkanDevice *>(&state.runtime->get());
      if (!device)
        throw std::invalid_argument("AOT texture requires its Vulkan device");
      const auto image = std::get<0>(device->get_vk_image(lease->get_device_allocation()));
      // Preserve the allocation's capabilities; imports neither grant missing
      // storage usage nor erase storage/attachment usage of a JIT image.
      if (!(image->usage & VK_IMAGE_USAGE_SAMPLED_BIT))
        throw std::invalid_argument("Vulkan texture import requires sampled image usage");
      if (image->usage & VK_IMAGE_USAGE_STORAGE_BIT)
        params.usage = params.usage | ImageAllocUsage::Storage;
      if (image->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
        params.usage = params.usage | ImageAllocUsage::Attachment;
    }
#endif
    // Import follows JIT's existing initialization contract: the caller provides
    // valid contents before sampling. No copy or new sampler is introduced.
    auto object = std::make_shared<BorrowedSampledImage>(
        params, std::move(lease), state.arch, state.runtime->get(), cuda_storage);
    TiNativeResource handle;
    {
      std::lock_guard<std::recursive_mutex> lock(state.runtime->submission_mutex());
      handle = state.runtime->native_resource_registry().publish(
          std::move(object), {}, {});
    }
    leave();
    return reinterpret_cast<std::uintptr_t>(handle);
  } catch (...) {
    leave();
    throw;
  }
}

AotRuntimeClient::NdarrayImport AotRuntimeClient::import_dense_view(
    const storage::DenseStorageDescriptor &descriptor) {
  auto &state = *state_;
  std::lock_guard<std::recursive_mutex> lock(state.mutex);
  if (state.closed || !state.program || !state.runtime)
    throw std::runtime_error("AOT runtime is closed");
  if (!state.scopes.empty())
    throw std::logic_error(
        "Borrowing dense storage must run outside a borrowed submission scope");
  auto tree_guard = state.program->acquire_snode_tree_lifecycle_read_guard();
  auto resource_guard =
      state.program->acquire_runtime_resource_submission_guard();
  state.program->ensure_runtime_submission_allowed("borrow dense view");
  std::lock_guard<std::recursive_mutex> runtime_lock(
      state.runtime->submission_mutex());
  if (!descriptor.scalar_type()->is<PrimitiveType>())
    throw std::invalid_argument(
        "Dense view requires scalar or tensor elements");
  auto shape = [](const std::vector<std::int64_t> &extents) {
    std::vector<int> result;
    if (extents.size() > 16)
      throw std::invalid_argument(
          "Dense view exceeds C API dimension capacity");
    for (auto extent : extents) {
      if (extent < 0 || extent > std::numeric_limits<int>::max())
        throw std::invalid_argument("Dense view extent exceeds C API capacity");
      result.push_back(static_cast<int>(extent));
    }
    return result;
  };
  NdarrayImport result{0, 0, descriptor.scalar_type()->to_string(),
                       shape(descriptor.index_shape()),
                       shape(descriptor.element_shape())};
  auto borrowed = std::make_shared<::Runtime::BorrowedMemory>();
  auto retirement = std::make_shared<BorrowedStorageRetirement>(*state.runtime);
  auto owner = std::shared_ptr<BorrowedStorageOwner>(
      new BorrowedStorageOwner{*state.program, retirement, {}});
  retirement->storage = borrowed;
  auto binding = state.program->borrow_dense_storage(descriptor, retirement);
  owner->lease = std::move(binding.lease);
  if (binding.allocation.device != &state.runtime->get())
    throw std::invalid_argument("Dense view belongs to another backend/device");
  borrowed->owner = std::move(owner);
  borrowed->external_access = std::move(binding.external_access);
  const auto handle = state.runtime->register_memory(
      binding.allocation, binding.byte_size, false, binding.byte_offset, {},
      binding.byte_offset, false, std::move(borrowed));
  std::get<0>(result) = reinterpret_cast<std::uintptr_t>(handle);
  std::get<1>(result) = binding.byte_size;
  return result;
}

AotRuntimeClient::NdarrayImport AotRuntimeClient::import_ndarray(
    const Ndarray *array) {
  enter();
  try {
    auto &state = *state_;
    if (state.closed || !state.runtime || !state.scopes.back().owns_lock())
      throw std::runtime_error("AOT runtime is closed");
    if (!array)
      throw std::invalid_argument("AOT import requires a live JIT ndarray");
    // Check the owner's live map before dereferencing the source pointer.
    state.program->validate_ndarrays_for_external_submission(&array, 1);
    auto lease = state.program->acquire_ndarray_external_lease(
        array->runtime_resource_handle());
    const auto element_shape = lease->get_element_shape();
    if (lease->layout == ExternalArrayLayout::kSOA && !element_shape.empty())
      throw std::invalid_argument(
          "AOT ndarray import requires compact AOS layout");
    const auto dtype = lease->get_element_data_type();
    if (!dtype->is<PrimitiveType>())
      throw std::invalid_argument(
          "AOT ndarray import requires scalar or tensor elements");
    if (lease->shape.size() > 16 || element_shape.size() > 16)
      throw std::invalid_argument(
          "AOT ndarray rank exceeds the C API dimension capacity");
    const auto width = lease->get_element_size();
    const auto count = lease->get_nelement();
    if (!width || count > std::numeric_limits<std::uint64_t>::max() / width)
      throw std::invalid_argument("AOT ndarray extent overflows address space");
    const auto bytes = static_cast<std::uint64_t>(width) * count;
    NdarrayImport result{0, bytes, dtype->to_string(), lease->shape,
                         element_shape};
    const auto allocation = lease->get_device_allocation();
    if (allocation.device != &state.runtime->get())
      throw std::invalid_argument(
          "AOT ndarray storage belongs to another device");
    auto owner =
        std::make_shared<Program::NdarrayResourceLease>(std::move(lease));
    const auto handle = state.runtime->register_memory(allocation, bytes, false,
                                                       0, std::move(owner));
    std::get<0>(result) = reinterpret_cast<std::uintptr_t>(handle);
    leave();
    return result;
  } catch (...) {
    leave();
    throw;
  }
}
}  // namespace taichi::lang
