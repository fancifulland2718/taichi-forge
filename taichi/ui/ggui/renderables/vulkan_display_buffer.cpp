#include "taichi/ui/ggui/renderables/vulkan_display_buffer.h"

#include <atomic>
#include <limits>
#include "taichi/program/ndarray.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/program/storage_view.h"
#include "taichi/runtime/vulkan_execution_domain.h"

namespace taichi::ui::vulkan {
using namespace taichi::lang;

namespace {
using Domain = taichi::lang::vulkan::ExecutionDomain;
using Lease = std::shared_ptr<Program::NdarrayResourceLease>;
std::atomic<std::uint64_t> next_display_buffer_identity{1};

// Neither Commands nor completion operands refer back to the pool owner or
// to a DisplayStorage facade. Renderer reference counts remain meaningful.
struct CopyPayload final : RuntimeCompletionResources {
  std::vector<Lease> sources;
  std::vector<DeviceAllocationUnique> buffers;
  std::uint32_t source_kind{0};
  std::atomic<std::size_t> source_count{0};
  std::size_t retained_resource_count(std::uint32_t kind) const noexcept override {
    return kind == source_kind ? source_count.load(std::memory_order_acquire) : 0;
  }
  void retire() {
    // A concurrent completion query can briefly retain this payload after
    // publishing completion. Release device objects at the proven owner
    // retirement boundary, not at the last ticket/shared_ptr destruction.
    buffers.clear();
    sources.clear();
    source_count.store(0, std::memory_order_release);
  }
};

class CopyOwner final : public RuntimeClient {
 public:
  struct Slot {
    RuntimeCompletion copy;
    std::vector<StreamSemaphore> readers;
  };
  std::shared_ptr<RuntimeFaultDomain> fault;
  Program *program{nullptr};  // Identity only after cold preparation.
  std::shared_ptr<CopyPayload> payload;
  std::vector<FieldInfo> sources;
  std::vector<std::shared_ptr<const Domain::Commands>> commands;
  std::vector<Slot> slots;
  mutable std::mutex mutex;
  bool closed{false};
  bool graphics_unsealed{false};

  void check_open() const {
    fault->throw_if_new_submission_disallowed("Vulkan display copy");
    TI_ERROR_IF(closed, "Vulkan display storage is closed");
    TI_ERROR_IF(graphics_unsealed,
                "Vulkan display submission has no graphics retirement proof");
  }

  bool slot_ready(std::size_t index) const {
    check_open();
    const auto &slot = slots[index];
    if (slot.copy.valid() && !slot.copy.done())
      return false;
    for (const auto &reader : slot.readers)
      if (reader && !reader->is_ready())
        return false;
    return true;
  }

  // ExecutionDomain has stopped this instance and proved its exact copy
  // prefix before invoking us. Independently prove the graphics consumers.
  // A failed drain leaves the entire closure intact for domain quarantine.
  void retire() override {
    std::unique_lock<std::mutex> lock(mutex);
    if (closed)
      return;
    TI_ERROR_IF(!fault->backend_calls_safe(),
                "Cannot prove Vulkan display retirement after a fatal fault");
    TI_ERROR_IF(graphics_unsealed,
                "Cannot retire an unsealed Vulkan display submission");
    for (const auto &slot : slots) {
      for (const auto &reader : slot.readers)
        TI_ERROR_IF(reader && !reader->wait(),
                    "Vulkan display reader has no retirement proof");
    }
    auto released_commands = std::move(commands);
    auto released_slots = std::move(slots);
    auto released_payload = std::move(payload);
    closed = true;
    lock.unlock();
    // Last lease/native owner destruction must not reenter while we hold the
    // display mutex. No new access can enter after closed was published.
    released_commands.clear();
    for (auto &slot : released_slots) {
      slot.copy = {};
      for (auto &reader : slot.readers)
        reader.reset();
    }
    released_payload->retire();
    released_payload.reset();
  }
};
}  // namespace

struct VulkanDisplayBuffer::Pool {
  std::shared_ptr<Domain> domain;
  std::shared_ptr<taichi::lang::vulkan::ExecutionInstance> registration;
  std::shared_ptr<CopyOwner> owner;
  int width{0}, height{0};
  storage::DenseStorageBuildResult description{
      storage::StorageFailureReason::kUnsupportedStorageKind, std::nullopt};

  void close() {
    if (registration)
      domain->retire_instance(registration);
  }
  ~Pool() {
    try {
      close();
    } catch (...) {
      // The domain's registration retains the failed owner, not this facade.
    }
  }
};

std::vector<std::shared_ptr<DisplayStorage>> VulkanDisplayBuffer::create_pool(
    AppContext &context,
    std::shared_ptr<Domain> domain,
    const std::vector<FieldInfo> &sources,
    std::size_t capacity,
    std::size_t graphics_capacity) {
  auto *program = context.prog();
  TI_ERROR_IF(!program || context.config.ti_arch != Arch::vulkan ||
                  context.config.ggui_arch != Arch::vulkan ||
                  program->get_graphics_device() != &context.device() ||
                  !domain || !domain->belongs_to(program->runtime_fault_domain()),
              "Vulkan display requires its borrowed Program and device");
  TI_ERROR_IF(sources.empty() || capacity < 2 || graphics_capacity == 0,
              "Prepare nonempty sources and at least two display slots");
  TI_ERROR_IF(sources.size() > std::numeric_limits<std::size_t>::max() / capacity,
              "Vulkan display command count overflows");
  auto guard = program->acquire_runtime_resource_submission_guard();
  program->ensure_runtime_submission_allowed("prepare Vulkan display sources");
  auto &device = static_cast<taichi::lang::vulkan::VulkanDevice &>(context.device());
  auto pool = std::make_shared<Pool>();
  pool->domain = std::move(domain);
  pool->owner = std::make_shared<CopyOwner>();
  auto &owner = *pool->owner;
  owner.fault = program->runtime_fault_domain();
  owner.program = program;
  owner.payload = std::make_shared<CopyPayload>();
  owner.sources = sources;
  owner.payload->sources.reserve(sources.size());
  std::size_t bytes = 0;
  for (const auto &source : sources) {
    TI_ERROR_IF(!source.valid || source.field_source != FieldSource::TaichiNDarray ||
                    !source.runtime_resource_handle ||
                    source.runtime_resource_program != program ||
                    source.dtype != PrimitiveType::u32 || source.shape.size() != 2 ||
                    source.shape[0] <= 0 || source.shape[1] <= 0,
                "Vulkan display sources must be retained same-Program 2D u32 Ndarrays");
    auto lease = std::make_shared<Program::NdarrayResourceLease>(
        program->acquire_ndarray_external_lease(source.runtime_resource_handle));
    auto *array = lease->get();
    TI_ERROR_IF(lease->handle() != source.runtime_resource_handle ||
                    array->get_device_allocation() != source.dev_alloc ||
                    source.dev_alloc.device != &device ||
                    array->dtype != PrimitiveType::u32 ||
                    !array->get_element_shape().empty() || array->shape != source.shape,
                "Vulkan display source identity, allocation or layout mismatch");
    const auto width = static_cast<std::size_t>(source.shape[0]);
    const auto height = static_cast<std::size_t>(source.shape[1]);
    TI_ERROR_IF(width > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) / height,
                "Vulkan display source byte extent overflows");
    const auto source_bytes = width * height * sizeof(std::uint32_t);
    TI_ERROR_IF(width * height - 1 >
                    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
                "Vulkan display pixel index exceeds the signed shader ABI");
    TI_ERROR_IF(source_bytes > device.get_storage_buffer_binding_limits().max_range,
                "Vulkan display image exceeds the device storage-buffer range");
    TI_ERROR_IF(array->get_nelement() != width * height ||
                    array->get_element_size() != sizeof(std::uint32_t) ||
                    source.num_elements != width * height ||
                    device.get_vkbuffer_size(source.dev_alloc) < source_bytes,
                "Vulkan display source does not cover the full packed image");
    if (bytes) {
      TI_ERROR_IF(pool->width != source.shape[0] || pool->height != source.shape[1],
                  "Prepared display sources must have the same shape");
    } else {
      bytes = source_bytes;
      pool->width = source.shape[0];
      pool->height = source.shape[1];
    }
    owner.payload->sources.push_back(std::move(lease));
  }
  owner.payload->source_kind = owner.payload->sources.front()->handle().kind;
  owner.payload->source_count.store(owner.payload->sources.size());
  TI_ERROR_IF(capacity > std::numeric_limits<std::size_t>::max() / bytes,
              "Vulkan display pool byte extent overflows");
  owner.slots.resize(capacity);
  for (auto &slot : owner.slots)
    slot.readers.resize(graphics_capacity);
  owner.payload->buffers.reserve(capacity);
  owner.commands.reserve(capacity * sources.size());
  pool->registration = pool->domain->prepare_owned_instance(pool->owner, [&](const auto &) {
    for (std::size_t i = 0; i < capacity; ++i) {
      auto allocation = device.allocate_memory_unique(
          {bytes, false, false, false, AllocUsage::Storage});
      TI_ERROR_IF(allocation.second != RhiResult::success || !allocation.first,
                  "Unable to allocate Vulkan display buffer");
      owner.payload->buffers.push_back(std::move(allocation.first));
      for (const auto &source : sources) {
        auto commands = pool->domain->new_command_list();
        commands->buffer_copy(owner.payload->buffers.back()->get_ptr(),
                              source.dev_alloc.get_ptr(), bytes);
        owner.commands.push_back(pool->domain->prepare_commands(
            {commands.get()}, owner.payload));
      }
    }
  });
  std::vector<std::shared_ptr<DisplayStorage>> buffers;
  buffers.reserve(capacity);
  for (std::size_t i = 0; i < capacity; ++i)
    buffers.emplace_back(new VulkanDisplayBuffer(pool, i));
  return buffers;
}

VulkanDisplayBuffer::VulkanDisplayBuffer(std::shared_ptr<Pool> pool, std::size_t slot)
    : pool_(std::move(pool)), slot_(slot),
      identity_(next_display_buffer_identity.fetch_add(1)),
      pointer_(kDeviceNullPtr) {
  TI_ERROR_IF(!identity_, "Vulkan display identity exhausted");
  std::lock_guard<std::mutex> lock(pool_->owner->mutex);
  pool_->owner->check_open();
  pointer_ = pool_->owner->payload->buffers[slot_]->get_ptr();
}
VulkanDisplayBuffer::~VulkanDisplayBuffer() = default;
const storage::DenseStorageBuildResult &VulkanDisplayBuffer::description() const {
  return pool_->description;
}
std::uint64_t VulkanDisplayBuffer::identity() const noexcept { return identity_; }
DevicePtr VulkanDisplayBuffer::vulkan_ptr() const noexcept { return pointer_; }
int VulkanDisplayBuffer::width() const noexcept { return pool_->width; }
int VulkanDisplayBuffer::height() const noexcept { return pool_->height; }
bool VulkanDisplayBuffer::ready_for_vulkan_submit() const noexcept {
  std::lock_guard<std::mutex> lock(pool_->owner->mutex);
  return !pool_->owner->closed && pool_->domain->snapshot().accepting &&
         pool_->owner->slots[slot_].copy.valid();
}
bool VulkanDisplayBuffer::copy_ready() const {
  std::lock_guard<std::mutex> lock(pool_->owner->mutex);
  TI_ERROR_IF(!pool_->domain->snapshot().accepting,
              "Vulkan display domain is not accepting work");
  return pool_->owner->slot_ready(slot_);
}
void VulkanDisplayBuffer::prepare_write() {
  TI_ERROR_IF(!copy_ready(), "Vulkan display copy slot is busy");
}
void VulkanDisplayBuffer::finish_write() {
  TI_ERROR("Vulkan display buffers are copy-only; use prepare_frame_sources");
}
bool VulkanDisplayBuffer::try_finish_write(const RuntimeCompletion &producer) {
  TI_ERROR_IF(producer.valid(), "Vulkan display buffers do not accept writable producers");
  std::lock_guard<std::mutex> lock(pool_->owner->mutex);
  pool_->owner->check_open();
  TI_ERROR_IF(!pool_->domain->snapshot().accepting,
              "Vulkan display domain is not accepting work");
  return pool_->owner->slots[slot_].copy.valid();
}

Lease VulkanDisplayBuffer::source_lease(const FieldInfo &source) const {
  auto &owner = *pool_->owner;
  std::lock_guard<std::mutex> lock(owner.mutex);
  owner.check_open();
  TI_ERROR_IF(!source.valid || source.field_source != FieldSource::TaichiNDarray ||
                  source.runtime_resource_program != owner.program ||
                  source.dtype != PrimitiveType::u32 ||
                  source.shape.size() != 2 || source.shape[0] != width() ||
                  source.shape[1] != height(),
              "Vulkan display source does not match its prepared binding");
  for (std::size_t i = 0; i < owner.sources.size(); ++i) {
    const auto &prepared = owner.sources[i];
    if (source.runtime_resource_handle == prepared.runtime_resource_handle &&
        source.dev_alloc == prepared.dev_alloc &&
        source.num_elements == prepared.num_elements)
      return owner.payload->sources[i];
  }
  TI_ERROR("Vulkan display source was not cold-prepared; call prepare_frame_sources");
}
void VulkanDisplayBuffer::validate_producer(const RuntimeCompletion &producer) const {
  auto &owner = *pool_->owner;
  TI_ERROR_IF(!producer.valid(), "Vulkan display copy requires source_completion");
  // Same-owner peer domains are valid producers. This is not a writable-slot
  // producer-origin check, and never substitutes the selected domain's tail.
  producer.reusable_stream_dependency(Arch::vulkan, owner.fault->program_domain(),
                                      owner.fault);
}
bool VulkanDisplayBuffer::try_copy_image(DevicePtr source,
                                         const RuntimeCompletion &producer,
                                         Lease lease) {
  validate_producer(producer);
  auto &owner = *pool_->owner;
  std::lock_guard<std::mutex> lock(owner.mutex);
  owner.check_open();
  std::size_t index = owner.sources.size();
  for (std::size_t i = 0; i < owner.sources.size(); ++i) {
    if (lease && lease->handle() == owner.sources[i].runtime_resource_handle &&
        source == owner.sources[i].dev_alloc.get_ptr()) {
      index = i;
      break;
    }
  }
  TI_ERROR_IF(index == owner.sources.size(), "Vulkan display copy source is not prepared");
  if (!owner.slot_ready(slot_))
    return false;
  RuntimeCompletion copy;
  try {
    copy = pool_->domain->submit_instance(
        pool_->registration, owner.commands[slot_ * owner.sources.size() + index],
        {producer}, owner.payload);
  } catch (const Domain::Busy &) {
    return false;
  }
  // Only an accepted enqueue replaces this proof. Cancelling the pending
  // graphics frame never clears it or releases its source payload.
  owner.slots[slot_].copy = std::move(copy);
  return true;
}
RuntimeCompletion VulkanDisplayBuffer::source_completion() const {
  std::lock_guard<std::mutex> lock(pool_->owner->mutex);
  TI_ERROR_IF(pool_->owner->closed, "Vulkan display storage is closed");
  return pool_->owner->slots[slot_].copy;
}
bool VulkanDisplayBuffer::has_live_execution_domain() const {
  return !pool_->domain->instance_retired(pool_->registration);
}
void VulkanDisplayBuffer::close() { pool_->close(); }
StreamSemaphore VulkanDisplayBuffer::submit_vulkan_frame(
    taichi::lang::vulkan::VulkanStream &stream, CommandList *commands,
    const std::vector<StreamSemaphore> &additional_waits) {
  auto &owner = *pool_->owner;
  std::lock_guard<std::mutex> lock(owner.mutex);
  owner.check_open();
  TI_ERROR_IF(!pool_->domain->snapshot().accepting,
              "Vulkan display domain is not accepting work");
  auto &slot = owner.slots[slot_];
  TI_ERROR_IF(!slot.copy.valid(), "Vulkan display slot has no accepted copy");
  auto available = slot.readers.end();
  for (auto it = slot.readers.begin(); it != slot.readers.end(); ++it) {
    if (!*it || (*it)->is_ready()) {
      available = it;
      break;
    }
  }
  TI_ERROR_IF(available == slot.readers.end(), "Vulkan display reader capacity is full");
  auto dependency = slot.copy.reusable_stream_dependency(
      Arch::vulkan, owner.fault->program_domain(), owner.fault);
  TI_ERROR_IF(!dependency, "Vulkan display copy has no enqueued timeline point");
  auto waits = additional_waits;
  waits.push_back(std::move(dependency));
  // RHI tracking can still throw after enqueue. Without an exact returned
  // proof, quarantine this owner rather than freeing a possibly live buffer.
  try {
    auto complete = stream.submit(commands, waits);
    TI_ERROR_IF(!complete, "Vulkan display submission returned no completion");
    *available = complete;
    return complete;
  } catch (...) {
    owner.graphics_unsealed = true;
    throw;
  }
}

}  // namespace taichi::ui::vulkan
