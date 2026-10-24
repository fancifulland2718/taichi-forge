#include "taichi/runtime/gfx/range_dispatch.h"

#include "taichi/codegen/spirv/spirv_ir_builder.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang::gfx {
namespace {
std::vector<uint32_t> controller_spirv(
    const DeviceCapabilityConfig &caps,
    const RangeForBounds &bounds,
    uint32_t block,
    uint32_t grid_cap) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto i32 = builder.i32_type();
  const auto u32 = builder.u32_type();
  const auto temporary = builder.buffer_argument(i32, 0, 0, "range_snapshot");
  const auto packet = builder.buffer_argument(u32, 0, 1, "range_packet");
  const auto function = builder.new_function();
  builder.start_function(function);
  const auto literal = [&](uint32_t value) {
    return builder.uint_immediate_number(u32, value);
  };
  const auto read = [&](const RangeBound &bound) {
    if (bound.kind == RangeBoundKind::constant_i32)
      return builder.int_immediate_number(i32, bound.value);
    return builder.load_variable(
        builder.struct_array_access(i32, temporary, literal(bound.value / 4)),
        i32);
  };
  const auto begin = read(bounds.begin);
  const auto end = read(bounds.end);
  const auto zero = literal(0);
  const auto one = literal(1);
  // Signed comparison followed by unsigned subtraction represents the full
  // positive difference of two i32 bounds, without requiring shaderInt64.
  const auto extent = builder.select(
      builder.gt(end, begin),
      builder.sub(builder.cast(u32, end), builder.cast(u32, begin)), zero);
  const auto divisor = literal(block);
  // Avoid overflow in the usual (extent + block - 1) expression.
  const auto quotient = builder.div(extent, divisor);
  const auto remainder = builder.sub(extent, builder.mul(quotient, divisor));
  const auto ceiling = builder.add(
      quotient, builder.select(builder.gt(remainder, zero), one, zero));
  const auto cap = literal(grid_cap);
  const auto positive = builder.select(builder.eq(ceiling, zero), one, ceiling);
  const auto groups = builder.select(builder.gt(positive, cap), cap, positive);
  for (uint32_t index = 0; index != 3; ++index)
    builder.store_variable(
        builder.struct_array_access(u32, packet, literal(index)),
        index == 0 ? groups : one);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300)
    interfaces = {temporary, packet};
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}
}  // namespace

void RangeDispatch::validate(const spirv::TaskAttributes &task,
                             std::size_t global_tmp_bytes) {
  TI_ERROR_IF(task.device_range_dispatch_version != 1 ||
                  !task.valid_device_range_dispatch_contract(),
              "Task '{}' has no qualified device range dispatch contract",
              task.name);
  for (const auto &bound : {task.range_for_attribs->begin,
                            task.range_for_attribs->end}) {
    TI_ERROR_IF(bound.kind == RangeBoundKind::unknown ||
                    (bound.kind == RangeBoundKind::global_tmp_i32 &&
                     (!bound.fits_global_tmps(global_tmp_bytes) ||
                      uint64_t(bound.value) / 4 > UINT32_MAX)),
                "Task '{}' range snapshot exceeds its resolved workspace",
                task.name);
  }
}

RangeDispatch::RangeDispatch(Device &device,
                             const spirv::TaskAttributes &task,
                             DeviceAllocation global_tmps,
                             std::size_t global_tmp_bytes)
    : global_tmps_(global_tmps), global_tmp_bytes_(global_tmp_bytes) {
  validate(task, global_tmp_bytes);
  TI_ERROR_IF(global_tmps.device != &device || !global_tmps.alloc_id,
              "Device range dispatch requires the resolved task workspace");
  const auto block = uint32_t(task.advisory_num_threads_per_group);
  auto groups = (uint64_t(task.advisory_total_num_threads) + block - 1) / block;
  TI_ERROR_IF(device.arch() != Arch::vulkan,
              "Device range controller requires Vulkan");
#ifdef TI_WITH_VULKAN
  // Indirect dispatch has no CPU-side count validation. Grid-striding can
  // cover the same logical range within the actual device's dispatch limit.
  groups = std::min<uint64_t>(groups,
      static_cast<vulkan::VulkanDevice &>(device)
          .get_vk_physical_device_props().limits.maxComputeWorkGroupCount[0]);
#endif
  TI_ERROR_IF(groups == 0, "Device range controller has no physical grid capacity");
  auto [packet, allocation_status] = device.allocate_memory_unique(
      {kPacketBytes, false, false, false, AllocUsage::Storage | AllocUsage::Indirect});
  TI_ERROR_IF(allocation_status != RhiResult::success,
              "Device range packet allocation failed");
  packet_ = std::move(packet);
  const auto code = controller_spirv(device.get_caps(), *task.range_for_attribs,
                                      block, uint32_t(groups));
  PipelineSourceDesc source{PipelineSourceType::spirv_binary, code.data(),
                            code.size() * sizeof(uint32_t),
                            PipelineStageType::compute};
  auto [pipeline, status] =
      device.create_pipeline_unique(source, "task_range_controller_v1");
  TI_ERROR_IF(status != RhiResult::success || !pipeline,
              "Device range controller pipeline creation failed");
  pipeline_ = std::move(pipeline);
  resources_ = device.create_resource_set_unique();
  resources_->rw_buffer(0, global_tmps_);
  resources_->rw_buffer(1, *packet_);
  TI_ERROR_IF(resources_->prepare_for_replay(false) != RhiResult::success,
              "Device range controller resource preparation failed");
}

void RangeDispatch::record_controller(CommandList &commands) const {
  commands.buffer_transition(
      global_tmps_.get_ptr(), global_tmp_bytes_,
      {BufferBarrierStage::Host | BufferBarrierStage::Transfer |
           BufferBarrierStage::Compute,
       BufferBarrierAccess::HostWrite | BufferBarrierAccess::TransferWrite |
           BufferBarrierAccess::ShaderWrite,
       BufferBarrierStage::Compute, BufferBarrierAccess::ShaderRead});
  commands.bind_pipeline(pipeline_.get());
  TI_ERROR_IF(commands.bind_shader_resources(resources_.get()) !=
                      RhiResult::success ||
                  commands.dispatch(1) != RhiResult::success,
              "Device range controller recording failed");
  // Protect subsequent reuse of the same snapshot location, including a later
  // serial task in this kernel. Packet write -> indirect read belongs to the
  // shared record_graph_dispatch routine, after the payload is rebound.
  commands.buffer_transition(
      global_tmps_.get_ptr(), global_tmp_bytes_,
      {BufferBarrierStage::Compute, BufferBarrierAccess::ShaderRead,
       BufferBarrierStage::Compute,
       BufferBarrierAccess::ShaderRead | BufferBarrierAccess::ShaderWrite});
}

}  // namespace taichi::lang::gfx
