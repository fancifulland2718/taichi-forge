#pragma once
// =============================================================================
// Phase 2a — pointer-on-Vulkan device-side node allocator (SKELETON, NOT WIRED)
// =============================================================================
//
// 本文件是 SNode_Vulkan_规划.md §3 Phase 2a 的低耦合接口骨架：
//
//   - 只有声明 / 内联 POD，不实现核心方法；
//   - 不进入 CMake 任何 target（[taichi/runtime/gfx/CMakeLists.txt] 不引用）；
//   - 不被 codegen / runtime 任何路径引用；
//   - 默认 vanilla 行为完全不受影响。
//
// 设计目的：先把「allocator 接口」与「codegen 如何调用 allocator」解耦定下，
// 这样 2a 选定的 bump-only 实现与未来可能的 freelist/hybrid 实现可以互换，
// codegen 与 SNodeTree 元数据零修改。
//
// 路线选择依据：tests/p4/phase2a_workload_analysis.py 与 phase2a_workload_mpm.py
//   - 实时渲染 (taichi_sparse logo) 与物理仿真 (MPM) 两类典型负载下，
//     帧间 cell 集合重合率 97-99%，「整池清零 + 重 alloc」是热路径；
//   - 个体 recycle 的潜在收益 < 1% 帧时间，不值得引入 lock-free freelist 复杂度。
//
// 约束（来自 .github/instructions/optimization-workflow.instructions.md）：
//   - 不破坏既有 BufferType 枚举语义（NodeAllocator 是新增项，旧路径不感知）；
//   - 默认 ti.init(vulkan_pointer_experimental=False) 时本接口不被构造；
//   - CMake `TI_WITH_VULKAN_POINTER` 默认 OFF，本文件根本不参与编译；
//   - 任何子阶段失败可独立 git revert（2a 全部代码集中在 snode_allocator.{h,cpp}
//     和 runtime.{h,cpp} 的少量 hook 里）。
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>

#include "taichi/rhi/device.h"

namespace taichi::lang {

class CommandList;  // taichi/rhi/public_device.h

namespace gfx {

// -----------------------------------------------------------------------------
// SpirvAllocatorContract —— codegen 端唯一可见的 allocator 视图
// -----------------------------------------------------------------------------
//
// codegen 通过这个 POD 拿到「在 root_buffer 内 emit 对该 SNode allocator 的
// SPIR-V atomic 调用所需要的全部信息」。换实现（bump → freelist）时，
// 这个 struct 的字段含义可能扩展（例如加 freelist_head_offset），
// codegen 仍只发射「一处 atomic 调用」，路径不动。
//
// 当前 2a (bump-only) 语义：
//   activate 槽 = OpAtomicIAdd(root_buf[watermark_offset_in_root], 1)
//                 if (slot >= pool_capacity) → 抛 runtime error 路径
//                 else → return pool_data_offset_in_root + slot * cell_stride_bytes
//   clear_all  = host 侧 vkCmdFillBuffer 把 [watermark_offset_in_root, +4) 置 0
//                                       + [pool_data_offset_in_root, +pool_capacity*cell_stride_bytes) 置 0
//
struct SpirvAllocatorContract {
  // u32 atomic 计数器在 root_buffer 内的字节偏移（SPIR-V binding=0 的 buffer）
  uint32_t watermark_offset_in_root{0};
  // pool 数据起始字节偏移（cell payload 数组的开头）
  uint32_t pool_data_offset_in_root{0};
  // 池容量（cell 数）—— 编译期常量，等于该 SNode 的 num_cells_per_container
  // worst-case 池占用 = 100%（见 workload 分析）
  uint32_t pool_capacity{0};
  // 单 cell payload 字节数（含子 SNode container 全部内容）
  uint32_t cell_stride_bytes{0};
  // 该 allocator 服务的 SNode id，仅供调试 / runtime error 信息使用
  int32_t snode_id{-1};
};

// -----------------------------------------------------------------------------
// DeviceNodeAllocator —— 抽象基类
// -----------------------------------------------------------------------------
//
// runtime 持有 `std::vector<std::unique_ptr<DeviceNodeAllocator>>`，
// 每棵 SNodeTree 内每一个需要 allocator 的 SNode（pointer / 未来 dynamic / hash）
// 拥有一个实例。
//
class DeviceNodeAllocator {
 public:
  virtual ~DeviceNodeAllocator() = default;

  // host 侧池 buffer（可能是独立 buffer，也可能是 root_buffer 的子区间——
  // 当前 2a 实现选「子区间」让 codegen 端只看 root_buffer，最简）
  virtual DeviceAllocation pool_buffer() const = 0;

  // 池容量（cell 数）
  virtual std::size_t pool_capacity() const = 0;

  // 单 cell payload 字节数
  virtual std::size_t cell_payload_bytes() const = 0;

  // 整池清零：被 SNodeTree::deactivate_all / GfxRuntime::deactivate_all_snodes
  // 调用。实现录制一条 GPU 命令把 watermark + data 区清零。
  // 注意：调用方负责前后 memory barrier；本接口只发射一条命令。
  virtual void clear_all(CommandList *cmd) = 0;

  // codegen 端用：返回发射 SPIR-V allocate/lookup 所需的 contract
  virtual SpirvAllocatorContract spirv_contract() const = 0;
};

// -----------------------------------------------------------------------------
// BumpOnlyDeviceNodeAllocator —— Phase 2a 唯一实现
// -----------------------------------------------------------------------------
//
// 实现策略：
//   - 池 = 静态分配 num_cells × cell_stride_bytes 字节（编译期已知 num_cells）
//   - watermark = u32，初始 0；activate 时 atomic add 1，越界抛 RuntimeError
//   - clear_all = vkCmdFillBuffer (整池 + watermark)
//   - 个体 deactivate(idx)：仅由 codegen 在 cell layout 内置 lock=0、data_ptr=0；
//     **池 slot 不回收**，等下一次 clear_all 整池重置（leak 容忍度参见 workload）
//
// 本类构造严格在 `ti.init(vulkan_pointer_experimental=True)` 时由
// GfxRuntime::init_node_allocator_buffers() 创建；默认 OFF 时不构造。
//
class BumpOnlyDeviceNodeAllocator final : public DeviceNodeAllocator {
 public:
  struct Params {
    Device *device{nullptr};
    int32_t snode_id{-1};
    // 「池容量」必须等于 SNode 的 num_cells_per_container（worst case 100%）
    std::size_t pool_capacity{0};
    std::size_t cell_payload_bytes{0};
    // 池在 root_buffer 内的起始偏移（host 侧由 SNodeTree struct compiler 填）
    // —— 把池放在 root_buffer 子区间以保持 codegen 端只看 root_buffer
    uint32_t watermark_offset_in_root{0};
    uint32_t pool_data_offset_in_root{0};
    DeviceAllocation root_buffer_alloc;
  };

  explicit BumpOnlyDeviceNodeAllocator(const Params &p);
  ~BumpOnlyDeviceNodeAllocator() override;

  DeviceAllocation pool_buffer() const override;
  std::size_t pool_capacity() const override { return params_.pool_capacity; }
  std::size_t cell_payload_bytes() const override {
    return params_.cell_payload_bytes;
  }
  void clear_all(CommandList *cmd) override;
  SpirvAllocatorContract spirv_contract() const override;

 private:
  Params params_;
};

// 工厂：路线切换的唯一入口（未来 freelist 实现也走这里）
//
// 当前实现只支持 bump-only；后续若加 freelist，只在此处加一个 enum 分支。
std::unique_ptr<DeviceNodeAllocator> create_device_node_allocator(
    const BumpOnlyDeviceNodeAllocator::Params &params);

}  // namespace gfx
}  // namespace taichi::lang
