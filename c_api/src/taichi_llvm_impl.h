#pragma once
#ifdef TI_WITH_LLVM

#include "taichi_core_impl.h"

#ifdef TI_WITH_CUDA
#include "taichi/platform/cuda/detect_cuda.h"
#endif

namespace taichi::lang {
class LlvmRuntimeExecutor;
struct CompileConfig;
struct NativePreparationContext;
}  // namespace taichi::lang

namespace capi {

struct DenseSNodeGroup;

class LlvmRuntime : public Runtime {
 public:
  LlvmRuntime(taichi::Arch arch);
  LlvmRuntime(taichi::lang::LlvmRuntimeExecutor &executor,
              taichi::uint64 *result_buffer);
  virtual ~LlvmRuntime();

  void check_runtime_error();
  taichi::lang::Device &get() override;
  void prepare_cpu_scheduler(taichi::lang::NativePreparationContext &context);
  std::pair<const taichi::lang::StructType *, std::size_t>
  resolve_argpack_type(const taichi::lang::StructType *type,
                       const std::string &layout);

  TiMemory create_dense_snode_group(const std::string &schema,
                                    const std::string &dimensions);
  const std::string &dense_snode_group_layout(TiMemory group);
  TiNdArray dense_snode_group_member(TiMemory group,
                                      const std::string &paths,
                                      TiDataType dtype,
                                      const TiNdShape &element_shape);

 private:
  /* Internally used interfaces */
  TiAotModule load_aot_module(const char *module_path) override;
  TiMemory allocate_memory(
      const taichi::lang::Device::AllocParams &params) override;
  void free_memory(TiMemory devmem) override;

  void buffer_copy(const taichi::lang::DevicePtr &dst,
                   const taichi::lang::DevicePtr &src,
                   size_t size) override;

  void flush() override;

  void wait() override;

 private:
  std::unique_ptr<taichi::lang::CompileConfig> cfg_{nullptr};
  std::unique_ptr<taichi::lang::LlvmRuntimeExecutor> owned_executor_;
  taichi::lang::LlvmRuntimeExecutor *executor_{nullptr};
  taichi::uint64 *result_buffer{nullptr};
  std::unordered_map<TiMemory, std::shared_ptr<DenseSNodeGroup>> dense_groups_;
  void retire_dense_groups();
  void free_dense_group(TiMemory group);
};

}  // namespace capi

#endif  // TI_WITH_LLVM
