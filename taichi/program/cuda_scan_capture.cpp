#include "taichi/program/cuda_scan_capture.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

#include <algorithm>

namespace taichi::lang {
namespace {

class CudaScanCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaScanCaptureCommand(Program *program,
                         const aot::Arg &values,
                         const aot::Arg &workspace,
                         int num_items,
                         int value_type)
      : program_(program),
        values_(values),
        workspace_(workspace),
        num_items_(num_items),
        value_type_(value_type),
        workspace_words_(std::max<std::size_t>(
            1,
            cuda_scan_capture_workspace_bytes(num_items, value_type) / 4)) {
    TI_ERROR_IF(
        program == nullptr || program->compile_config().arch != Arch::cuda,
        "CUDA scan recording requires a CUDA Program");
    TI_ERROR_IF(
        values.name == workspace.name || values.tag != aot::ArgKind::kNdarray ||
            workspace.tag != aot::ArgKind::kNdarray || values.field_dim != 1 ||
            workspace.field_dim != 1 || !values.element_shape.empty() ||
            !workspace.element_shape.empty() ||
            PrimitiveType::get(values.dtype_id) != dtype() ||
            PrimitiveType::get(workspace.dtype_id) != PrimitiveType::u32,
        "CUDA scan recording requires distinct scalar values and u32 scratch");
  }

  const char *kind() const override {
    return "driver_scan_retained_workspace";
  }
  Program *program() const override {
    return program_;
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    auto *values = array(values_, args, dtype(), num_items_, true);
    auto *workspace =
        array(workspace_, args, PrimitiveType::u32, workspace_words_);
    return &program == program_ && values && workspace &&
           values->get_device_allocation() !=
               workspace->get_device_allocation();
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    TI_ERROR_IF(!supports(args, program),
                "CUDA scan recording bindings are incompatible");
#if defined(TI_WITH_CUDA)
    auto guard = CUDAContext::get_instance().get_guard();
    cuda::driver_prepare_scan();
#endif
    // No user mathematics, allocation or arena mutation occurs in capture.
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(!supports(args, program),
                "CUDA scan capture bindings are incompatible");
#if defined(TI_WITH_CUDA)
    auto *values = array(values_, args, dtype(), num_items_, true);
    auto *workspace =
        array(workspace_, args, PrimitiveType::u32, workspace_words_);
    cuda::driver_inclusive_scan_with_workspace(
        reinterpret_cast<void *>(program.get_ndarray_data_ptr_as_int(values)),
        num_items_, static_cast<cuda::CudaTransformValueType>(value_type_),
        reinterpret_cast<void *>(
            program.get_ndarray_data_ptr_as_int(workspace)),
        stream);
#endif
  }

 private:
  DataType dtype() const {
    return value_type_ == 0 ? PrimitiveType::i32 : PrimitiveType::u32;
  }

  Ndarray *array(const aot::Arg &symbol,
                 const std::unordered_map<std::string, aot::IValue> &args,
                 DataType dtype,
                 std::size_t count,
                 bool prefix = false) const {
    const auto found = args.find(symbol.name);
    if (found == args.end() || found->second.tag != aot::ArgKind::kNdarray)
      return nullptr;
    auto *value = reinterpret_cast<Ndarray *>(found->second.val);
    return value && value->owning_program() == program_ &&
                   value->get_element_data_type() == dtype &&
                   value->get_element_shape().empty() &&
                   value->shape.size() == 1 &&
                   (prefix ? value->get_nelement() >= count
                           : value->get_nelement() == count)
               ? value
               : nullptr;
  }

  Program *program_;
  aot::Arg values_, workspace_;
  int num_items_, value_type_;
  std::size_t workspace_words_;
};

}  // namespace

std::size_t cuda_scan_capture_workspace_bytes(int num_items, int value_type) {
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "Retained Graph scan supports i32/u32 only");
#if defined(TI_WITH_CUDA)
  return cuda::driver_scan_workspace_bytes(
      num_items, static_cast<cuda::CudaTransformValueType>(value_type));
#else
  TI_ERROR("Retained Graph scan requires the CUDA backend");
#endif
}

std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_scan_capture_command(
    Program *program,
    const aot::Arg &values,
    const aot::Arg &workspace,
    int num_items,
    int value_type) {
  return std::make_shared<CudaScanCaptureCommand>(program, values, workspace,
                                                  num_items, value_type);
}

}  // namespace taichi::lang
