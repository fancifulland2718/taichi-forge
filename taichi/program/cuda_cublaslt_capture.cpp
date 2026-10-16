#include "taichi/program/cuda_cublaslt_capture.h"
#include "taichi/external_runtime/forge_cublaslt_provider.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include <algorithm>

namespace taichi::lang {
namespace {

class CudaCublasLtCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaCublasLtCaptureCommand(Program *program,
                             const CudaCublasLtCapturePlan &plan,
                             const std::vector<aot::Arg> &arguments)
      : program_(program), plan_(plan), arguments_(arguments) {
    TI_ERROR_IF(
        program == nullptr || program->compile_config().arch != Arch::cuda,
        "cuBLASLt capture requires a CUDA Program");
    TI_ERROR_IF(!plan.execute_address || !plan.error_address ||
                    !plan.plan_handle || plan.shapes.size() != 3 ||
                    arguments.size() != (plan.workspace_bytes ? 4 : 3),
                "cuBLASLt capture plan is incomplete");
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto &arg = arguments[i];
      TI_ERROR_IF(
          arg.tag != aot::ArgKind::kNdarray || !arg.element_shape.empty() ||
              PrimitiveType::get(arg.dtype_id) !=
                  (i < 3 ? PrimitiveType::f32 : PrimitiveType::u8),
          "cuBLASLt capture requires scalar f32 matrices and byte workspace");
      if (i < 3) {
        TI_ERROR_IF(plan.shapes[i].size() < 2 || plan.shapes[i].size() > 3 ||
                        arg.field_dim != plan.shapes[i].size(),
                    "cuBLASLt capture matrix rank is incompatible");
        for (auto extent : plan.shapes[i]) {
          TI_ERROR_IF(extent <= 0, "cuBLASLt capture matrix extent is invalid");
        }
      } else {
        TI_ERROR_IF(arg.field_dim != 1,
                    "cuBLASLt workspace must have rank one");
      }
      for (std::size_t j = 0; j < i; ++j) {
        TI_ERROR_IF(arg.name == arguments[j].name,
                    "cuBLASLt capture binding names must be distinct");
      }
    }
  }

  const char *kind() const override {
    return "cublaslt_retained_matmul_f32";
  }

  bool supports_binding_frames() const override {
    // Immutable frame capture changes only operand addresses. The Python
    // recording lease pins the fixed descriptors, algorithm and workspace
    // until its owning Graph closes the native executor.
    return true;
  }

  Program *program() const override {
    return program_;
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (&program != program_) {
      return false;
    }
    for (std::size_t i = 0; i < arguments_.size(); ++i) {
      auto *value = array(i, args);
      if (!value) {
        return false;
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (i >= 2 && value->get_device_allocation() ==
                          array(j, args)->get_device_allocation()) {
          return false;
        }
      }
    }
    return true;
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    TI_ERROR_IF(!supports(args, program),
                "cuBLASLt capture bindings are incompatible");
    // The provider already selected the algorithm and allocated workspace.
    // Do not execute beta*C feedback merely to prepare a capture.
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(!supports(args, program),
                "cuBLASLt capture bindings are incompatible");
    auto pointer = [&](std::size_t i) -> std::uint64_t {
      return program.get_ndarray_data_ptr_as_int(array(i, args));
    };
    TiForgeCublasLtExecDesc execution{sizeof(execution),
                                      0,
                                      pointer(0),
                                      pointer(1),
                                      pointer(2),
                                      plan_.workspace_bytes ? pointer(3) : 0,
                                      plan_.workspace_bytes,
                                      reinterpret_cast<std::uint64_t>(stream)};
    const auto invoke =
        reinterpret_cast<TiForgeCublasLtExecuteFn>(plan_.execute_address);
    const auto status =
        invoke(reinterpret_cast<void *>(plan_.plan_handle), &execution);
    if (status != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
      const auto error = reinterpret_cast<TiForgeRuntimeProviderGetLastErrorFn>(
          plan_.error_address);
      std::vector<char> message(std::max(error(nullptr, 0), std::size_t(1)),
                                '\0');
      error(message.data(), message.size());
      message.back() = '\0';
      TI_ERROR("cuBLASLt Graph capture failed with status {}: {}",
               static_cast<int>(status), message.data());
    }
  }

 private:
  Ndarray *array(
      std::size_t index,
      const std::unordered_map<std::string, aot::IValue> &args) const {
    const auto found = args.find(arguments_[index].name);
    if (found == args.end() || found->second.tag != aot::ArgKind::kNdarray) {
      return nullptr;
    }
    auto *value = reinterpret_cast<Ndarray *>(found->second.val);
    if (!value || value->owning_program() != program_ ||
        !value->get_element_shape().empty() ||
        value->get_element_data_type() !=
            (index < 3 ? PrimitiveType::f32 : PrimitiveType::u8)) {
      return nullptr;
    }
    if (index < 3) {
      return value->shape == plan_.shapes[index] ? value : nullptr;
    }
    return value->shape.size() == 1 &&
                   value->get_nelement() == plan_.workspace_bytes
               ? value
               : nullptr;
  }

  Program *program_;
  const CudaCublasLtCapturePlan plan_;
  const std::vector<aot::Arg> arguments_;
};

}  // namespace

std::shared_ptr<aot::CudaGraphCaptureCommand>
make_cuda_cublaslt_capture_command(Program *program,
                                   const CudaCublasLtCapturePlan &plan,
                                   const std::vector<aot::Arg> &arguments) {
  return std::make_shared<CudaCublasLtCaptureCommand>(program, plan, arguments);
}

}  // namespace taichi::lang
