#include "taichi/program/cuda_addon_capture.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace taichi::lang {
namespace {

class CudaAddonCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaAddonCaptureCommand(Program *program,
                         std::uint64_t invoke_address,
                         const std::string &payload,
                         std::size_t stream_offset,
                         const std::vector<aot::Arg> &arguments,
                         const std::vector<std::size_t> &pointer_offsets,
                         const std::vector<std::size_t> &scalar_counts,
                         const std::vector<bool> &writable,
                         std::uint64_t error_address)
      : program_(program),
        invoke_(reinterpret_cast<Invoke>(invoke_address)),
        error_(reinterpret_cast<Error>(error_address)),
        payload_((payload.size() + sizeof(std::max_align_t) - 1) /
                 sizeof(std::max_align_t)),
        stream_offset_(stream_offset),
        arguments_(arguments),
        pointer_offsets_(pointer_offsets),
        scalar_counts_(scalar_counts),
        writable_(writable) {
    TI_ERROR_IF(program == nullptr ||
                    program->compile_config().arch != Arch::cuda || !invoke_,
                "CUDA addon capture requires a live CUDA Program and callback");
    TI_ERROR_IF(arguments.empty() || pointer_offsets.size() != arguments.size() ||
                    scalar_counts.size() != arguments.size() ||
                    writable.size() != arguments.size(),
                "CUDA addon capture binding metadata is inconsistent");
    auto offsets = pointer_offsets;
    offsets.push_back(stream_offset);
    std::sort(offsets.begin(), offsets.end());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      TI_ERROR_IF(offsets[i] > payload.size() ||
                      payload.size() - offsets[i] < sizeof(void *) ||
                      (i && offsets[i] - offsets[i - 1] < sizeof(void *)),
                  "CUDA addon capture relocations overlap or exceed the payload");
    }
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      TI_ERROR_IF(arguments[i].tag != aot::ArgKind::kNdarray ||
                      !arguments[i].element_shape.empty() ||
                      arguments[i].field_dim == 0 || scalar_counts[i] == 0,
                  "CUDA addon capture requires nonempty scalar ndarray bindings");
    }
    std::memcpy(payload_.data(), payload.data(), payload.size());
  }

  const char *kind() const override {
    return "source_addon_c_abi";
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
      auto *current = array(i, args);
      if (!current) {
        return false;
      }
      for (std::size_t j = 0; j < i; ++j) {
        if ((writable_[i] || writable_[j]) &&
            current->get_device_allocation() ==
                array(j, args)->get_device_allocation()) {
          return false;
        }
      }
    }
    return true;
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    // The addon has already queried workspace and retained its resources.
    // Never execute user mathematics to prepare a capture (feedback safety).
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(!supports(args, program),
                "CUDA addon capture bindings are stale, incompatible or alias writable storage");
    // A command can be shared by multiple materializations. Its template is
    // immutable; the aligned local copy exists only at the capture boundary.
    auto payload = payload_;
    auto *bytes = reinterpret_cast<unsigned char *>(payload.data());
    for (std::size_t i = 0; i < arguments_.size(); ++i) {
      auto pointer = reinterpret_cast<void *>(
          program.get_ndarray_data_ptr_as_int(array(i, args)));
      std::memcpy(bytes + pointer_offsets_[i], &pointer, sizeof(pointer));
    }
    std::memcpy(bytes + stream_offset_, &stream, sizeof(stream));
    const auto status = invoke_(bytes);
    if (status) {
      char message[512]{};
      if (error_) {
        error_(message, sizeof(message));
        message[sizeof(message) - 1] = '\0';
      }
      TI_ERROR("CUDA addon capture callback failed ({}): {}", status, message);
    }
  }

 private:
  using Invoke = std::uint32_t (*)(const void *);
  using Error = std::size_t (*)(char *, std::size_t);

  Ndarray *array(std::size_t index,
                 const std::unordered_map<std::string, aot::IValue> &args) const {
    const auto &symbol = arguments_[index];
    const auto found = args.find(symbol.name);
    if (found == args.end() || found->second.tag != aot::ArgKind::kNdarray) {
      return nullptr;
    }
    auto *value = reinterpret_cast<Ndarray *>(found->second.val);
    if (!value || value->owning_program() != program_ ||
        value->get_element_data_type() != PrimitiveType::get(symbol.dtype_id) ||
        !value->get_element_shape().empty() ||
        value->shape.size() != symbol.field_dim ||
        value->get_nelement() != scalar_counts_[index]) {
      return nullptr;
    }
    return value;
  }

  Program *program_;
  Invoke invoke_;
  Error error_;
  std::vector<std::max_align_t> payload_;
  std::size_t stream_offset_;
  std::vector<aot::Arg> arguments_;
  std::vector<std::size_t> pointer_offsets_;
  std::vector<std::size_t> scalar_counts_;
  std::vector<bool> writable_;
};

}  // namespace

std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_addon_capture_command(
    Program *program,
    std::uint64_t invoke_address,
    const std::string &payload,
    std::size_t stream_offset,
    const std::vector<aot::Arg> &arguments,
    const std::vector<std::size_t> &pointer_offsets,
    const std::vector<std::size_t> &scalar_counts,
    const std::vector<bool> &writable,
    std::uint64_t error_address) {
  return std::make_shared<CudaAddonCaptureCommand>(
      program, invoke_address, payload, stream_offset, arguments, pointer_offsets,
      scalar_counts, writable, error_address);
}

}  // namespace taichi::lang
