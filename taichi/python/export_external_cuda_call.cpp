#include "taichi/python/export_external_cuda_call.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "taichi/program/program.h"

namespace taichi {
namespace {

// Private shim thunk for an already prepared C-ABI provider launch. It owns
// neither a vendor context nor GPU storage and is not a CUDA Graph host node.
// The existing Program submission scope still owns resource pinning/order.
class ExternalCudaCall {
 public:
  using Launch = int (*)(void *, std::uint64_t);
  using LastError = std::size_t (*)(char *, std::size_t);

  ExternalCudaCall(lang::Program &program,
                   const lang::PreparedExternalCudaStorage &storage,
                   std::uintptr_t launch,
                   std::uintptr_t handle,
                   std::uintptr_t last_error)
      : program_(&program),
        storage_(storage),
        launch_(reinterpret_cast<Launch>(launch)),
        handle_(reinterpret_cast<void *>(handle)),
        last_error_(reinterpret_cast<LastError>(last_error)) {
    if (!storage_.storage || !launch_ || !handle_ || !last_error_) {
      throw std::invalid_argument(
          "Prepared CUDA call requires storage and C-ABI entries");
    }
  }

  void run() {
    // No GIL release: owner retirement and this pointer handoff are serialized
    // exactly as other Python-owned prepared commands. No Python is called by
    // the normal native submission path below.
    if (!launch_) {
      throw std::runtime_error("Prepared CUDA provider call has been retired");
    }
    program_->invoke_external_cuda_prepared(storage_, [&] {
      const int result = launch_(handle_, 0);
      if (result != 0) {
        const auto required = last_error_(nullptr, 0);
        std::string message = "Prepared CUDA provider call failed (" +
                              std::to_string(result) + ")";
        if (required > 1) {
          std::vector<char> error(required, 0);
          last_error_(error.data(), error.size());
          error.back() = 0;
          message += ": " + std::string(error.data());
        }
        // Throw inside the existing scope so partially submitted failures are
        // recorded and retained by the same lifecycle mechanism as callbacks.
        throw std::runtime_error(message);
      }
    });
  }

  void invalidate() {
    launch_ = nullptr;
    handle_ = nullptr;
    last_error_ = nullptr;
    storage_ = {};
  }

 private:
  lang::Program *program_;
  lang::PreparedExternalCudaStorage storage_;
  Launch launch_;
  void *handle_;
  LastError last_error_;
};

}  // namespace

void export_external_cuda_call(py::module &m) {
  py::class_<ExternalCudaCall, std::shared_ptr<ExternalCudaCall>>(
      m, "_PreparedExternalCudaCall")
      .def("__call__", &ExternalCudaCall::run)
      .def("invalidate", &ExternalCudaCall::invalidate);
  m.def(
      "_bind_external_cuda_call",
      [](lang::Program &program,
         const lang::PreparedExternalCudaStorage &storage,
         std::uintptr_t launch,
         std::uintptr_t handle,
         std::uintptr_t last_error,
         py::object library_owner) {
        return std::make_shared<ExternalCudaCall>(
            program, storage, launch, handle, last_error);
      },
      py::keep_alive<0, 1>(), py::keep_alive<0, 6>());
}

}  // namespace taichi
