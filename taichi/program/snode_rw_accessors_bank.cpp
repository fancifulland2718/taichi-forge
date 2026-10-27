#include "taichi/program/snode_rw_accessors_bank.h"

#include "taichi/program/program.h"

namespace taichi::lang {

namespace {
void set_kernel_args(const std::vector<int> &I,
                     int num_active_indices,
                     LaunchContextBuilder *launch_ctx) {
  for (int i = 0; i < num_active_indices; i++) {
    launch_ctx->set_arg_int({i}, I[i]);
  }
}
}  // namespace

SNodeRwAccessorsBank::Accessors SNodeRwAccessorsBank::get(SNode *snode) {
  return Accessors(snode, this);
}

Kernel *SNodeRwAccessorsBank::get_kernel(const SNode *snode, bool write) {
  auto &kernels = snode_to_kernels_[snode];
  auto &kernel = write ? kernels.writer : kernels.reader;
  if (kernel == nullptr) {
    auto *mutable_snode = const_cast<SNode *>(snode);
    kernel = write ? &program_->get_snode_writer(mutable_snode)
                   : &program_->get_snode_reader(mutable_snode);
  }
  return kernel;
}

SNodeRwAccessorsBank::Accessors::Accessors(const SNode *snode,
                                         SNodeRwAccessorsBank *bank)
    : snode_(snode), bank_(bank), prog_(bank->program_) {
}
void SNodeRwAccessorsBank::Accessors::write_float(const std::vector<int> &I,
                                                  float64 val) {
  auto *writer = bank_->get_kernel(snode_, /*write=*/true);
  auto launch_ctx = writer->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_float({snode_->num_active_indices}, val);
  prog_->synchronize();
  const auto &compiled_kernel_data = prog_->compile_kernel_execution_handle(
      prog_->compile_config(), prog_->get_device_caps(), *writer);
  prog_->launch_kernel(*compiled_kernel_data, launch_ctx);
}

float64 SNodeRwAccessorsBank::Accessors::read_float(const std::vector<int> &I) {
  prog_->synchronize();
  auto *reader = bank_->get_kernel(snode_, /*write=*/false);
  auto launch_ctx = reader->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  const auto &compiled_kernel_data = prog_->compile_kernel_execution_handle(
      prog_->compile_config(), prog_->get_device_caps(), *reader);
  prog_->launch_kernel(*compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_float({0});
}

// for int32 and int64
void SNodeRwAccessorsBank::Accessors::write_int(const std::vector<int> &I,
                                                int64 val) {
  auto *writer = bank_->get_kernel(snode_, /*write=*/true);
  auto launch_ctx = writer->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_int({snode_->num_active_indices}, val);
  prog_->synchronize();
  const auto &compiled_kernel_data = prog_->compile_kernel_execution_handle(
      prog_->compile_config(), prog_->get_device_caps(), *writer);
  prog_->launch_kernel(*compiled_kernel_data, launch_ctx);
}

// for int32 and int64
void SNodeRwAccessorsBank::Accessors::write_uint(const std::vector<int> &I,
                                                 uint64 val) {
  auto *writer = bank_->get_kernel(snode_, /*write=*/true);
  auto launch_ctx = writer->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_uint({snode_->num_active_indices}, val);
  prog_->synchronize();
  const auto &compiled_kernel_data = prog_->compile_kernel_execution_handle(
      prog_->compile_config(), prog_->get_device_caps(), *writer);
  prog_->launch_kernel(*compiled_kernel_data, launch_ctx);
}

int64 SNodeRwAccessorsBank::Accessors::read_int(const std::vector<int> &I) {
  prog_->synchronize();
  auto *reader = bank_->get_kernel(snode_, /*write=*/false);
  auto launch_ctx = reader->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  const auto &compiled_kernel_data = prog_->compile_kernel_execution_handle(
      prog_->compile_config(), prog_->get_device_caps(), *reader);
  prog_->launch_kernel(*compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_int({0});
}

uint64 SNodeRwAccessorsBank::Accessors::read_uint(const std::vector<int> &I) {
  prog_->synchronize();
  auto *reader = bank_->get_kernel(snode_, /*write=*/false);
  auto launch_ctx = reader->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  const auto &compiled_kernel_data = prog_->compile_kernel_execution_handle(
      prog_->compile_config(), prog_->get_device_caps(), *reader);
  prog_->launch_kernel(*compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_uint({0});
}

}  // namespace taichi::lang
