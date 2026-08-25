#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace taichi::lang::spirv {

struct SpvStats {
  std::string kernel_name;
  int task_id{-1};
  std::string task_name;
  std::string task_type;
  int snode_id{-1};
  bool listgen_related{false};
  bool pointer_related{false};
  bool sparse_related{false};
  bool opt_run{false};
  bool opt_ok{true};
  std::size_t before_words{0};
  std::size_t after_words{0};
  std::size_t ray_query_initialize_before{0};
  std::size_t ray_query_initialize_after{0};
  std::size_t ray_query_getter_before{0};
  std::size_t ray_query_getter_after{0};
  std::size_t function_variable_before{0};
  std::size_t function_variable_after{0};
  std::size_t phi_before{0};
  std::size_t phi_after{0};
  double opt_us{0.0};
  std::string skipped_passes;
};

std::vector<SpvStats> get_last_spv_stats();
void clear_last_spv_stats();

}  // namespace taichi::lang::spirv
