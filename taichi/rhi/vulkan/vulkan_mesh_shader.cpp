#include "taichi/rhi/vulkan/vulkan_mesh_shader.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "external/SPIRV-Headers/include/spirv/unified1/spirv.hpp"

namespace taichi::lang::vulkan {

void validate_mesh_shader_limits(const void *code,
                                 size_t size,
                                 uint32_t entry_id,
                                 VkShaderStageFlagBits stage,
                                 const VulkanCapabilities &caps) {
  if (stage != VK_SHADER_STAGE_MESH_BIT_EXT &&
      stage != VK_SHADER_STAGE_TASK_BIT_EXT)
    return;
  const bool task = stage == VK_SHADER_STAGE_TASK_BIT_EXT;
  const std::string prefix = task ? "Task shader " : "Mesh shader ";
  if (!caps.mesh_shader || (task && !caps.task_shader))
    throw std::invalid_argument(
        prefix + "requires an enabled VK_EXT_mesh_shader feature");

  if (!code || size < 5 * sizeof(uint32_t) || size % sizeof(uint32_t))
    throw std::invalid_argument(prefix + "requires complete SPIR-V words");
  // Source buffers need not be aligned. Do not allocate from the module's ID
  // bound: the small maps below grow only with actual constant declarations.
  const auto word = [&](size_t index) {
    uint32_t value;
    std::memcpy(&value, static_cast<const char *>(code) + index * 4, 4);
    return value;
  };
  if (word(0) != spv::MagicNumber)
    throw std::invalid_argument(prefix + "requires SPIR-V magic");
  using Size = std::array<uint32_t, 3>;
  std::optional<Size> local_size, local_size_ids;
  std::optional<uint32_t> workgroup_size_id, vertices, primitives;
  std::unordered_set<uint32_t> int32_types;
  std::unordered_map<uint32_t, uint32_t> constants;
  std::unordered_map<uint32_t, Size> composites;
  bool ambiguous_workgroup_size = false;
  for (size_t offset = 5; offset < size / 4;) {
    const auto count = word(offset) >> 16;
    const auto op = static_cast<spv::Op>(word(offset) & 0xffff);
    if (!count || count > size / 4 - offset)
      throw std::invalid_argument(prefix +
                                  "has a truncated SPIR-V instruction");
    const auto at = [&](size_t i) { return word(offset + i); };
    if (op == spv::OpExecutionMode && count >= 3 && at(1) == entry_id) {
      if (at(2) == spv::ExecutionModeLocalSize && count == 6)
        local_size = Size{at(3), at(4), at(5)};
      if (at(2) == spv::ExecutionModeOutputVertices && count == 4)
        vertices = at(3);
      // OutputPrimitivesEXT and OutputPrimitivesNV have the same opcode.
      if (at(2) == spv::ExecutionModeOutputPrimitivesNV && count == 4)
        primitives = at(3);
    } else if (op == spv::OpExecutionModeId && count == 6 &&
               at(1) == entry_id && at(2) == spv::ExecutionModeLocalSizeId) {
      local_size_ids = Size{at(3), at(4), at(5)};
    } else if (op == spv::OpDecorate && count == 4 &&
               at(2) == spv::DecorationBuiltIn &&
               at(3) == spv::BuiltInWorkgroupSize) {
      ambiguous_workgroup_size |=
          workgroup_size_id && *workgroup_size_id != at(1);
      workgroup_size_id = at(1);
    } else if (op == spv::OpTypeInt && count == 4 && at(2) == 32) {
      int32_types.insert(at(1));
    } else if ((op == spv::OpConstant || op == spv::OpSpecConstant) &&
               count == 4 && int32_types.count(at(1))) {
      constants.emplace(at(2), at(3));
    } else if (op == spv::OpConstantNull && count == 3 &&
               int32_types.count(at(1))) {
      constants.emplace(at(2), 0);
    } else if ((op == spv::OpConstantComposite ||
                op == spv::OpSpecConstantComposite) &&
               count == 6) {
      composites.emplace(at(2), Size{at(3), at(4), at(5)});
    }
    offset += count;
  }
  const auto resolve = [&](const Size &ids) -> std::optional<Size> {
    Size values;
    for (size_t i = 0; i < 3; ++i) {
      const auto found = constants.find(ids[i]);
      if (found == constants.end())
        return std::nullopt;
      values[i] = found->second;
    }
    return values;
  };
  // Vulkan's WorkgroupSize built-in takes precedence over
  // LocalSize/LocalSizeId. An unresolved built-in must NOT fall back to a
  // shadowed literal mode.
  if (workgroup_size_id) {
    const auto found = composites.find(*workgroup_size_id);
    local_size = !ambiguous_workgroup_size && found != composites.end()
                     ? resolve(found->second)
                     : std::nullopt;
  } else if (local_size_ids) {
    local_size = resolve(*local_size_ids);
  }

  const auto check = [&](const std::string &label, uint32_t value,
                         uint32_t limit) {
    if (!value || value > limit)
      throw std::invalid_argument(prefix + label + " " + std::to_string(value) +
                                  " exceeds device limit " +
                                  std::to_string(limit) + " (must be nonzero)");
  };
  if (local_size) {
    const auto &limits =
        task ? caps.max_task_work_group_size : caps.max_mesh_work_group_size;
    uint32_t remaining = task ? caps.max_task_work_group_invocations
                              : caps.max_mesh_work_group_invocations;
    for (size_t i = 0; i < 3; ++i) {
      check(std::string("local_size_") + "xyz"[i], (*local_size)[i], limits[i]);
      // Division avoids overflowing even uint64 for three uint32 dimensions.
      if ((*local_size)[i] > remaining)
        throw std::invalid_argument(
            prefix + "local workgroup invocations exceed device limit " +
            std::to_string(task ? caps.max_task_work_group_invocations
                                : caps.max_mesh_work_group_invocations));
      remaining /= (*local_size)[i];
    }
  }
  if (!task) {
    if (vertices)
      check("OutputVertices", *vertices, caps.max_mesh_output_vertices);
    if (primitives)
      check("OutputPrimitivesEXT", *primitives,
            caps.max_mesh_output_primitives);
  }
}

}  // namespace taichi::lang::vulkan
