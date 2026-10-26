#include "taichi/rhi/vulkan/shader_specialization.h"

#include <stdexcept>
#include <unordered_set>

#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/util/parse_number.h"

namespace taichi::lang::vulkan {

std::vector<uint32_t> specialize_spirv_shader(
    const std::vector<uint32_t> &code,
    const std::unordered_map<uint32_t, std::string> &values) {
  constexpr auto environment = SPV_ENV_VULKAN_1_3;
  std::string diagnostic;
  const auto consumer = [&](spv_message_level_t level, const char *,
                            const spv_position_t &, const char *message) {
    if (level <= SPV_MSG_ERROR)
      diagnostic = message;
  };
  spvtools::SpirvTools validator(environment);
  validator.SetMessageConsumer(consumer);
  if (!validator.Validate(code))
    throw std::invalid_argument("Invalid Vulkan 1.3 specialization source: " +
                                diagnostic);

  auto context =
      spvtools::BuildModule(environment, consumer, code.data(), code.size());
  if (!context)
    throw std::invalid_argument("Cannot read SPIR-V specialization source: " +
                                diagnostic);
  std::unordered_map<uint32_t, std::vector<uint32_t>> bits;
  for (const auto &instruction : context->module()->types_values()) {
    const auto opcode = instruction.opcode();
    if (opcode != spv::Op::OpSpecConstant &&
        opcode != spv::Op::OpSpecConstantTrue &&
        opcode != spv::Op::OpSpecConstantFalse)
      continue;
    context->get_decoration_mgr()->ForEachDecoration(
        instruction.result_id(), uint32_t(spv::Decoration::SpecId),
        [&](const spvtools::opt::Instruction &decoration) {
          const auto id = decoration.GetSingleWordInOperand(2);
          const auto value = values.find(id);
          if (value == values.end())
            return;
          const auto *type =
              context->get_def_use_mgr()->GetDef(instruction.type_id());
          std::vector<uint32_t> encoded;
          const auto fail = [&](const std::string &reason) {
            throw std::invalid_argument("Specialization constant " +
                                        std::to_string(id) + ": " + reason);
          };
          if (type->opcode() == spv::Op::OpTypeBool) {
            if (value->second != "true" && value->second != "false")
              fail("requires a bool");
            encoded.push_back(value->second == "true");
          } else {
            spvtools::utils::NumberType number{};
            number.bitwidth = type->GetSingleWordInOperand(0);
            if (type->opcode() == spv::Op::OpTypeInt) {
              number.kind = type->GetSingleWordInOperand(1)
                                ? SPV_NUMBER_SIGNED_INT
                                : SPV_NUMBER_UNSIGNED_INT;
            } else if (type->opcode() == spv::Op::OpTypeFloat) {
              number.kind = SPV_NUMBER_FLOATING;
            } else {
              fail("requires a scalar bool, integer or float declaration");
            }
            std::string error;
            const auto status = spvtools::utils::ParseAndEncodeNumber(
                value->second.c_str(), number,
                [&](uint32_t word) { encoded.push_back(word); }, &error);
            if (status != spvtools::utils::EncodeNumberStatus::kSuccess)
              fail("value does not fit its declared scalar type: " + error);
          }
          const auto existing = bits.find(id);
          if (existing != bits.end() && existing->second != encoded)
            fail("shared SpecId declarations have incompatible scalar types");
          bits[id] = std::move(encoded);
        });
  }
  for (const auto &[id, value] : values) {
    if (!bits.count(id))
      throw std::invalid_argument("Unknown specialization constant ID " +
                                  std::to_string(id));
  }
  context.reset();
  spvtools::Optimizer optimizer(environment);
  optimizer.SetMessageConsumer(consumer);
  // The pinned upstream passes understand grouped decorations and scalar types.
  // Keep this a small explicit transform, not the full optimization pipeline.
  optimizer.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass(bits));
  optimizer.RegisterPass(spvtools::CreateFlattenDecorationPass());
  optimizer.RegisterPass(spvtools::CreateFreezeSpecConstantValuePass());
  optimizer.RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass());
  std::vector<uint32_t> output;
  spvtools::OptimizerOptions options;
  options.set_run_validator(
      false);  // source validated above; check the result below
  if (!optimizer.Run(code.data(), code.size(), &output, options) ||
      !validator.Validate(output))
    throw std::invalid_argument("Invalid specialized Vulkan 1.3 shader: " +
                                diagnostic);
  return output;
}

}  // namespace taichi::lang::vulkan
