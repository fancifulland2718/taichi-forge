#include "taichi/rhi/vulkan/shader_specialization.h"

#include <stdexcept>
#include <unordered_set>
#include <map>
#include <limits>

#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/constants.h"
#include "source/opt/type_manager.h"
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

std::vector<uint32_t> specialize_spirv_arguments(
    const std::vector<uint32_t> &code, uint32_t argument_binding,
    const std::vector<ShaderArgumentConstant> &values, size_t *folded) {
  if (folded)
    *folded = 0;
  if (values.empty())
    return code;
  constexpr auto environment = SPV_ENV_VULKAN_1_3;
  std::string diagnostic;
  const auto consumer = [&](spv_message_level_t level, const char *,
                            const spv_position_t &, const char *message) {
    if (level <= SPV_MSG_ERROR)
      diagnostic = message;
  };
  spvtools::Optimizer flatten(environment);
  flatten.SetMessageConsumer(consumer);
  flatten.RegisterPass(spvtools::CreateFlattenDecorationPass());
  spvtools::OptimizerOptions options;
  options.set_run_validator(false);
  std::vector<uint32_t> normalized;
  if (!flatten.Run(code.data(), code.size(), &normalized, options))
    throw std::invalid_argument("Cannot prepare SPIR-V arguments: " + diagnostic);
  auto context = spvtools::BuildModule(environment, consumer, normalized.data(), normalized.size());
  if (!context)
    throw std::invalid_argument("Cannot read SPIR-V arguments: " + diagnostic);
  auto *defs = context->get_def_use_mgr();
  std::map<uint32_t, uint32_t> bindings, sets, strides;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> offsets;
  for (auto &instruction : context->module()->annotations()) {
    if (instruction.opcode() == spv::Op::OpDecorate) {
      const auto id = instruction.GetSingleWordInOperand(0);
      const auto decoration = instruction.GetSingleWordInOperand(1);
      if (decoration == uint32_t(spv::Decoration::Binding))
        bindings[id] = instruction.GetSingleWordInOperand(2);
      else if (decoration == uint32_t(spv::Decoration::DescriptorSet))
        sets[id] = instruction.GetSingleWordInOperand(2);
      else if (decoration == uint32_t(spv::Decoration::ArrayStride))
        strides[id] = instruction.GetSingleWordInOperand(2);
    } else if (instruction.opcode() == spv::Op::OpMemberDecorate &&
               instruction.GetSingleWordInOperand(2) == uint32_t(spv::Decoration::Offset)) {
      offsets[{instruction.GetSingleWordInOperand(0), instruction.GetSingleWordInOperand(1)}] =
          instruction.GetSingleWordInOperand(3);
    }
  }
  uint32_t argument = 0;
  for (auto &instruction : context->module()->types_values()) {
    if (instruction.opcode() != spv::Op::OpVariable ||
        instruction.GetSingleWordInOperand(0) != uint32_t(spv::StorageClass::Uniform))
      continue;
    const auto id = instruction.result_id();
    if (bindings.count(id) && bindings.at(id) == argument_binding &&
        sets.count(id) && sets.at(id) == 0) {
      if (argument)
        throw std::invalid_argument("Ambiguous SPIR-V argument binding");
      argument = id;
    }
  }
  if (!argument)
    return code;
  std::map<uint32_t, ShaderArgumentConstant> constants;
  for (const auto &value : values)
    constants.emplace(value.offset, value);
  std::vector<spvtools::opt::Instruction *> replaced;
  context->module()->ForEachInst([&](spvtools::opt::Instruction *load) {
    // Additional memory operands (volatile/visibility/etc.) retain their load.
    if (load->opcode() != spv::Op::OpLoad || load->NumInOperands() != 1)
      return;
    auto *type = defs->GetDef(load->type_id());
    if (!type || type->opcode() != spv::Op::OpTypeInt)
      return;
    std::vector<std::vector<uint32_t>> chains;
    uint32_t pointer = load->GetSingleWordInOperand(0);
    std::unordered_set<uint32_t> seen;
    while (pointer != argument) {
      auto *chain = defs->GetDef(pointer);
      if (!chain || !seen.insert(pointer).second ||
          (chain->opcode() != spv::Op::OpAccessChain &&
           chain->opcode() != spv::Op::OpInBoundsAccessChain))
        return;
      std::vector<uint32_t> indices;
      for (uint32_t i = 1; i < chain->NumInOperands(); ++i) {
        auto *index = defs->GetDef(chain->GetSingleWordInOperand(i));
        if (!index || index->opcode() != spv::Op::OpConstant ||
            index->NumInOperands() != 1)
          return;
        indices.push_back(index->GetSingleWordInOperand(0));
      }
      chains.push_back(std::move(indices));
      pointer = chain->GetSingleWordInOperand(0);
    }
    auto *pointer_type = defs->GetDef(defs->GetDef(argument)->type_id());
    uint32_t member_type = pointer_type->GetSingleWordInOperand(1);
    uint64_t offset = 0;
    for (auto chain = chains.rbegin(); chain != chains.rend(); ++chain) {
      for (const auto index : *chain) {
        auto *aggregate = defs->GetDef(member_type);
        if (!aggregate)
          return;
        if (aggregate->opcode() == spv::Op::OpTypeStruct) {
          const auto member = offsets.find({member_type, index});
          if (index >= aggregate->NumInOperands() || member == offsets.end())
            return;
          offset += member->second;
          member_type = aggregate->GetSingleWordInOperand(index);
        } else if (aggregate->opcode() == spv::Op::OpTypeArray) {
          const auto stride = strides.find(member_type);
          if (stride == strides.end())
            return;
          offset += uint64_t(index) * stride->second;
          member_type = aggregate->GetSingleWordInOperand(0);
        } else if (aggregate->opcode() == spv::Op::OpTypeVector) {
          member_type = aggregate->GetSingleWordInOperand(0);
          auto *element = defs->GetDef(member_type);
          if (element->opcode() != spv::Op::OpTypeInt ||
              index >= aggregate->GetSingleWordInOperand(1))
            return;
          offset += uint64_t(index) * (element->GetSingleWordInOperand(0) / 8);
        } else {
          return;
        }
        if (offset > std::numeric_limits<uint32_t>::max())
          return;
      }
    }
    const auto value = constants.find(static_cast<uint32_t>(offset));
    if (member_type != load->type_id() || value == constants.end() ||
        type->GetSingleWordInOperand(0) != value->second.bits)
      return;
    auto *manager = context->get_constant_mgr();
    // SPIR-V integer literals smaller than one word must be sign extended.
    const auto *constant = manager->GetIntConst(
        value->second.value, value->second.bits,
        type->GetSingleWordInOperand(1) != 0);
    auto *definition = manager->GetDefiningInstruction(constant, load->type_id());
    context->ReplaceAllUsesWith(load->result_id(), definition->result_id());
    replaced.push_back(load);
  });
  if (replaced.empty())
    return code;
  if (folded)
    *folded = replaced.size();
  for (auto *load : replaced)
    context->KillInst(load);
  std::vector<uint32_t> transformed;
  context->module()->ToBinary(&transformed, false);
  context.reset();
  spvtools::Optimizer optimizer(environment);
  optimizer.SetMessageConsumer(consumer);
  optimizer.RegisterPass(spvtools::CreateCCPPass());
  optimizer.RegisterPass(spvtools::CreateSimplificationPass());
  optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
  optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
  std::vector<uint32_t> output;
  if (!optimizer.Run(transformed.data(), transformed.size(), &output, options))
    throw std::invalid_argument("Cannot specialize SPIR-V arguments: " + diagnostic);
  return output;
}

}  // namespace taichi::lang::vulkan
