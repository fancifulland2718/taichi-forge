#include "taichi/rhi/vulkan/shader_specialization.h"
#include "taichi/system/profiler.h"

#include <stdexcept>
#include <map>
#include <limits>

#include "spirv-tools/libspirv.hpp"
#include "source/opt/aggressive_dead_code_elim_pass.h"
#include "source/opt/build_module.h"
#include "source/opt/ccp_pass.h"
#include "source/opt/dead_branch_elim_pass.h"
#include "source/opt/eliminate_dead_functions_pass.h"
#include "source/opt/flatten_decoration_pass.h"
#include "source/opt/fold_spec_constant_op_and_composite_pass.h"
#include "source/opt/freeze_spec_constant_value_pass.h"
#include "source/opt/inline_exhaustive_pass.h"
#include "source/opt/ir_context.h"
#include "source/opt/constants.h"
#include "source/opt/local_single_block_elim_pass.h"
#include "source/opt/merge_return_pass.h"
#include "source/opt/pass_manager.h"
#include "source/opt/set_spec_constant_default_value_pass.h"
#include "source/opt/simplification_pass.h"
#include "source/opt/type_manager.h"
#include "source/util/parse_number.h"

namespace taichi::lang::vulkan {

std::vector<uint32_t> specialize_spirv_shader(
    const std::vector<uint32_t> &code,
    const std::unordered_map<uint32_t, std::string> &values) {
  TI_COMPILE_PROFILER("cpp.vulkan.freeze_shader_specialization");
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
  // The scalar declarations and their grouped decorations were resolved on
  // this private module. Freeze its values without parsing the code again.
  spvtools::opt::PassManager passes;
  passes.SetTargetEnv(environment);
  passes.SetMessageConsumer(consumer);
  passes.AddPass<spvtools::opt::SetSpecConstantDefaultValuePass>(std::move(bits));
  passes.AddPass<spvtools::opt::FlattenDecorationPass>();
  passes.AddPass<spvtools::opt::FreezeSpecConstantValuePass>();
  passes.AddPass<spvtools::opt::FoldSpecConstantOpAndCompositePass>();
  if (passes.Run(context.get()) == spvtools::opt::Pass::Status::Failure)
    throw std::invalid_argument("Invalid specialized Vulkan 1.3 shader: " +
                                diagnostic);
  std::vector<uint32_t> output;
  context->module()->ToBinary(&output, true);
  context.reset();
  if (!validator.Validate(output))
    throw std::invalid_argument("Invalid specialized Vulkan 1.3 shader: " +
                                diagnostic);
  return output;
}

namespace {
std::vector<uint32_t> bind_spirv_arguments(
    const std::vector<uint32_t> &code, uint32_t argument_binding,
    const std::vector<ShaderArgumentConstant> &values, size_t *folded,
    bool layout_words_only) {
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
  auto context = spvtools::BuildModule(environment, consumer, code.data(), code.size());
  if (!context)
    throw std::invalid_argument("Cannot read SPIR-V arguments: " + diagnostic);
  // Normalize decorations, replace ABI loads and clean up the same module.
  // A prepared entry can have millions of instructions before specialization.
  spvtools::opt::PassManager passes;
  passes.SetTargetEnv(environment);
  passes.SetMessageConsumer(consumer);
  passes.AddPass<spvtools::opt::FlattenDecorationPass>();
  if (passes.Run(context.get()) == spvtools::opt::Pass::Status::Failure)
    throw std::invalid_argument("Cannot prepare SPIR-V arguments: " + diagnostic);
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
  // Only pointers derived from the argument UBO can denote a prepare-time
  // fact. Follow their uses once per access chain, sharing the accumulated ABI
  // offset across its loads. Other buffers and dynamic indices stay live.
  auto *defs = context->get_def_use_mgr();
  struct Address {
    uint32_t pointer;
    uint32_t type;
    uint32_t offset;
  };
  auto *pointer_type = defs->GetDef(defs->GetDef(argument)->type_id());
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer)
    return code;
  std::vector<Address> pending{
      {argument, pointer_type->GetSingleWordInOperand(1), 0}};
  using Instruction = spvtools::opt::Instruction;
  std::vector<std::pair<Instruction *, ShaderArgumentConstant>> loads;
  while (!pending.empty()) {
    const auto base = pending.back();
    pending.pop_back();
    defs->ForEachUser(base.pointer, [&](Instruction *user) {
      const auto op = user->opcode();
      if (op == spv::Op::OpLoad) {
        // Loads with explicit memory operands retain their access semantics.
        if (user->NumInOperands() != 1 || user->type_id() != base.type)
          return;
        const auto value = constants.find(base.offset);
        if (value == constants.end())
          return;
        auto *type = defs->GetDef(base.type);
        if (type && type->opcode() == spv::Op::OpTypeInt &&
            type->GetSingleWordInOperand(0) == value->second.bits)
          loads.emplace_back(user, value->second);
        return;
      }
      if ((op != spv::Op::OpAccessChain && op != spv::Op::OpInBoundsAccessChain) ||
          user->GetSingleWordInOperand(0) != base.pointer)
        return;
      uint32_t member_type = base.type;
      uint64_t offset = base.offset;
      for (uint32_t i = 1; i < user->NumInOperands(); ++i) {
        auto *index_def = defs->GetDef(user->GetSingleWordInOperand(i));
        if (!index_def || index_def->opcode() != spv::Op::OpConstant ||
            index_def->NumInOperands() != 1)
          return;
        const auto index = index_def->GetSingleWordInOperand(0);
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
          if (!element || element->opcode() != spv::Op::OpTypeInt ||
              index >= aggregate->GetSingleWordInOperand(1))
            return;
          offset += uint64_t(index) * (element->GetSingleWordInOperand(0) / 8);
        } else {
          return;
        }
        if (offset > std::numeric_limits<uint32_t>::max())
          return;
      }
      pending.push_back({user->result_id(), member_type, static_cast<uint32_t>(offset)});
    });
  }
  if (loads.empty())
    return code;
  if (folded)
    *folded = loads.size();
  auto *manager = context->get_constant_mgr();
  for (const auto &[load, value] : loads) {
    auto *type = defs->GetDef(load->type_id());
    const auto *constant = manager->GetIntConst(
        value.value, value.bits, type->GetSingleWordInOperand(1) != 0);
    auto *definition = manager->GetDefiningInstruction(constant, load->type_id());
    context->ReplaceAllUsesWith(load->result_id(), definition->result_id());
    context->KillInst(load);
  }
  auto has_outlined_functions = [&] {
    auto function = context->module()->begin();
    return function != context->module()->end() && ++function != context->module()->end();
  };
  if (layout_words_only && !has_outlined_functions()) {
    std::vector<uint32_t> output;
    context->module()->ToBinary(&output, true);
    return output;
  }
  // Kernel metadata owns the descriptor ABI and prepared resource views reuse
  // it. Fold unreachable computations while retaining their binding declarations;
  // removing a variable here would disagree with that immutable task contract.
  context->set_preserve_bindings(true);
  context->set_preserve_spec_constants(true);
  // A branch guard may be stored in a function-local temporary. Forward
  // its last store in the same block before CCP so a disabled shared-memory
  // access disappears together with its branch when ABI values become known.
  passes.AddPass<spvtools::opt::LocalSingleBlockLoadStoreElimPass>();
  passes.AddPass<spvtools::opt::CCPPass>();
  // Remove the branches disabled by the bound arguments before simplifying
  // their bodies. Simplification can expose additional constant conditions.
  passes.AddPass<spvtools::opt::DeadBranchElimPass>();
  passes.AddPass<spvtools::opt::SimplificationPass>();
  passes.AddPass<spvtools::opt::DeadBranchElimPass>();
  passes.AddPass<spvtools::opt::AggressiveDCEPass>(false, false);
  if (passes.Run(context.get()) == spvtools::opt::Pass::Status::Failure)
    throw std::invalid_argument("Cannot specialize SPIR-V arguments: " + diagnostic);
  // Prune with the bound ABI first. Only the surviving outlined calls can
  // expose new caller-local values when the prepared pipeline is materialized.
  if (has_outlined_functions()) {
    passes.AddPass<spvtools::opt::MergeReturnPass>();
    passes.AddPass<spvtools::opt::InlineExhaustivePass>();
    passes.AddPass<spvtools::opt::EliminateDeadFunctionsPass>();
    passes.AddPass<spvtools::opt::AggressiveDCEPass>(false, false);
    if (passes.Run(context.get()) == spvtools::opt::Pass::Status::Failure)
      throw std::invalid_argument("Cannot inline prepared SPIR-V calls: " + diagnostic);
  }
  std::vector<uint32_t> output;
  context->module()->ToBinary(&output, true);
  return output;
}
}  // namespace

std::vector<uint32_t> specialize_spirv_arguments(
    const std::vector<uint32_t> &code, uint32_t argument_binding,
    const std::vector<ShaderArgumentConstant> &values, size_t *folded) {
  return bind_spirv_arguments(code, argument_binding, values, folded, false);
}

std::vector<uint32_t> bind_spirv_layout_words(
    const std::vector<uint32_t> &code, uint32_t argument_binding,
    const std::vector<ShaderArgumentConstant> &values, size_t *folded) {
  return bind_spirv_arguments(code, argument_binding, values, folded, true);
}

}  // namespace taichi::lang::vulkan
