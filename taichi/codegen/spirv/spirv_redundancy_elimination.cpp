// Adapted from SPIRV-Tools redundancy_elimination.cpp and
// local_redundancy_elimination.cpp, Copyright (c) 2017 Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "taichi/codegen/spirv/spirv_redundancy_elimination.h"

#include "source/opt/local_redundancy_elimination.h"
#include "source/opt/value_number_table.h"

namespace taichi::lang::spirv {
namespace {

class ScopedRedundancyEliminationPass
    : public spvtools::opt::LocalRedundancyEliminationPass {
 public:
  const char *name() const override {
    return "scoped-redundancy-elimination";
  }

  Status Process() override {
    using spvtools::opt::DominatorTreeNode;
    spvtools::opt::ValueNumberTable values(context());
    bool modified = false;
    for (auto &function : *get_module()) {
      if (function.IsDeclaration())
        continue;
      auto &tree = context()->GetDominatorAnalysis(&function)->GetDomTree();
      struct Frame {
        DominatorTreeNode *node;
        size_t next_child;
        size_t undo_begin;
      };
      std::map<uint32_t, uint32_t> available;
      std::vector<uint32_t> undo;
      std::vector<Frame> stack;
      auto enter = [&](DominatorTreeNode *node) {
        stack.push_back({node, 0, undo.size()});
        node->bb_->ForEachInst([&](spvtools::opt::Instruction *instruction) {
          if (instruction->result_id() == 0)
            return;
          const auto value = values.GetValueNumber(instruction);
          if (value == 0)
            return;
          auto [candidate, inserted] =
              available.emplace(value, instruction->result_id());
          if (inserted) {
            undo.push_back(value);
          } else {
            context()->KillNamesAndDecorates(instruction);
            context()->ReplaceAllUsesWith(instruction->result_id(),
                                           candidate->second);
            context()->KillInst(instruction);
            modified = true;
          }
        });
      };
      enter(tree.GetRoot());
      // Iterative DFS also handles deeply nested generated code without
      // depending on the host call-stack limit. Sibling definitions must not
      // remain visible after leaving their dominator subtree.
      while (!stack.empty()) {
        auto &frame = stack.back();
        if (frame.next_child < frame.node->children_.size()) {
          auto *child = frame.node->children_[frame.next_child++];
          enter(child);
        } else {
          while (undo.size() > frame.undo_begin) {
            available.erase(undo.back());
            undo.pop_back();
          }
          stack.pop_back();
        }
      }
    }
    return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
  }
};

}  // namespace

spvtools::Optimizer::PassToken create_scoped_redundancy_elimination_pass() {
  return spvtools::Optimizer::PassToken(
      std::make_unique<ScopedRedundancyEliminationPass>());
}

}  // namespace taichi::lang::spirv
