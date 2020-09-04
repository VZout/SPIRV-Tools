// Copyright (c) 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "relooper.h"

namespace spvtools {
namespace struc {

Relooper::Relooper(opt::Module* module) :
    root(nullptr),
    min_size(false),
    block_id_counter(1), // block ID 0 is reserved for clearings
    shape_id_counter(0)  {
}

Relooper::~Relooper() {
  for (auto& block : blocks) {
    delete block;
  }
  for (auto& shape : shapes) {
    delete shape;
  }
}

void Relooper::Calculate(Block* entry) {}

std::unique_ptr<opt::Function> Relooper::Render(RelooperBuilder& builder,
                                                opt::Function& old_function,
                                                opt::IRContext* new_context) {
  // Create a new function from existing funciton
  auto func = std::make_unique<opt::Function>(
      old_function.DefInst().CloneSPTR(new_context));

  //func->AddBasicBlocks(blocks.begin(), blocks.end(), blocks.begin());

  return nullptr;
}   

Block* Block::FromOptBasicBlock(opt::BasicBlock* block) 
{ 
    return new Block(block);
}

Block::Block(opt::BasicBlock* code, Operand switch_condition)
    : code(code), switch_condition(switch_condition), is_checked_multiple_entry(false) {}

Block::~Block() {
  for (auto& iter : processed_branches_out) {
    delete iter.second;
  }
  for (auto& iter : branches_out) {
    delete iter.second;
  }
}

void Block::AddBranchTo(Block* target, Operand condition,
                        opt::BasicBlock* code) {
  // cannot add more than one branch to the same target
  assert(!contains(branches_out, target));
  branches_out[target] = new Branch(condition, code);
}

void Block::AddSwitchBranchTo(Block* target, std::vector<std::size_t>&& values,
                              opt::BasicBlock* code) {
  // cannot add more than one branch to the same target
  assert(!contains(branches_out, target));
  branches_out[target] = new Branch(std::move(values), code);
}

opt::BasicBlock* Block::Render(RelooperBuilder& builder, opt::IRContext* context) {
  auto block = new opt::BasicBlock(builder.NewLabel(context->TakeNextUniqueId()));
  block->AddInstructions(code);
  return block;
}

Branch::Branch(Operand condition, opt::BasicBlock* code)
    : condition(condition), code(code) {}

Branch::Branch(std::vector<std::size_t> switch_values, opt::BasicBlock* code)
    : condition(NULL_OPERAND), code(code), switch_values(switch_values) {}

opt::Instruction* Branch::Render(RelooperBuilder& builder,
                                 opt::IRContext* context, Block* target,
    bool set_label)
{

    std::unique_ptr<opt::Instruction> label = nullptr;
    if (set_label) {
      label = builder.NewLabel(target->id);
    }
    opt::BasicBlock* ret = new opt::BasicBlock(std::move(label));

    if (code) {
      ret->AddInstructions(code);
    }

    return nullptr;
}

// TODO-VIK: duplicate
std::unique_ptr<opt::Instruction> RelooperBuilder::NewLabel(uint32_t label_id) {
  std::unique_ptr<opt::Instruction> newLabel(
      new opt::Instruction(GetContext(), SpvOpLabel, 0, label_id, {}));
  return std::move(newLabel);
}

}  // namespace struc
}  // namespace spvtools
