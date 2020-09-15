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

#include "analyzer.h"
#include "preoptimizer.h"

namespace spvtools {
namespace struc {

Relooper::Relooper(opt::Module* module)
    : root(nullptr),
      min_size(false),
      block_id_counter(1),  // block ID 0 is reserved for clearings
      shape_id_counter(0) {}

Relooper::~Relooper() {
  for (auto& block : blocks) {
    delete block;
  }
}

// utility
static opt::Operand::OperandData CreateOperandDataFromU64(std::size_t val) {
  utils::SmallVector<uint32_t, 2> ret {
      (std::uint32_t)((val & 0xFFFFFFFF00000000LL) >>
                      32),                          // lower int_type_id
      (std::uint32_t)(val & 0xFFFFFFFFLL),  // higher int_type_id
  };

  return ret;
}

static std::unique_ptr<opt::BasicBlock> HandleFollowupMultiplies(
    std::unique_ptr<opt::BasicBlock> in, Shape* parent,
    RelooperBuilder& builder, opt::Function* new_func, bool in_loop) {
  if (!parent->next) {
    return std::move(in);
  }

  auto curr = std::move(in);
  // VIK-TODO: This is useless since we will always have a block
  // if (!curr || curr->name.is()) {
  //  curr = Builder.makeBlock(Ret);
  //}

  // for each multiple after us, we create a block target for breaks to reach
  while (parent->next) {
    auto* multiple = Shape::IsMultiple(parent->next);
    if (!multiple) {
      break;
    }
    for (auto& iter : multiple->inner_map) {
      int id = iter.first;
      Shape* body = iter.second;
      // VIK-TODO: insert debug name for current?
      curr = builder.MakeNewBlockFromBlock(curr.get());
      curr->AddInstructions(
          (body->Render(builder, new_func,
                        in_loop)));  // VIK-TODO: Adding here is correct right?
    }
    parent->next = parent->next->next;
  }
  // after the multiples is a simple or a loop, in both cases we must hit an
  // entry block, and so this is the last one we need to take into account now
  // (this is why we require that loops hit an entry).
  if (parent->next) {
    auto* simple = Shape::IsSimple(parent->next);
    if (simple) {
      // breaking on the next block's id takes us out, where we
      // will reach its rendering
      // VIK-TODO: insert debug name for current?
    } else {
      // add one break target per entry for the loop
      auto* loop = Shape::IsLoop(parent->next);
      assert(loop);
      assert(loop->entries.size() > 0);
      if (loop->entries.size() == 1) {
        // VIK-TODO: insert debug name for current?
      } else {
        for (auto* entry : loop->entries) {
          // VIK-TODO: insert debug name for current?
          curr = builder.MakeNewBlockFromBlock(curr.get());
        }
      }
    }
    return std::move(curr);
  }
}

void Relooper::Calculate(Block* entry) {
  auto pre_optimizer = PreOptimizer(this);
  auto live = pre_optimizer.FindLive(entry);

  // Add incoming branches from live blocks, ignoring dead code
  for (unsigned i = 0; i < blocks.size(); i++) {
    Block* curr = blocks[i];
    if (!contains(live, curr)) {
      continue;
    }
    for (auto& iter : curr->branches_out) {
      iter.first->branches_in.insert(curr);
    }
  }

  BlockSet all_blocks;
  for (auto* Curr : live) {
    all_blocks.insert(Curr);
#ifdef RELOOPER_DEBUG
    PrintDebug("Adding block %d (%s)\n", Curr->Id, Curr->Code);
#endif
  }

  BlockSet entries;
  entries.insert(entry);
  root = Analyzer(this).Process(all_blocks, entries);
  assert(root);
}

std::unique_ptr<opt::Function> Relooper::Render(opt::IRContext* new_context,
                                                opt::Function& old_function) {
  // Create a new function from existing funciton
  auto def_inst = old_function.DefInst().CloneSPTR(new_context);
  auto def_ptr = def_inst.get();
  auto func = std::make_unique<opt::Function>(std::move(def_inst));

  auto builder = RelooperBuilder(new_context, def_ptr,
                                 opt::IRContext::Analysis::kAnalysisNone);

  auto basic_block = root->Render(builder, func.get(), false);

  func->SetFunctionEnd(old_function.EndInst()->CloneSPTR(new_context));

  return func;
}

Block* Block::FromOptBasicBlock(opt::BasicBlock* block) {
  return new Block(block);
}

Block::Block(opt::BasicBlock* code, Operand switch_condition)
    : code(code),
      switch_condition(switch_condition),
      is_checked_multiple_entry(false) {}

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

std::unique_ptr<opt::BasicBlock> Block::Render(RelooperBuilder& builder,
                                               opt::Function* new_func,
                                               bool in_loop) {
  auto ret = std::make_unique<opt::BasicBlock>(
      builder.NewLabel(builder.GetContext()->TakeNextUniqueId()));

  if (is_checked_multiple_entry && in_loop) {
    // ret->list.push_back(builder.makeSetLabel(0));
  }

  if (code) {
    ret->AddInstructions(code);
  }

  if (!processed_branches_out.size()) {
    return ret;
  }

  return ret;
}

Branch::Branch(Operand condition, opt::BasicBlock* code)
    : condition(condition), code(code) {}

Branch::Branch(std::vector<std::size_t> switch_values, opt::BasicBlock* code)
    : condition(NULL_OPERAND), code(code), switch_values(switch_values) {}

opt::Instruction* Branch::Render(RelooperBuilder& builder, Block* target,
                                 bool set_label) {
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

std::unique_ptr<opt::BasicBlock> RelooperBuilder::MakeNewBlockFromBlock(
    opt::BasicBlock* block) {
  auto retval = std::make_unique<opt::BasicBlock>(
      NewLabel(GetContext()->TakeNextUniqueId()));

  retval->AddInstructions(block);

  return std::move(retval);
}

std::unique_ptr<opt::Instruction> RelooperBuilder::MakeCheckLabel(std::size_t value) {
  // check whether we have a int type.
  // if not add the int type
  // insert a constant for the label id.
  // insert the pointer type into the function.
  // OpVariable in the basic block.
  // insert a store op to insert value into the constant.

    /*
    * 
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %PSMain "PSMain"
               OpExecutionMode %PSMain OriginUpperLeft
               OpSource HLSL 600
               OpName %PSMain "PSMain"
               OpName %src_PSMain "src.PSMain"
               OpName %bb_entry "bb.entry"
               OpName %x "x"
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
%_ptr_Function_int = OpTypePointer Function %int
     %PSMain = OpFunction %void None %5
          %6 = OpLabel
          %7 = OpFunctionCall %void %src_PSMain
               OpReturn
               OpFunctionEnd
 %src_PSMain = OpFunction %void None %5
   %bb_entry = OpLabel
          %x = OpVariable %_ptr_Function_int Function
               OpStore %x %int_0
               OpReturn
               OpFunctionEnd
    */
  
  // Create int type. VIK-TODO: Should be optimized away.
  std::size_t int_type_id = GetContext()->TakeNextUniqueId();
  {
    auto data_result = CreateOperandDataFromU64(int_type_id);
    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result)};

    auto new_int_type = std::make_unique<opt::Instruction>(
        SpvOp::SpvOpTypeInt, false, true, operands);

    GetContext()->AddType(std::move(new_int_type));
  }

  // Create constant. VIK-TODO: Should be optimized away.
  std::size_t int_0_constant_id = GetContext()->TakeNextUniqueId();
  {
    auto data_param_0 = CreateOperandDataFromU64(int_type_id);
    auto data_param_1 = CreateOperandDataFromU64(0);
    auto data_result = CreateOperandDataFromU64(int_0_constant_id);

    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),
        opt::Operand(SPV_OPERAND_TYPE_LITERAL_INTEGER, data_param_1)
    };

    auto new_0_const = std::make_unique<opt::Instruction>(
        SpvOp::SpvOpConstant, true, true, operands);

    GetContext()->AddType(std::move(new_0_const));
  }

        // Create pointer type. VIK-TODO: Should be optimized away.
  std::size_t int_pointer_constant_id = GetContext()->TakeNextUniqueId();
  {
    auto data_param_0 =
        CreateOperandDataFromU64(SpvStorageClass::SpvStorageClassFunction);
    auto data_param_1 = CreateOperandDataFromU64(int_type_id);
    auto data_result = CreateOperandDataFromU64(int_pointer_constant_id);

        std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_1)};

    auto new_pointer_int_type = std::make_unique<opt::Instruction>(
        SpvOp::SpvOpTypePointer, true, true, operands);

    GetContext()->AddType(std::move(new_pointer_int_type));
  }

  // storage
 auto data_param_0 =
      CreateOperandDataFromU64(int_pointer_constant_id);
 auto data_param_1 = CreateOperandDataFromU64(SpvStorageClassFunction);
 auto data_result = CreateOperandDataFromU64(GetContext()->TakeNextUniqueId(););

  std::vector<opt::Operand> operands = {
      opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
      opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),
      opt::Operand(SPV_OPERAND_TYPE_STORAGE_CLASS, data_param_1)};

  auto inst = std::make_unique<opt::Instruction>(
      GetContext(), SpvOp::SpvOpVariable, true, true, &operands);

  return inst;
}

std::unique_ptr<opt::BasicBlock> RelooperBuilder::MakeSequence(opt::BasicBlock* lh,
                                               opt::BasicBlock* rh) {
  auto label = NewLabel(GetContext()->TakeNextUniqueId()); // TODO: get a new unique id or reuse previous one?
  auto ret = std::make_unique<opt::BasicBlock>(std::move(label));
  ret->AddInstructions(lh);
  ret->AddInstructions(rh);

  return ret;
}

opt::BasicBlock* SimpleShape::Render(RelooperBuilder& builder,
                                     opt::Function* new_func, bool in_loop) {
  auto ret = inner->Render(builder, new_func, in_loop);
   ret = HandleFollowupMultiplies(std::move(ret), this, builder, new_func, in_loop);
  if (next) {
    ret->AddInstructions(next->Render(builder, new_func, in_loop));
    ret = builder.MakeSequence(ret.get(), next->Render(builder, new_func, in_loop));
  }

  auto ptr = ret.get();
  new_func->AddBasicBlock(std::move(ret));
  return ptr;
}

opt::BasicBlock* MultipleShape::Render(RelooperBuilder& builder,
                                       opt::Function* new_func, bool in_loop) {
  return nullptr;
}

opt::BasicBlock* LoopShape::Render(RelooperBuilder& builder,
                                   opt::Function* new_func, bool in_loop) {
  return nullptr;
}

}  // namespace struc
}  // namespace spvtools
