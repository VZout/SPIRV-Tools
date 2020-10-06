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

Relooper::Relooper(opt::IRContext* context)
    : context(context), root(nullptr),
      min_size(false),
      block_id_counter(1),  // block ID 0 is reserved for clearings
      shape_id_counter(0) {

}

Relooper::~Relooper() {
  //for (auto& block : blocks) {
  //  delete block;
  //}
}

// utility
static opt::Operand::OperandData CreateOperandDataFromU64(std::size_t val) {
  utils::SmallVector<uint32_t, 2> ret(std::vector<uint32_t>{
      (std::uint32_t)((val & 0xFFFFFFFF00000000LL) >> 32),  // lower int_type_id
      (std::uint32_t)(val & 0xFFFFFFFFLL),  // higher int_type_id
  });

  return ret;
}

static opt::Operand::OperandData CreateOperandDataFromU32(std::uint32_t val) {
  
    utils::SmallVector<uint32_t, 2> ret(std::vector<uint32_t>{
      val
    });

  return ret;
}

static opt::BasicBlock* HandleFollowupMultiplies(
    opt::BasicBlock* in, Shape* parent,
    RelooperBuilder& builder, opt::Function* new_func, bool in_loop) {
  if (!parent->next) {
    return in;
  }

  auto curr = std::move(in);
  // VIK-TODO: This is useless since we will always have a block WHAT ABOUT THE NULLPTR CASE?
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
      //curr = builder.MakeNewBlockFromBlock(curr.get());
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
      //curr->GetLabel()->SetResultId(simple->inner->code->id() + 10000);
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
          //curr = builder.MakeNewBlockFromBlock(curr.get());
        }
      }
    }
  }
  return curr;
}

void Relooper::Calculate(Block* entry) {
  auto pre_optimizer = PreOptimizer(this);
  auto live = pre_optimizer.FindLive(entry);

  // Add incoming branches from live blocks, ignoring dead code
  for (unsigned i = 0; i < blocks.size(); i++) {
    Block* curr = blocks[i].get();
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
  old_function.ForEachParam([&](opt::Instruction* param) {
    func->AddParameter(param->CloneSPTR(new_context));
  });

  auto builder = RelooperBuilder(new_context, def_ptr,
                                 opt::IRContext::Analysis::kAnalysisNone);

  auto basic_block = root->Render(builder, func.get(), false);
  
  func->SetFunctionEnd(old_function.EndInst()->CloneSPTR(new_context));

  return std::move(func);
}

Branch* Relooper::AddBranch(Operand condition, opt::BasicBlock* code) {
  auto branch = std::make_unique<Branch>(condition, code);
  auto ptr = branch.get();
  branches.emplace_back(std::move(branch));
  return ptr;
}

Block* Relooper::NewBlock() {

      std::unique_ptr<opt::Instruction> label(
      new opt::Instruction(context, SpvOpLabel, 0, context->TakeNextUniqueId(), {}));
  opt::BasicBlock* bb = new opt::BasicBlock(std::move(label)); // VIK-TODO: mem leak

  auto block =
      std::make_unique<Block>(this, bb, NULL_OPERAND);  // VIK_TODO: SWITCH OMITTED.
  block->id = block_id_counter++;
  auto ptr = block.get();
  blocks.push_back(std::move(block));

  if (ptr->id == 59) {
    int x = 0;
  }

  return ptr;
}

Block* Relooper::AddBlock(opt::BasicBlock* code,
                          opt::BasicBlock* switch_condition) {

  auto block = std::make_unique<Block>(this, code, NULL_OPERAND); // VIK_TODO: SWITCH OMITTED.
  block->id = block_id_counter++;
  auto ptr = block.get();
  blocks.push_back(std::move(block));
  
    if (ptr->id == 48) {
    int x = 0;
  }

  return ptr;   
}

void Relooper::ForEachBlock(std::function<void(Block*)> lambda) {
  for (auto& b : blocks) {
    lambda(b.get());
  }
}

Block::Block(Relooper* relooper, opt::BasicBlock* code, Operand switch_condition)
    : relooper(relooper),
    code(code),
      switch_condition(switch_condition),
      is_checked_multiple_entry(false) {

// validate correctness of instructions. i.e no labels and no return ops.

   auto x = 0;
    code->ForEachInst([&](opt::Instruction* inst) { x++;
    });
}

Block::~Block() {
  //for (auto& iter : processed_branches_out) {
    //delete iter.second;
  //}
  //for (auto& iter : branches_out) {
    //delete iter.second;
  //}
}

void Block::AddBranchTo(Block* target, Operand condition,
                        opt::BasicBlock* code) {


  // cannot add more than one branch to the same target
  assert(!contains(branches_out, target));
  branches_out[target] = new Branch(condition, code);
  branches_out[target] = relooper->AddBranch(condition, code);
}

void Block::AddSwitchBranchTo(Block* target, std::vector<std::size_t>&& values,
                              opt::BasicBlock* code) {
  // cannot add more than one branch to the same target
  assert(!contains(branches_out, target));
  branches_out[target] = new Branch(std::move(values), code);
}

// possible fix: add to func on create. don't return smart ptr. handlemultiples would be problem. but should investigate.
opt::BasicBlock* Block::Render(RelooperBuilder& builder,
                                               opt::Function* new_func,
                                               bool in_loop) {

  auto ret_org = std::make_unique<opt::BasicBlock>(
        builder.NewLabel(builder.GetContext()->TakeNextUniqueId()));
  auto ret = ret_org.get();
  if (new_func) {
    new_func->AddBasicBlock(std::move(ret_org));
  
  } else {
    builder.trash.push_back(std::move(ret_org));
  }

  if (is_checked_multiple_entry && in_loop) {
    ret->AddInstruction(builder.makeSetLabel(
        0));  // VIK-TODO: This is incorrect. Need to make a label first.
  }

  if (code) {
    ret->AddInstructions(code); // TODO: Why is this not doing anything?
  }

  if (processed_branches_out.empty()) {
    return ret;
  }

  bool set_label = true;

  MultipleShape* fused = Shape::IsMultiple(parent->next);
  if (fused) {
    // PrintDebug("Fusing Multiple to Simple\n", 0);
    parent->next = parent->next->next;
    if (set_label && fused->inner_map.size() == processed_branches_out.size() &&
        switch_condition == NULL_OPERAND) {
      set_label = false;
    }
  }

  Block* default_target = nullptr;
  // Find default target
  for (auto& it : processed_branches_out) {
    if ((switch_condition == NULL_OPERAND &&
         it.second->condition == NULL_OPERAND) ||
        (switch_condition != NULL_OPERAND &&
         it.second->switch_values.empty())) {
      assert(!default_target &&
             "block has branches without a default (nullptr for the "
             "condition)");  // Must be exactly one default // nullptr
      default_target = it.first;
    }
  }

  // Each block must branch somewhere.
  assert(default_target);

  opt::BasicBlock* root = nullptr;

  // Emit a list of if/else
  if (switch_condition == NULL_OPERAND) {
    opt::BasicBlock* curr_if = nullptr;
    std::vector<std::unique_ptr<opt::BasicBlock>> finalize_stack;
    opt::BasicBlock* remaining_conditions = nullptr;

    for (auto it = processed_branches_out.begin();; it++) {
      Block* target;
      Branch* details;
      if (it != processed_branches_out.end()) {
        target = it->first;
        if (target == default_target) {
          continue;  // done at the end
        }
        details = it->second;
        assert(details->condition !=
               NULL_OPERAND);  // non-default targets always have a condition.
      } else {
        target = default_target;
        details = processed_branches_out[default_target];
      }
      bool set_curr_label = set_label && target->is_checked_multiple_entry;
      bool has_fused_content = fused && contains(fused->inner_map, target->id);
      if (has_fused_content) {
        assert(details->Type == Branch::FlowType::Break);
        details->Type = Branch::FlowType::Direct;
      }
      opt::BasicBlock* curr_content = nullptr;
      bool is_default = it == processed_branches_out.end();
      if (set_curr_label || details->Type != Branch::FlowType::Direct ||
          has_fused_content) {
        curr_content = details->Render(builder, target, new_func, set_curr_label);

        if (has_fused_content) {
          curr_content = builder.Blockify(
              curr_content, fused->inner_map.find(target->id)
                                ->second->Render(builder, nullptr, in_loop));
        }
      }
      // There is nothing to show in this branch, omit the condition
      if (curr_content) {
        if (is_default) {
          opt::BasicBlock* now = nullptr;
          if (remaining_conditions) {
            auto ifblock = builder.MakeIf(  
                BB_INTO_OPERAND(remaining_conditions),
                curr_content);  // VIK-TODO: This should probably return a basic
                                // block instead. or, return instruct and append
                                // it here to the curr_content
            finalize_stack.push_back(std::move(ifblock));
            now = ifblock.get();
          } else {
            now = curr_content;
          }
          if (!curr_if) {
            assert(!root);
            root = now;
          } else {
            builder.SetIfFalse(
                curr_if, now);  // VIK-TODO: Make a builder wrapper for this.
          }
        } else {
          auto ifblock = builder.MakeIf(details->condition, curr_content);
          auto now = ifblock.get();
          builder.trash.push_back(std::move(ifblock));
          //finalize_stack.push_back(std::move(ifblock));
          if (!curr_if) {
            assert(!root);
            root = curr_if = now;
          } else {
            builder.SetIfFalse(curr_if, now);  // VIK-TODO: use wrapper
            curr_if = now;
          }
        }
      } else {
        auto* now = builder.MakeUnary(RelooperBuilder::UnaryType::AndInt32,
                                      details->condition);
        if (remaining_conditions) {
          remaining_conditions = builder.MakeBinary(
              RelooperBuilder::BinaryType::AndInt32,
              RelooperBuilder::OperandFromBasicBlock(remaining_conditions),
              RelooperBuilder::OperandFromBasicBlock(now));
        } else {
          remaining_conditions = now;
        }
      }
      if (is_default) {
        break;
      }
    }

    // Finalize the if chains.
    // VIK-TODO: Irrelevant for us.
    for (auto& bb : finalize_stack) {
      new_func->AddBasicBlock(std::move(bb)); // doesn't do anything
    }
    finalize_stack.clear();

  }
  // Emit switch
  else {
    std::cout << "ABORT ABORT WE DO NOT SUPPORT SWITCHES";
  }

   if (root) {
    ret->AddInstructions(root);
  }

  return ret;
}

Branch::Branch(Operand condition, opt::BasicBlock* code)
    : condition(condition), code(code) {}

Branch::Branch(std::vector<std::size_t> switch_values, opt::BasicBlock* code)
    : condition(NULL_OPERAND), code(code), switch_values(switch_values) {}

opt::BasicBlock* Branch::Render(RelooperBuilder& builder, Block* target,
                                opt::Function* new_func, bool set_label) {
  auto ret = std::make_unique<opt::BasicBlock>(builder.NewLabel(builder.GetContext()->TakeNextUniqueId())); // VIK-TODO: Wrong label id? prefer unique
  if (set_label) {
    auto label = builder.makeSetLabel(target->id);
    ret->AddInstruction(std::move(label));
  }

  if (code) {
    ret->AddInstructions(code);
  }
  if (set_label) {
    ret->AddInstruction(std::move(builder.makeSetLabel(target->id)));
  }
  if (Type == FlowType::Break) {
    std::vector<opt::Operand> operands = {
        condition,  // condition
        opt::Operand(SPV_OPERAND_TYPE_ID,
                     CreateOperandDataFromU32(target->code->id())),
    };

    ret->AddInstruction(std::make_unique<opt::Instruction>(builder.GetContext(), SpvOpBranch, false, false, operands));
  } else if (Type == FlowType::Continue) {
    // VIK-TODO: make continue shape.
  }

  auto ptr = ret.get();
  new_func->AddBasicBlock(std::move(ret));
  return ptr;
}

// TODO-VIK: duplicate
std::unique_ptr<opt::Instruction> RelooperBuilder::NewLabel(uint32_t label_id) {
  std::unique_ptr<opt::Instruction> newLabel(
      new opt::Instruction(GetContext(), SpvOpLabel, 0, label_id, {}));

    if (label_id == 2189) {
    int x = 0;

  }

  return std::move(newLabel);
}

std::unique_ptr<opt::BasicBlock> RelooperBuilder::MakeNewBlockFromBlock(
    opt::BasicBlock* block) {
  auto retval = std::make_unique<opt::BasicBlock>(
      NewLabel(GetContext()->TakeNextUniqueId()));

  retval->AddInstructions(block);

  return std::move(retval);
}

opt::BasicBlock* RelooperBuilder::Blockify(opt::BasicBlock* lh,
                                           opt::BasicBlock* rh) {

  lh->AddInstructions(rh);

  return lh;
}

std::uint32_t RelooperBuilder::MakeType(SpvOp op) {
  std::uint32_t type_id = GetContext()->TakeNextUniqueId();
  {
    auto data_result = CreateOperandDataFromU64(type_id);
    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result)};

    auto new_int_type = std::make_unique<opt::Instruction>(
        GetContext(), op, false, true, operands);

    GetContext()->AddType(std::move(new_int_type));
  }

  return type_id;
}

std::uint32_t RelooperBuilder::MakeLabelType() {  // Create int type. VIK-TODO:
                                                  // Should be optimized away.
  std::uint32_t int_type_id = GetContext()->TakeNextUniqueId();
  {
    auto data_result = CreateOperandDataFromU64(int_type_id);
    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result)};

    auto new_int_type = std::make_unique<opt::Instruction>(
        GetContext(), SpvOp::SpvOpTypeInt, false, true, operands);

    GetContext()->AddType(std::move(new_int_type));
  }

  return int_type_id;
}

std::uint32_t RelooperBuilder::MakeBoolType() {
  std::uint32_t type_id = GetContext()->TakeNextUniqueId();
  {
    auto data_result = CreateOperandDataFromU64(type_id);
    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result)};

    auto new_int_type = std::make_unique<opt::Instruction>(
        GetContext(), SpvOp::SpvOpTypeBool, false, true, operands);

    GetContext()->AddType(std::move(new_int_type));
  }

  return type_id;
}

std::uint32_t RelooperBuilder::MakeLabelPtrType(std::size_t type_id) {
  // Create pointer type. VIK-TODO: Should be optimized away.
  std::uint32_t ptr_type_id = GetContext()->TakeNextUniqueId();
  {
    auto data_param_0 =
        CreateOperandDataFromU64(SpvStorageClass::SpvStorageClassFunction);
    auto data_param_1 = CreateOperandDataFromU64(type_id);
    auto data_result = CreateOperandDataFromU64(ptr_type_id);

    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_1)};

    auto new_pointer_int_type = std::make_unique<opt::Instruction>(
        GetContext(), SpvOp::SpvOpTypePointer, true, true, operands);

    GetContext()->AddType(std::move(new_pointer_int_type));
  }

  return ptr_type_id;
}

std::uint32_t RelooperBuilder::MakeConstant(std::size_t type_id,
                                            std::size_t value) {
  // Create constant. VIK-TODO: Should be optimized away.
  std::uint32_t int_0_constant_id = GetContext()->TakeNextUniqueId();
  {
    auto data_param_0 = CreateOperandDataFromU64(type_id);
    auto data_param_1 = CreateOperandDataFromU64(value);
    auto data_result = CreateOperandDataFromU64(int_0_constant_id);

    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),
        opt::Operand(SPV_OPERAND_TYPE_LITERAL_INTEGER, data_param_1)};

    auto new_0_const = std::make_unique<opt::Instruction>(
        GetContext(), SpvOp::SpvOpConstant, true, true, operands);

    GetContext()->AddType(std::move(new_0_const));
  }

  return int_0_constant_id;
}

std::unique_ptr<opt::Instruction> RelooperBuilder::MakeLabel() {
  if (label_id != std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("Uh. looks like we already have a label. Abort!");
  }

  auto label_type_id = MakeLabelType();
  auto ptr_type_id = MakeLabelPtrType(label_type_id);

  // create the variable parameter
  auto data_param_0 = CreateOperandDataFromU64(ptr_type_id);
  auto data_param_1 = CreateOperandDataFromU64(SpvStorageClassFunction);
  auto data_result = CreateOperandDataFromU64(GetContext()->TakeNextUniqueId());

  std::vector<opt::Operand> operands = {
      opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
      opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),
      opt::Operand(SPV_OPERAND_TYPE_STORAGE_CLASS, data_param_1)};

  auto inst = std::make_unique<opt::Instruction>(
      GetContext(), SpvOp::SpvOpVariable, true, true, operands);

  this->label_id = inst->result_id();

  makeSetLabel(0);  // VIK-TODO: This instruction gets discarded!!! fixme plz

  return inst;
}

std::unique_ptr<opt::Instruction> RelooperBuilder::MakeCheckLabel(
    std::size_t value) {
  if (label_id == std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("There is no label here");
  }

  label_type_id = MakeLabelType();
  auto bool_type_id = MakeBoolType();
  auto value_const_id = MakeConstant(label_type_id, value);
  auto get_id =
      makeGetLabel()->result_id();  // VIK-TODO: We require to emit this!!!!

  // create the variable parameter
  {
    auto data_param_0 = CreateOperandDataFromU64(bool_type_id);
    auto data_param_1 = CreateOperandDataFromU64(get_id);
    auto data_param_2 = CreateOperandDataFromU64(value_const_id);
    auto data_result =
        CreateOperandDataFromU64(GetContext()->TakeNextUniqueId());

    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),   // bool type id
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_1),   // left hand id
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_2)};  // right hand id

    auto inst = std::make_unique<opt::Instruction>(
        GetContext(), SpvOp::SpvOpIEqual, true, true, operands);

    return inst;
  }
}

std::unique_ptr<opt::Instruction> RelooperBuilder::makeSetLabel(
    std::size_t value) {
  auto const_id = MakeConstant(label_type_id, value);

  // create the store op
  {
    auto data_param_0 = CreateOperandDataFromU64(label_id);
    auto data_param_1 = CreateOperandDataFromU64(const_id);
    auto data_result =
        CreateOperandDataFromU64(GetContext()->TakeNextUniqueId());

    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),   // target
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_1)};  // constant

    auto inst = std::make_unique<opt::Instruction>(
        GetContext(), SpvOp::SpvOpStore, true, true, operands);

    return inst;
  }

  return std::unique_ptr<opt::Instruction>();
}

std::unique_ptr<opt::Instruction> RelooperBuilder::makeGetLabel() {
  // op load
  // create the variable parameter
  {
    auto data_param_0 = CreateOperandDataFromU64(label_type_id);
    auto data_param_1 = CreateOperandDataFromU64(label_id);
    auto data_result =
        CreateOperandDataFromU64(GetContext()->TakeNextUniqueId());

    std::vector<opt::Operand> operands = {
        opt::Operand(SPV_OPERAND_TYPE_RESULT_ID, data_result),
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_0),   // type id
        opt::Operand(SPV_OPERAND_TYPE_ID, data_param_1)};  // var to load

    auto inst = std::make_unique<opt::Instruction>(
        GetContext(), SpvOp::SpvOpLoad, true, true, operands);

    return inst;
  }
}

opt::BasicBlock* RelooperBuilder::MakeUnary(UnaryType type,
                                            opt::Operand condition) {
  // VIK-TODO: Do I need a label here?
  std::unique_ptr<opt::Instruction> label =
      NewLabel(GetContext()->TakeNextUniqueId());
  opt::BasicBlock* ret = new opt::BasicBlock(std::move(label));

  SpvOp type_op = SpvOpNop;
  SpvOp op = SpvOpNop;
  switch (type) {
    case UnaryType::AndInt32:
      type_op = SpvOpTypeInt;
      op = SpvOpINotEqual;
      break;
  }

  auto null_const_type =
      MakeType(type_op);  // VIK-TODO: Could be optimized away.
  auto null_const_id = MakeConstant(null_const_type, 0);

  std::vector<opt::Operand> operands = {
      condition,
      Operand(SPV_OPERAND_TYPE_ID, CreateOperandDataFromU64(type_op))};

  auto inst = std::make_unique<opt::Instruction>(GetContext(), op, true, true,
                                                 operands);

  ret->AddInstruction(std::move(inst));

  return ret;
}

opt::BasicBlock* RelooperBuilder::MakeBinary(BinaryType type,
                                             opt::Operand lh_cond,
                                             opt::Operand rh_cond) {
  // VIK-TODO: Do I need a label here?
  std::unique_ptr<opt::Instruction> label =
      NewLabel(GetContext()->TakeNextUniqueId());
  opt::BasicBlock* ret = new opt::BasicBlock(std::move(label));

  SpvOp op = SpvOpNop;
  switch (type) {
    case BinaryType::AndInt32:
      op = SpvOpIEqual;
      break;
  }

  std::vector<opt::Operand> operands = {lh_cond, rh_cond};

  auto inst = std::make_unique<opt::Instruction>(GetContext(), op, true, true,
                                                 operands);

  ret->AddInstruction(std::move(inst));

  return ret;
}

std::unique_ptr<opt::BasicBlock> RelooperBuilder::MakeIf(
    opt::Operand condition,
                                         opt::BasicBlock* true_branch,
                                         opt::BasicBlock* false_branch) {

    auto unique = GetContext()->TakeNextUniqueId();
  // VIK-TODO: Do I need a label here?
  std::unique_ptr<opt::Instruction> label = NewLabel(unique);
  auto ret = std::make_unique<opt::BasicBlock>(std::move(label));

  std::vector<opt::Operand> operands = {
      condition,  // condition
      opt::Operand(SPV_OPERAND_TYPE_ID,
                   CreateOperandDataFromU32(
                       true_branch->id())),
  };

  if (false_branch) {
    operands.push_back(opt::Operand(
        SPV_OPERAND_TYPE_ID, CreateOperandDataFromU64(false_branch->id())));
  }
  auto inst = std::make_unique<opt::Instruction>(
      GetContext(), SpvOp::SpvOpBranchConditional, false, false, operands);

  ret->AddInstruction(std::move(inst));

  return ret;
}

void RelooperBuilder::SetIfFalse(opt::Instruction* in,
                                 opt::BasicBlock* false_branch) {
  in->AddOperand(opt::Operand(SPV_OPERAND_TYPE_ID,
                              CreateOperandDataFromU32(false_branch->id())));
}

void RelooperBuilder::SetIfFalse(opt::BasicBlock* in,
                                 opt::BasicBlock* false_branch) {
  in->tail()->AddOperand(opt::Operand(
      SPV_OPERAND_TYPE_ID, CreateOperandDataFromU32(false_branch->id())));
}

std::unique_ptr<opt::BasicBlock> RelooperBuilder::MakeSequence(
    opt::BasicBlock* lh, opt::BasicBlock* rh) {
  auto label =
      NewLabel(GetContext()->TakeNextUniqueId());  // TODO: get a new unique id
                                                   // or reuse previous one?
  auto ret = std::make_unique<opt::BasicBlock>(std::move(label));
  ret->AddInstructions(lh);
  ret->AddInstructions(rh);

  return ret;
}

opt::Operand RelooperBuilder::OperandFromBasicBlock(opt::BasicBlock* bb) {
  return opt::Operand(SPV_OPERAND_TYPE_ID,
                      CreateOperandDataFromU64(bb->tail()->result_id()));
}

opt::BasicBlock* SimpleShape::Render(RelooperBuilder& builder,
                                     opt::Function* new_func, bool in_loop) {
    auto ret = inner->Render(builder, new_func, in_loop);

  if (next) {
    auto new_ret =
        HandleFollowupMultiplies(ret, this, builder, new_func, in_loop);
    // ret->AddInstructions(next->Render(builder, new_func, in_loop));
    auto ret2 =
        builder.MakeSequence(new_ret, next->Render(builder, new_func, in_loop));
    auto ptr = ret2.get();
    new_func->AddBasicBlock(std::move(ret2));
    return ptr;
  }

  return ret;
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
