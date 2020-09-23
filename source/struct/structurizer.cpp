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

#include "source/struct/structurizer.h"

#include "source/opt/build_module.h"
#include "source/opt/iterator.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace struc {

struct Task;

struct Triager {
  using TaskPtr = std::shared_ptr<Task>;

  opt::Function& function;
  Relooper& relooper;
  Block* curr_relooper_block;
  std::vector<TaskPtr> stack;
  std::map<std::uint32_t, Block*> break_targets;
  std::map<std::uint32_t, Block*> id_block_map;

  Triager(Relooper& relooper, opt::Function& function)
      : relooper(relooper), function(function) {}

  void Triage(opt::Instruction* curr) {}
  void Triage(opt::BasicBlock* curr) {}

  std::unique_ptr<opt::Instruction> CopyInst(opt::Instruction* inst) {
    return inst->CloneSPTR(relooper.GetContext());
  }

  void AddBranch(
      Block* from, Block* to,
      Operand condition =
          NULL_OPERAND) {  // VIK-TODO operand might need to be changed to bb.
    from->AddBranchTo(to, condition);
  }

  void StopControlFlow() {
    StartBlock();
  }

  Block* MakeBlock() { return relooper.NewBlock(); }

  Block* StartBlock() { return SetCurrBlock(MakeBlock()); }

  void FinishBlock() {
    // irrelevant for spirv. could extent for validaiton.
  }

  Block* SetCurrBlock(Block* curr) {
    if (curr) {
      FinishBlock();
    }
    return curr_relooper_block = curr;
  }

  Block* GetCurrBlock() { return curr_relooper_block; }

  opt::BasicBlock* GetCurrNativeBlock() { return curr_relooper_block->code; }

  // VIK-TODO: Not sure if this will be usefull. Might be interesting for
  // undconditional branching.
  void AddBreakTarget(std::uint32_t id, Block* b) {
    break_targets.insert({id, b});
  }

  Block* GetBreakTarget(std::uint32_t id) { return break_targets[id];
  }

  std::uint32_t GetBranchTargetID(opt::Instruction* branch_inst) {
    return (int)branch_inst->GetOperand(0).AsLiteralUint64();
  }
  Operand GetConditionalBranchCondition(opt::Instruction* branch_inst) {
    return branch_inst->GetOperand(0);
  }
  opt::BasicBlock* GetConditionalBranchTrueBranch(
      opt::Instruction* branch_inst) {
    return function.FindBlock((int)branch_inst->GetOperand(1).AsLiteralUint64())
        .Get()
        ->get();  // VIK-TODO: Is this valid? can you find a block by using the
                  // operand like this? is it the same id?
  }
  opt::BasicBlock* GetConditionalBranchFalseBranch(
      opt::Instruction* branch_inst) {
    return function.FindBlock((int)branch_inst->GetOperand(2).AsLiteralUint64())
        .Get()
        ->get();  // VIK-TODO: Is this valid?
  }
};

struct Task {
  Triager& parent;
  Task(Triager& parent) : parent(parent) {}
  virtual void Run() { assert(false); };
};

struct TriageTask : Task {
  opt::Instruction* curr;
  opt::BasicBlock* curr_bb;

  TriageTask(Triager& parent, opt::Instruction* curr)
      : Task(parent), curr(curr), curr_bb(nullptr) {}

  TriageTask(Triager& parent, opt::BasicBlock* curr)
      : Task(parent), curr(nullptr), curr_bb(curr) {}

  void Run() override {
    if (curr) {
      parent.Triage(curr);
    } else if (curr_bb) {
      parent.Triage(curr_bb);
    }
  }
};

// VIK-TODO: Not sure if identifying blocks are possible...
struct BlockTask : Task {
  opt::BasicBlock* curr;
  Block* later;

  BlockTask(Triager& parent, opt::BasicBlock* curr)
      : Task(parent), curr(curr) {}

  static void Handle(Triager& parent, opt::BasicBlock* curr) {
    if (curr->GetLabel()) {
      // we may be branched to. create a target, and
      // ensure we are called at the join point
      auto task = std::make_shared<BlockTask>(parent, curr);
      task->curr = curr;
      task->later = parent.MakeBlock();
      parent.AddBreakTarget(curr->id(), task->later);
      parent.stack.push_back(task);
    }
    curr->ForEachInst([&](opt::Instruction* inst) {
      parent.stack.push_back(std::make_shared<TriageTask>(parent, inst));
    });
  }

  void Run() override { parent.AddBranch(parent.GetCurrBlock(), later); }
};

struct LoopTask : Task {
  LoopTask(Triager& parent) : Task(parent) {}

  static void Handle(Triager& parent, opt::BasicBlock* curr) {
    // VIK-TODO
  }

  void Run() override {
    // VIK-TODO
  }
};

  struct ReturnTask : public Task {
  static void handle(Triager& parent, opt::Instruction* curr) {
    // reuse the return
    parent.GetCurrNativeBlock()->AddInstruction(parent.CopyInst(curr));
    parent.StopControlFlow();
  }
};

    struct UnreachableTask : public Task {
  static void handle(Triager& parent, opt::Instruction* curr) {
    // reuse the return
    parent.GetCurrNativeBlock()->AddInstruction(parent.CopyInst(curr));
    parent.StopControlFlow();
  }
};

struct IfTask : Task {
  opt::Instruction* curr;
  Block* condition;
  Block* if_true_end;
  int phase = 0;

  IfTask(Triager& parent, opt::Instruction* curr) : Task(parent), curr(curr) {}

  static void Handle(Triager& parent, opt::Instruction* curr) {
    auto task = std::make_shared<IfTask>(parent, curr);
    task->curr = curr;
    task->condition = parent.GetCurrBlock();
    auto* if_true_begin = parent.StartBlock();
    parent.AddBranch(task->condition, if_true_begin,
                     parent.GetConditionalBranchCondition(curr));
    // we always have a false in spirv. // VIK-TODO: But this logic might
    // interfere with the searching for a default target when rendering a block
    parent.stack.push_back(task);
    parent.stack.push_back(std::make_shared<TriageTask>(
        parent, parent.GetConditionalBranchFalseBranch(curr)));
    parent.stack.push_back(task);
    parent.stack.push_back(std::make_shared<TriageTask>(
        parent, parent.GetConditionalBranchTrueBranch(curr)));
  }

  // wtf happens here?
  void Run() override {
    if (phase == 0) {
      // end of ifTrue
      if_true_end = parent.GetCurrBlock();
      auto* after = parent.StartBlock();
      // if condition was false, go after the ifTrue, to ifFalse or outside
      parent.AddBranch(condition, after);
      // if (!curr->ifFalse) {
      //  parent.AddBranch(if_true_end,
      //                   after);  // VIK-TODO: Seems to imply the block always
      // has a false statement.
      //}
      phase++;
    } else if (phase == 1) {
      // end if ifFalse
      auto* if_false_end = parent.GetCurrBlock();
      auto* after = parent.StartBlock();
      parent.AddBranch(if_true_end, after);
      parent.AddBranch(if_false_end, after);
    }
  }
};

// VIK-TODO: Rename to branch target.
  struct BreakTask : public Task {
  static void handle(Triager& parent, opt::Instruction* curr) {
    // add the branch. note how if the condition is false, it is the right
    // value there as well
    auto* before = parent.GetCurrBlock();
    parent.AddBranch(before, parent.GetBreakTarget(parent.GetBranchTargetID(curr)),
                     NULL_OPERAND);
    if (/*curr->condition*/ false) { // SPIRV's OpBranch is unconditional.
      auto* after = parent.StartBlock();
      parent.AddBranch(before, after);
    } else {
      parent.StopControlFlow();
    }
  }
};

struct Triage {
  std::unordered_map<std::uint32_t, Block*> block_list;
  std::uint32_t entry_id = 0;

  ~Triage() {
    for (auto& block : block_list) {
      delete block.second;
    }
    block_list.clear();
  }

  void CreateBlock(opt::Function& func, opt::BasicBlock* block) {
    opt::Operand switch_condition = NULL_OPERAND;
    if (block->tail()->opcode() == SpvOp::SpvOpSwitch) {
      switch_condition = block->tail()->GetInOperand(0);
    }
    auto new_block = new Block(block, switch_condition);

    auto id = block->GetLabelInst()->result_id();
    block_list[id] = new_block;

    block->ForEachSuccessorLabel([&](std::uint32_t* id) {
      if (id) {
        auto it = func.FindBlock(*id);
        CreateBlock(func, it.Get()->get());
      }
    });
  }

  void CreateBranches() {
    for (auto& id_block_pair : block_list) {
      auto& block = id_block_pair.second;
      auto tail = block->code->tail();
      if (block->switch_condition != NULL_OPERAND) {  // switch statement
        auto num_operands = tail->NumOperands();

        // Skip the selector and default, (we also skip the first conditional
        // var so we can add 2)
        for (std::uint32_t i = 3; i < num_operands; i += 2) {
          auto target_id = tail->GetSingleWordOperand(
              i);  // VIK-TODO: Is this correct? Does this convert a operand to
                   // the underlying id? %blabla -> [0..]
          auto target = block_list[target_id];
          auto value = tail->GetSingleWordOperand(i) - 1;

          block->AddSwitchBranchTo(
              target, {value},
              block->code);  // VIK-TODO: This block->code bit should not
                             // contain the branching operator anymore.
        }

        // Do default statement
        {
          auto default_operand_idx = 1;
          auto target_id = tail->GetSingleWordOperand(default_operand_idx);
          auto target = block_list[target_id];
          block->AddSwitchBranchTo(
              target, {},
              block->code);  // VIK-TODO: This block->code bit should not
                             // contain the branching operator anymore.
        }

      } else if (tail->opcode() ==
                 SpvOp::SpvOpBranchConditional) {  // if statement
        auto condition = tail->GetOperand(0);
        auto true_target_id = tail->GetSingleWordOperand(1);
        auto false_target_id = tail->GetSingleWordOperand(2);
        block->AddBranchTo(block_list[true_target_id], condition);
        block->AddBranchTo(
            block_list[false_target_id],
            NULL_OPERAND);  // VIK-TODO: NULL-OPERAND since the default target
                            // should not have a condition.
      } else if (tail->opcode() == SpvOp::SpvOpBranch) {  // jumperoni statement
        auto target_id = tail->GetSingleWordOperand(0);
        auto target = block_list[target_id];
        block->AddBranchTo(target, NULL_OPERAND);
      }
    }
  }

  void OptFunctionToLooperBlocks(opt::Function& func) {
    entry_id = func.entry().get()->id();
    CreateBlock(func, func.entry().get());
    CreateBranches();
  }

  Block* GetEntry() { return block_list[entry_id]; }
};

struct Structurizer::Impl {
  Impl(spv_target_env env, spv_validator_options options)
      : target_env(env), validator_options(options) {}

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.

  spv_validator_options validator_options;  // Options to control validation.
};

Structurizer::Structurizer(spv_target_env env,
                           spv_validator_options validator_options)
    : impl_(MakeUnique<Impl>(env, validator_options)) {}

void Structurizer::Run(const std::vector<uint32_t>& binary_in,
                       std::vector<uint32_t>* binary_out) {
  spvtools::SpirvTools tools(impl_->target_env);
  tools.SetMessageConsumer(impl_->consumer);
  if (!tools.IsValid()) {
    impl_->consumer(SPV_MSG_ERROR, nullptr, {},
                    "Failed to create SPIRV-Tools interface; stopping.");
    return;
  }

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in[0], binary_in.size(),
                      impl_->validator_options)) {
    impl_->consumer(SPV_MSG_ERROR, nullptr, {},
                    "Initial binary is invalid; stopping.");
    return;
  }

  // Build the module from the input binary.
  std::unique_ptr<opt::IRContext> ir_context = BuildModule(
      impl_->target_env, impl_->consumer, binary_in.data(), binary_in.size());
  assert(ir_context);

  auto relooper = std::make_unique<Relooper>(ir_context->module());

  // the new ir context
  auto target_irContext =
      MakeUnique<opt::IRContext>(impl_->target_env, impl_->consumer);

  // magic header
  opt::ModuleHeader header;
  header.bound = 0;
  header.generator = SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_ASSEMBLER, 0);
  header.magic_number = SpvMagicNumber;
  header.version = spvVersionForTargetEnv(impl_->target_env);
  header.reserved = 0;
  target_irContext->module()->SetHeader(header);

  // memory modal
  target_irContext->SetMemoryModel(
      ir_context->module()->GetMemoryModel()->CloneSPTR(
          target_irContext.get()));
  ir_context->module()->GetMemoryModel()->CloneSPTR(target_irContext.get());

  // types
  for (auto& ty : ir_context->module()->GetTypes()) {
    target_irContext->module()->AddType(ty->CloneSPTR(target_irContext.get()));
  }

  // constants
  for (auto& constant : ir_context->module()->GetConstants()) {
    target_irContext->module()->AddType(
        constant->CloneSPTR(target_irContext.get()));
  }

  // capabilities
  for (auto& capability : ir_context->module()->capabilities()) {
    target_irContext->module()->AddCapability(
        capability.CloneSPTR(target_irContext.get()));
  }

  // extensions
  for (auto& extension : ir_context->module()->extensions()) {
    target_irContext->module()->AddExtension(
        extension.CloneSPTR(target_irContext.get()));
  }

  // execution modes
  for (auto& mode : ir_context->module()->execution_modes()) {
    target_irContext->module()->AddExecutionMode(
        mode.CloneSPTR(target_irContext.get()));
  }

  // annotations modes
  for (auto& annotation : ir_context->module()->annotations()) {
    target_irContext->module()->AddAnnotationInst(
        annotation.CloneSPTR(target_irContext.get()));
  }

  // debug1 modes
  for (auto& inst : opt::make_range(ir_context->module()->debug1_begin(),
                                    ir_context->module()->debug1_end())) {
    target_irContext->module()->AddDebug1Inst(
        inst.CloneSPTR(target_irContext.get()));
  }

  for (opt::Function& function : *ir_context->module()) {
    // Convert the basic blocks into relooper friendly structures.
    Triage tri;
    tri.OptFunctionToLooperBlocks(function);

    // Restructure a function
    relooper->Calculate(tri.GetEntry());

    target_irContext->AddFunction(
        relooper->Render(target_irContext.get(), function));
  }

  std::vector<uint32_t>* structurized_binary = new std::vector<uint32_t>();
  target_irContext->module()->ToBinary(structurized_binary,
                                       /* skip_nop = */ true);

  *binary_out = *structurized_binary;
}

}  // namespace struc
}  // namespace spvtools

// VIK-TODO: double check i clone with the correct target ir
