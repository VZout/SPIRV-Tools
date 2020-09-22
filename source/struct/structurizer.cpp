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
        block->AddBranchTo(block_list[true_target_id], condition, block->code);
        block->AddBranchTo(block_list[false_target_id], condition, block->code);
      } else if (tail->opcode() == SpvOp::SpvOpBranch) {  // jumperoni statement
        auto target_id = tail->GetSingleWordOperand(0);
        auto target = block_list[target_id];
        block->AddBranchTo(target, NULL_OPERAND, block->code);
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
