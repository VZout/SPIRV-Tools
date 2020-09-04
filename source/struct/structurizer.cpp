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

#include "source/util/make_unique.h"
#include "source/opt/build_module.h"

namespace spvtools {
namespace struc {

struct Structurizer::Impl {
  Impl(spv_target_env env, spv_validator_options options)
      : target_env(env),
        validator_options(options) {}

  const spv_target_env target_env;       // Target environment.
  MessageConsumer consumer;              // Message consumer.

  spv_validator_options validator_options;  // Options to control validation.
};

Structurizer::Structurizer(spv_target_env env, spv_validator_options validator_options)
    : impl_(MakeUnique<Impl>(env, validator_options)) {
}

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
  for (opt::Function& function : *ir_context->module()) {
    // Restructure a function
    auto pair = OptFunctionToLooperBlocks(function);
    relooper->Calculate(pair.first);

    for (auto& block : pair.second) {
      delete block;
    }
  }
}

struct Triage {
  void BeginBlock();
  void EndBlock();

  Block* current;

  void TriageInst(opt::Instruction* inst) {
      // determine whether this instruction is a loop?

    switch (inst->opcode()) {
      case SpvOp::SpvOpLabel:
        // pressumably handle a new block?
      break;
      case SpvOp::SpvOpBranchConditional:
        HandleIf();
        break;
      case SpvOp::SpvOpSwitch:
        HandleSwitch();
        break;
      case SpvOp::SpvOpBranch:
          // Handle Break and loop?
        break;
      case SpvOp::SpvOpReturn || SpvOp::SpvOpReturnValue:
        HandleReturn();
        break;
      default:
        break; // Do nothing for default
    }
  }

  void TriageBlock(opt::Function& func, opt::BasicBlock* block) {
    block->ForEachInst(
        [&](opt::Instruction* inst) {
          TriageInst(inst);
        });

    block->ForEachSuccessorLabel([&](std::uint32_t* id) {
      if (id) {
        auto it = func.FindBlock(*id);
        TriageBlock(func, it.Get()->get());
      }
    }); 
  }
    
  void OptFunctionToLooperBlocks(opt::Function& func) {
    TriageBlock(func, func.entry().get());
  }
};

std::pair<Block*, std::vector<Block*>> Structurizer::OptFunctionToLooperBlocks(
    opt::Function& func) {

  std::function<Block*(opt::BasicBlock*)> recursive_node_conversion =
      [&](opt::BasicBlock* curr) -> Block* {
    auto new_block = new Block(curr);
    opt::Instruction* condition = nullptr;

    // is if statement
    curr->ForEachInst([&](opt::Instruction* inst) { // should just check the end instead
      if (inst->opcode() == SpvOp::SpvOpBranchConditional) {
        condition = inst;
      }
    });

    curr->ForEachSuccessorLabel([&](std::uint32_t* id) {
      if (id) {
        auto it = func.FindBlock(*id);
        if (it != func.end()) {
          auto child_block = recursive_node_conversion(it.Get()->get());

          new_block->AddBranchTo(child_block, condition, curr);
        }
      }
    });

    return new_block;
  };

  recursive_node_conversion(func.entry().get());

  return std::pair<Block*, std::vector<Block*>>();
}

}  // namespace struc
}  // namespace spvtools
