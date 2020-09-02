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

#ifndef SOURCE_STRUCT_STRUCTURIZER_H_
#define SOURCE_STRUCT_STRUCTURIZER_H_

#include "spirv-tools/libspirv.hpp"

#include "source/struct/relooper.h"

namespace spvtools {
namespace struc {

class Structurizer {
 public:
  Structurizer(spv_target_env env, spv_validator_options validator_options);

  // Disables copy/move constructor/assignment operations.
  Structurizer(const Structurizer&) = delete;
  Structurizer(Structurizer&&) = delete;
  Structurizer& operator=(const Structurizer&) = delete;
  Structurizer& operator=(Structurizer&&) = delete;

  void Run(const std::vector<uint32_t>& binary_in,
           std::vector<uint32_t>* binary_out);

  struct Task {
    Structurizer& parent;
    Task(Structurizer& parent) : parent(parent) {}
    virtual void run() = 0;
  };

  struct TriageTask final : public Task {
    opt::BasicBlock* curr;

    TriageTask(Structurizer& parent, opt::BasicBlock* curr) : Task(parent), curr(curr) {}

    void run() override { parent.Triage(curr); }
  };

  struct BlockTask final : public Task {
    opt::BasicBlock* curr;
    Block* later;

    BlockTask(Structurizer& parent, opt::BasicBlock* curr) : Task(parent), curr(curr) {}

    static void handle(Structurizer& parent, opt::BasicBlock* curr) {
      if (true/*curr->name*/) {
        // we may be branched to. create a target, and
        // ensure we are called at the join point
        auto task = std::make_shared<BlockTask>(parent, curr);
        task->curr = curr;
        task->later = parent.MakeRelooperBlock();
        parent.AddBreakTarget(curr->id(), task->later);
        parent.stack.push_back(task);
      }
      auto& list = curr->list;
      for (int i = int(list.size()) - 1; i >= 0; i--) {
        parent.stack.push_back(std::make_shared<TriageTask>(parent, list[i]));
      }
    }

    void run() override {
      // add fallthrough
      parent.AddBranch(parent.GetCurrentRelooperBlock(), later);
      parent.SetCurrentRelooperBlock(later);
    }
  };


 private:
  std::unique_ptr<Relooper> relooper;
  std::unique_ptr<RelooperBuilder> builder;

  Block* current_block;
  Block* MakeRelooperBlock();
  Block* SetCurrentRelooperBlock(Block* curr);
  Block* SetStartRelooperBlock(Block* curr);
  Block* GetCurrentRelooperBlock();
  opt::BasicBlock* GetCurrentBasicBlock();
  void AddBreakTarget(std::size_t label, Block* target);
  Block* GetBreakTarget(std::size_t label);
  void FinishBlock();
  void AddBranch(Block* from, Block* to, opt::Instruction* condition = nullptr);
  void AddSwitchBranch(Block* from, Block* to, const std::set<std::size_t>& values);

  void Triage(opt::BasicBlock* curr);

  typedef std::shared_ptr<Task> TaskPtr;
  std::vector<TaskPtr> stack;

  struct Impl;                  // Opaque struct for holding internal data.
  std::unique_ptr<Impl> impl_;  // Unique pointer to internal data.
 };

}  // namespace struc
}  // namespace spvtools

#endif  // SOURCE_STRUCT_STRUCTURIZER_H_