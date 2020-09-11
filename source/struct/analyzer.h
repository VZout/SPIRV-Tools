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

#ifndef SOURCE_STRUCT_ANALYZER_H_
#define SOURCE_STRUCT_ANALYZER_H_

#include "source/struct/relooper.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace struc {

class Analyzer {
 public:
  Analyzer(Relooper* parent) : parent(parent) {}

  // Create a list of entries from a block. If LimitTo is provided, only results
  // in that set will appear
  void GetBlocksOut(Block* source, BlockSet& entries,
                    BlockSet* limit_to = nullptr);
  // Converts all branchings to a specific target
  void Solipsize(Block* target, Branch::FlowType type, Shape* ancestor,
                 BlockSet& from);
  Shape* MakeSimple(BlockSet& blocks, Block* inner, BlockSet& next_entries);
  Shape* MakeLoop(BlockSet& blocks, BlockSet& entries, BlockSet& next_entries);
  Shape* MakeMultiple(BlockSet& blocks, BlockSet& entries,
                      BlockBlockSetMap& IndependentGroups,
                      BlockSet& next_entries, bool is_checked_multiple);
  void FindIndependentGroups(BlockSet& entries,
                             BlockBlockSetMap& independent_groups,
                             BlockSet* Ignore = nullptr);

  void InvalidateWithChildren(Block* New, BlockBlockSetMap& independent_groups,
                              BlockBlockMap& ownership);

  Shape* Process(BlockSet& blocks, BlockSet& initialEntries);

 private:
  Relooper* parent;
};

}  // namespace struc
}  // namespace spvtools

#endif  // SOURCE_STRUCT_ANALYZER_H_