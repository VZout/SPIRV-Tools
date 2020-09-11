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

#include "preoptimizer.h"

namespace spvtools {
namespace struc {

BlockSet PreOptimizer::FindLive(Block* entry) {
  BlockSet live;
  BlockList to_investigate;

  to_investigate.push_back(entry);
  while (to_investigate.size() > 0) {
    Block* Curr = to_investigate.front();
    to_investigate.pop_front();
    if (contains(live, Curr)) {
      continue;
    }
    live.insert(Curr);
    for (auto& iter : Curr->branches_out) {
      to_investigate.push_back(iter.first);
    }
  }

  return live;
}

}  // namespace struc
}  // namespace spvtools