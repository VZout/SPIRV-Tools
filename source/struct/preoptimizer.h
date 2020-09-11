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

#ifndef SOURCE_STRUCT_PREOPTIMIZER_H_
#define SOURCE_STRUCT_PREOPTIMIZER_H_

#include "source/struct/relooper.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace struc {

class PreOptimizer {
 public:
  PreOptimizer(Relooper* parent) : parent(parent) {}

  BlockSet FindLive(Block* entry);
  void SplitDeadEnds();

 private:
  Relooper* parent;
};

}  // namespace struc
}  // namespace spvtools

#endif  // SOURCE_STRUCT_PREOPTIMIZER_H_