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

  std::pair<Block*, std::vector<Block*>> OptFunctionToLooperBlocks(opt::Function& func);

 private:
  struct Impl;                  // Opaque struct for holding internal data.
  std::unique_ptr<Impl> impl_;  // Unique pointer to internal data.
 };

}  // namespace struc
}  // namespace spvtools

#endif  // SOURCE_STRUCT_STRUCTURIZER_H_