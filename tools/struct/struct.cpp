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

#include <cassert>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>

#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"
#include "source/spirv_validator_options.h"
#include "source/struct/structurizer.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/log.h"

int main(int argc, const char** argv) {
  /*const std::string source =
      "OpCapability Shader "
      "OpMemoryModel Logical GLSL450 "
      "OpEntryPoint GLCompute %PSMain \"PSMain\" "
      "OpExecutionMode %PSMain LocalSize 1 1 1 "
      "OpSource HLSL 640 "
      "OpName %PSMain \"PSMain\" "
      "%void = OpTypeVoid "
      "%3 = OpTypeFunction %void "
      "%PSMain = OpFunction %void None %3 "
      "%4 = OpLabel "
      "OpReturn "
      "OpFunctionEnd ";*/

  std::ifstream ifs("test.spv");
  std::string source((std::istreambuf_iterator<char>(ifs)),
                      (std::istreambuf_iterator<char>()));


  spvtools::SpirvTools core(SPV_ENV_UNIVERSAL_1_3);
  spvtools::Optimizer opt(SPV_ENV_UNIVERSAL_1_3);

  auto print_msg_to_stderr = [](spv_message_level_t, const char*,
                                const spv_position_t&, const char* m) {
    std::cerr << "error: " << m << std::endl;
  };
  core.SetMessageConsumer(print_msg_to_stderr);
  opt.SetMessageConsumer(print_msg_to_stderr);

  std::vector<uint32_t> spirv;
  if (!core.Assemble(source, &spirv)) return 1;
  if (!core.Validate(spirv)) return 1;

  std::vector<uint32_t> out_spirv;
  spv_validator_options_t spv_val_options;
  auto structurizer =
      new spvtools::struc::Structurizer(SPV_ENV_UNIVERSAL_1_3, &spv_val_options);
  structurizer->Run(spirv, &out_spirv);

  /*opt.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass({{1, "42"}}))
      .RegisterPass(spvtools::CreateFreezeSpecConstantValuePass())
      .RegisterPass(spvtools::CreateUnifyConstantPass())
      .RegisterPass(spvtools::CreateStripDebugInfoPass());
  if (!opt.Run(spirv.data(), spirv.size(), &spirv)) return 1;*/

  std::string disassembly;
  if (!core.Disassemble(out_spirv, &disassembly)) 
      return 1;
  std::cout << disassembly << "\n";

  return 0;
}
