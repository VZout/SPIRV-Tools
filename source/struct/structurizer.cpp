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


  relooper = std::make_unique<Relooper>(ir_context->module());
  for (opt::Function& function : *ir_context->module()) {
    // Restructure a function
    relooper->Calculate(Block::FromOptBasicBlock(function.entry().get()));
  }
}

}  // namespace struc
}  // namespace spvtools
