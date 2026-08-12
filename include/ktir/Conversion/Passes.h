//===- Passes.h - KTIR conversion pass registration --------------*- C++ -*-===//
//
// Copyright 2026 The KTIR Authors.
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
//
//===----------------------------------------------------------------------===//
//
// This file aggregates the KTIR conversion passes for registration. Include
// the per-conversion header instead when you need a specific pass or its
// populate* entry points.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_CONVERSION_PASSES_H
#define KTIR_CONVERSION_PASSES_H

#include "ktir/Conversion/ConvertToKTIR/ConvertToKTIR.h"

namespace ktir {

// Pull in generated pass registration helpers.
#define GEN_PASS_REGISTRATION
#include "ktir/Conversion/Passes.h.inc"
#undef GEN_PASS_REGISTRATION

} // namespace ktir

#endif // KTIR_CONVERSION_PASSES_H
