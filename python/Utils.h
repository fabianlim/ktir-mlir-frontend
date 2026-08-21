//===- Utils.h - Shared helpers for the KTIR Python bindings ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_PYTHON_UTILS_H_
#define KTIR_PYTHON_UTILS_H_

#include "mlir-c/Bindings/Python/Interop.h"

#include <nanobind/nanobind.h>

#include <type_traits>

/// Establishes a nanobind type caster for a tablegen-generated integer enum,
/// so a C enum in a binding signature accepts (and returns) the IntEnum that
/// mlir-tblgen already emits into `<pkg>.dialects.<dialect>`.  This keeps the
/// Python-facing API expressed in terms of the generated enum instead of
/// strings, and there is exactly one definition of the enum cases.
///
/// `cEnum` is the C-API enum type used in signatures; `dialect` is the Python
/// dialect submodule and `name` the class within it.
#define KTIR_IMPORT_INT_ENUM_TYPECASTER(cEnum, dialect, name)                  \
  template <>                                                                  \
  struct nanobind::detail::type_caster<cEnum> {                                \
    NB_TYPE_CASTER(cEnum, const_name(MAKE_MLIR_PYTHON_QUALNAME(                \
                              "dialects." #dialect "." #name)))                \
                                                                               \
    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {            \
      std::underlying_type_t<cEnum> asInt;                                      \
      if (!nanobind::try_cast(src, asInt))                                     \
        return false;                                                          \
      value = static_cast<cEnum>(asInt);                                       \
      return true;                                                             \
    }                                                                          \
                                                                               \
    static handle from_cpp(cEnum src, rv_policy, cleanup_list *) noexcept {     \
      const auto asInt = static_cast<std::underlying_type_t<cEnum>>(src);       \
      return nanobind::module_::import_(                                       \
                 MAKE_MLIR_PYTHON_QUALNAME("dialects." #dialect))              \
          .attr(#name)(asInt)                                                  \
          .release();                                                          \
    }                                                                          \
  }

#endif // KTIR_PYTHON_UTILS_H_
