//===-- KTIRModule.cpp ------------------------------------------*- c++ -*-===//
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

#include "ktir-c/Dialect/KTDP.h"
#include "ktir-c/Dialect/SpyreOp.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/Tensor.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"

#include "Utils.h"

namespace nb = nanobind;
namespace py = mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

// Bind the C-API memory-space enum to the tablegen-generated
// `mlir_ktdp.dialects.ktdp.MemorySpaceKind` IntEnum.
KTIR_IMPORT_INT_ENUM_TYPECASTER(MlirKTDPMemorySpaceKind, ktdp, MemorySpaceKind);

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktdp {

struct PyAccessTileType : PyConcreteType<PyAccessTileType, PyShapedType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAKTDPAccessTileType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDPAccessTileTypeGetTypeID;
  static constexpr const char *pyClassName = "AccessTileType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, MlirType elementType, 
              DefaultingPyMlirContext context) {
          return PyAccessTileType(
              context->getRef(),
              mlirKTDPAccessTileTypeGet(
                  shape.size(), shape.data(),
                  elementType));
        },
        nb::arg("shape"), nb::arg("element_type"), nb::arg("context") = nb::none());
  }

};

struct PyRuntimeArgType : PyConcreteType<PyRuntimeArgType, PyType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAKTDPRuntimeArgType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDPRuntimeArgTypeGetTypeID;
  static constexpr const char *pyClassName = "RuntimeArgType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirType underlyingType, std::optional<int64_t> granularity,
              std::optional<int64_t> upperbound, DefaultingPyMlirContext context) {
          return PyRuntimeArgType(
              context->getRef(),
              mlirKTDPRuntimeArgTypeGet(
                  context->get(), underlyingType,
                  granularity.value_or(-1),
                  upperbound.value_or(-1)));
        },
        nb::arg("underlying_type"),
        nb::arg("granularity") = nb::none(),
        nb::arg("upperbound") = nb::none(),
        nb::arg("context") = nb::none());
    c.def_prop_ro("underlying_type", [](PyRuntimeArgType &self) {
      return mlirKTDPRuntimeArgTypeGetUnderlyingType(self);
    });
    c.def_prop_ro("granularity", [](PyRuntimeArgType &self) -> std::optional<int64_t> {
      int64_t v = mlirKTDPRuntimeArgTypeGetGranularity(self);
      return v >= 0 ? std::optional<int64_t>(v) : std::nullopt;
    });
    c.def_prop_ro("upperbound", [](PyRuntimeArgType &self) -> std::optional<int64_t> {
      int64_t v = mlirKTDPRuntimeArgTypeGetUpperbound(self);
      return v >= 0 ? std::optional<int64_t>(v) : std::nullopt;
    });
  }
};

struct PyMemorySpaceAttr : PyConcreteAttribute<PyMemorySpaceAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAKTDPMemorySpaceAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKTDPMemorySpaceAttrGetTypeID;
  static constexpr const char *pyClassName = "MemorySpaceAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirKTDPMemorySpaceKind kind, std::optional<int32_t> ctId,
              DefaultingPyMlirContext context) {
          // ct_id is only valid for ct_local; let the attribute verifier
          // reject the invalid combinations rather than duplicating it here.
          // getChecked emits a diagnostic and returns null on failure, so
          // capture it and surface it as ir.MLIRError.
          PyMlirContextRef ctxRef = context->getRef();
          PyMlirContext::ErrorCapture errors(ctxRef);
          MlirAttribute attr = mlirKTDPMemorySpaceAttrGet(
              ctxRef->get(), kind, ctId.value_or(-1));
          if (mlirAttributeIsNull(attr))
            throw MLIRError("Invalid memory space attribute", errors.take());
          return PyMemorySpaceAttr(ctxRef, attr);
        },
        nb::arg("kind"), nb::arg("ct_id") = nb::none(),
        nb::arg("context") = nb::none());
    c.def_prop_ro("kind", [](PyMemorySpaceAttr &self) {
      return mlirKTDPMemorySpaceAttrGetKind(self);
    });
    c.def_prop_ro("ct_id", [](PyMemorySpaceAttr &self) -> std::optional<int32_t> {
      int32_t v = mlirKTDPMemorySpaceAttrGetCtId(self);
      return v >= 0 ? std::optional<int32_t>(v) : std::nullopt;
    });
  }
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktdp

//===----------------------------------------------------------------------===//
// _ktir Module
//===----------------------------------------------------------------------===//

NB_MODULE(_ktir, m) {
  m.def(
      "register_dialects",
      [](py::DefaultingPyMlirContext context, bool load) {
        MlirContext context_ = context.get()->get();
        for (auto handle : {
            mlirGetDialectHandle__arith__(),
            mlirGetDialectHandle__func__(),
            mlirGetDialectHandle__ktdp__(),
            mlirGetDialectHandle__linalg__(),
            mlirGetDialectHandle__math__(),
            mlirGetDialectHandle__scf__(),
            mlirGetDialectHandle__spyreop__(),
            mlirGetDialectHandle__tensor__(),
        }) {
          mlirDialectHandleRegisterDialect(handle, context_);
          if (load)
            mlirDialectHandleLoadDialect(handle, context_);
        }
      },
      nb::arg("context") = nb::none(), nb::arg("load") = true);

  // Register passes through the CAPI so registration lands in the single
  // PassRegistry embedded in the shared CAPI aggregate library (the same
  // registry PassManager pipeline parsing consults). Calling the C++
  // ktir::registerKTIRConversionPasses() directly here would populate _ktir.so's
  // own statically-linked registry instead, which PassManager never sees.
  m.def("register_passes", []() { mlirKTIRRegisterPasses(); });

  //===--------------------------------------------------------------------===//
  // _ktir.ktdp Module
  //===--------------------------------------------------------------------===//

  auto ktdp = m.def_submodule("ktdp");
  py::ktdp::PyAccessTileType::bind(ktdp);
  py::ktdp::PyRuntimeArgType::bind(ktdp);
  py::ktdp::PyMemorySpaceAttr::bind(ktdp);
}
