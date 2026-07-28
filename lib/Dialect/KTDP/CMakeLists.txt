add_ktir_library(KTIRKtdpDialect
    KTDPAttrs.cpp
    KTDPDialect.cpp
    KTDPOps.cpp
    KTDPTypes.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir/Dialect/KTDP
    ${PROJECT_BINARY_DIR}/include/ktir/Dialect/KTDP

  GROUP DIALECT
  EXPORT_NAME KtdpDialect

  DEPENDS
    KTIRKTDPAttrInterfacesIncGen
    KTIRKTDPAttrsIncGen
    KTIRKTDPIncGen
)

mlir_target_link_libraries(KTIRKtdpDialect PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAnalysis
  MLIRPresburger
)
