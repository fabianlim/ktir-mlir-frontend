add_ktir_library(KTIRKtdpDialect
    KtdpAttrs.cpp
    KtdpDialect.cpp
    KtdpOps.cpp
    KtdpTypes.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Ktdp
    ${PROJECT_BINARY_DIR}/include/Ktdp

  GROUP DIALECT
  EXPORT_NAME KtdpDialect

  DEPENDS
    KTIRKtdpAttrInterfacesIncGen
    KTIRKtdpAttrsIncGen
    KTIRKtdpOpsIncGen

  LINK_MLIR_LIBS
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
