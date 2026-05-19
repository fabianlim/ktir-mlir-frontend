add_ktir_dialect_library(KTIRKtdpDialect
    KtdpAttrs.cpp
    KtdpDialect.cpp
    KtdpOps.cpp
    KtdpTypes.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Ktdp
    ${PROJECT_BINARY_DIR}/include/Ktdp

  DEPENDS
    KTIRKtdpAttrInterfacesIncGen
    KTIRKtdpAttrsIncGen
    KTIRKtdpOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRArithDialect
)
