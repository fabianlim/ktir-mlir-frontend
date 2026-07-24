add_ktir_library(KTIRKtdpConversions
    KtdpPasses.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Ktdp
    ${PROJECT_BINARY_DIR}/include/Ktdp

  GROUP CONVERSION
  EXPORT_NAME KtdpConversions

  DEPENDS
    KTIRKtdpPassesIncGen

  LINK_LIBS PUBLIC
    KTIRKtdpDialect

  LINK_MLIR_LIBS
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRTransformUtils
)
