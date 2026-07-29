add_ktir_library(KTIRKtdpConversions
    KTDPPasses.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir/Dialect/KTDP
    ${PROJECT_BINARY_DIR}/include/ktir/Dialect/KTDP

  GROUP CONVERSION
  EXPORT_NAME KtdpConversions

  DEPENDS
    KTIRKTDPPassesIncGen

  LINK_LIBS PUBLIC
    KTIRKtdpDialect

  LINK_MLIR_LIBS
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRTransformUtils
)
