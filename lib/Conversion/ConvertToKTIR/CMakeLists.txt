add_ktir_library(KTIRConvertToKTIR
    KTIRCheckLegality.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir/Conversion
    ${PROJECT_BINARY_DIR}/include/ktir/Conversion

  GROUP CONVERSION
  EXPORT_NAME ConvertToKTIR

  DEPENDS
    KTIRConversionPassesIncGen

  LINK_LIBS PUBLIC
    KTIRKtdpDialect

  LINK_MLIR_LIBS
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRTransformUtils
)
