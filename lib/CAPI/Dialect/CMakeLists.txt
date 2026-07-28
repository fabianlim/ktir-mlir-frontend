add_ktir_public_c_api_library(KTIRKtdpCAPI
    KTDP.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir-c/Dialect

  EXPORT_NAME KtdpCAPI

  LINK_LIBS PUBLIC
    KTIRKtdpDialect
)
