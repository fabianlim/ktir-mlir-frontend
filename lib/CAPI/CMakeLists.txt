add_ktir_public_c_api_library(KTIRKtdpCAPI
    Dialects.cpp
  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Ktdp-c
  LINK_LIBS PUBLIC
    KTIRKtdpDialect
)
