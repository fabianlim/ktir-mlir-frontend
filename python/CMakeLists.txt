include(AddMLIRPython)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(KTDP_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ktdp")
set(KTDP_PYTHON_PACKAGE_DIR "${KTIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

# KTDP bindings contain their own copy of the MLIR bindings.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KtdpPythonSources
  ROOT_DIR "${KTDP_PYTHON_ROOT_DIR}"
  SOURCES
    __init__.py)
declare_mlir_python_sources(KtdpPythonExtensions
  ROOT_DIR "${KTDP_PYTHON_ROOT_DIR}")

declare_mlir_python_sources(KtdpPythonSources.Dialects
  ROOT_DIR "${KTDP_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT KtdpPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KtdpPythonSources.Dialects
  ROOT_DIR "${KTDP_PYTHON_ROOT_DIR}"
  TD_FILE dialects/KtdpOps.td
  SOURCES 
    dialects/ktdp_nanobind.py
    _mlir_libs/_ktdpDialectsNanobind/py.typed
  DIALECT_NAME ktdp
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(KtdpPythonExtensions.Main
  MODULE_NAME _ktdpDialectsNanobind
  ADD_TO_PARENT KtdpPythonExtensions
  SOURCES
    KtdpExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPILinalg
    MLIRCAPIMath
    MLIRCAPISCF
    MLIRCAPITensor
    MLIRCAPITransforms
    KTIRKtdpCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  get_target_property(_ktdpextension_srcs KtdpPythonExtensions.Main INTERFACE_SOURCES)
  mlir_generate_type_stubs(
    MODULE_NAME mlir_ktdp._mlir_libs._ktdpDialectsNanobind
    DEPENDS_TARGETS
      KtdpPythonModules.extension._mlir.dso
      KtdpPythonModules.extension._ktdpDialectsNanobind.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
    OUTPUTS
      _ktdpDialectsNanobind/__init__.pyi
      _ktdpDialectsNanobind/ktdp.pyi
    DEPENDS_TARGET_SRC_DEPS "${_ktdpextension_srcs}"
    IMPORT_PATHS "${KTDP_PYTHON_PACKAGE_DIR}/.."
  )
  set(_ktdpDialectsNanobind_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  declare_mlir_python_sources(
    KtdpPythonSources.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    ADD_TO_PARENT KtdpPythonSources
    SOURCES
      _mlir_libs/_ktdpDialectsNanobind/__init__.pyi
      _mlir_libs/_ktdpDialectsNanobind/ktdp.pyi
  )
endif()

################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  KtdpPythonSources
  KtdpPythonExtensions
  MLIRPythonExtension.Core.type_stub_gen
)

add_mlir_python_common_capi_library(KTIRKtdpPythonCAPI
  INSTALL_COMPONENT KtdpPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${KTDP_PYTHON_PACKAGE_DIR}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.." # NOTE: This is fragile.
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(KtdpPythonModules
  ROOT_PREFIX "${KTDP_PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    KTIRKtdpPythonCAPI
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  add_dependencies(KtdpPythonModules ${_ktdpDialectsNanobind_typestub_gen_target})
endif()
