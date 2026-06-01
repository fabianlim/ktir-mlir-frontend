include(AddKTIRPython)

# Where all the Python sources are.
set(KTIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ktdp")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KTIRPythonSources
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  SOURCES
    __init__.py
)
declare_mlir_python_sources(KTIRPythonSources.Dialects
  ADD_TO_PARENT KTIRPythonSources
)
declare_mlir_python_sources(KTIRPythonExtensions
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  SOURCES
    _mlir_libs/_ktir/py.typed
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${KTIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/KtdpOps.td
  SOURCES 
    dialects/ktdp.py
  DIALECT_NAME ktdp
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(KTIRPythonExtensions.Main
  MODULE_NAME _ktir
  ADD_TO_PARENT KTIRPythonExtensions
  SOURCES
    KTIRModule.cpp
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
  add_ktir_python_type_stubs(KTIRPythonExtensions.Main
    ADD_TO_PARENT KTIRPythonSources
    DEPENDS_TARGETS
      KTIRPythonModules.extension._mlir.dso
      KTIRPythonModules.extension._ktir.dso
    OUTPUTS
      _ktir.pyi
  )
endif()

################################################################################
# Generate packages and shared library
################################################################################

if(KTIR_STAND_ALONE)
  set(_source_components
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    KTIRPythonSources
    KTIRPythonExtensions
  )

  if(MLIR_PYTHON_STUBGEN_ENABLED)
    list(APPEND _source_components MLIRPythonExtension.Core.type_stub_gen)
  endif()

  add_mlir_python_common_capi_library(KTIRKtdpPythonCAPI
    INSTALL_COMPONENT KTIRPythonModules
    INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
    OUTPUT_DIRECTORY "${KTIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
    DECLARED_SOURCES ${_source_components}
  )

  ktir_python_set_rpath(KTIRKtdpPythonCAPI 
    RELATIVE_INSTALL_ROOT "../../../.." # NOTE: This is fragile.
  )

  add_mlir_python_modules(KTIRPythonModules
    ROOT_PREFIX "${KTIR_PYTHON_PACKAGE_DIR}"
    INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
    DECLARED_SOURCES ${_source_components}
    COMMON_CAPI_LINK_LIBS
      KTIRKtdpPythonCAPI
  )

  if(MLIR_PYTHON_STUBGEN_ENABLED)
    add_dependencies(KTIRPythonModules KTIRPythonTypeStubs)
  endif()
endif()
