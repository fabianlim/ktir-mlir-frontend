################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  
  KTIRPythonSources
  KTIRPythonExtensions
)


if(MLIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _source_components MLIRPythonExtension.Core.type_stub_gen)
endif()

# Single shared CAPI aggregate library that both _mlir.so and _ktir.so link
# against. This guarantees exactly ONE embedded MLIR runtime (and thus one
# PassRegistry / DialectRegistry) shared across the extension modules, rather
# than each extension statically embedding its own copy. Without this, passes
# registered via register_passes() land in a different registry than the one
# PassManager.parse() consults, and pipeline parsing fails.
add_mlir_python_common_capi_library(KTIRPythonCAPI
  INSTALL_COMPONENT KTIRPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${PYTHON_PACKAGE_DIR}/_mlir_libs"
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(KTIRPythonModules
  ROOT_PREFIX "${PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    KTIRPythonCAPI
)

