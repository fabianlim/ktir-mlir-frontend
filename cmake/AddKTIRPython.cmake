include(AddMLIRPython)

set(KTIR_PYTHON_PACKAGE_DIR "${KTIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

# KTIR bindings contain their own copy of the MLIR bindings.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}")

if(NOT TARGET KTIRPythonSources)
  declare_mlir_python_sources(KTIRPythonSources)
endif()
if(NOT TARGET KTIRPythonExtensions)
  declare_mlir_python_sources(KTIRPythonExtensions)
endif()
if(NOT TARGET KTIRPythonSources.Dialects)
  declare_mlir_python_sources(KTIRPythonSources.Dialects
    ADD_TO_PARENT KTIRPythonSources
  )
endif()
