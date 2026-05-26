include(AddMLIRPython)

set(KTIR_PYTHON_PACKAGE_DIR "${KTIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

# KTIR bindings contain their own copy of the MLIR bindings.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

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

# FIXME: This is a horrible work-around. Unfortunately, MLIR will not set the
#        INSTALL_RPATH when BUILD_SHARED_LIBS is off, even though we need to
#        link against the DSO. Since we're installing it into the lib/ dir, we
#        fix it by doing it manually.
function(ktir_python_fix_rpath target)
  cmake_parse_arguments(ARG "" "RELATIVE_INSTALL_ROOT" "" ${ARGN})

  if(LLVM_LINK_LLVM_DYLIB OR MLIR_LINK_MLIR_DYLIB)
    if(APPLE)
      set(_origin_prefix "@loader_path")
    elseif(UNIX)
      set(_origin_prefix "\$ORIGIN")
    else()
      return()
    endif()

    set_property(TARGET ${target} APPEND PROPERTY
      INSTALL_RPATH 
      "${_origin_prefix}/${ARG_RELATIVE_INSTALL_ROOT}/lib${LLVM_LIBDIR_SUFFIX}"
    )
  endif()
endfunction()
