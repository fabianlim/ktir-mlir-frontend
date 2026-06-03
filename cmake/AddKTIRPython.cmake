include(AddMLIRPython)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(KTIR_PYTHON_PACKAGE_DIR 
  "${KTIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
)
file(MAKE_DIRECTORY "${KTIR_PYTHON_PACKAGE_DIR}")

# FIXME: This is a horrible work-around. Unfortunately, MLIR will not set the
#        INSTALL_RPATH when BUILD_SHARED_LIBS is off, even though we need to
#        link against the DSO. Since we're installing it into the lib/ dir, we
#        fix it by doing it manually.
function(ktir_python_set_rpath target)
  cmake_parse_arguments(ARG "" "RELATIVE_INSTALL_ROOT" "" ${ARGN})

  if(LLVM_LINK_LLVM_DYLIB OR MLIR_LINK_MLIR_DYLIB OR BUILD_SHARED_LIBS)
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

if(MLIR_PYTHON_STUBGEN_ENABLED)
  function(add_ktir_python_type_stubs extension)
    cmake_parse_arguments(ARG "" "ADD_TO_PARENT" "DEPENDS_TARGETS;OUTPUTS;IMPORT_PATHS" ${ARGN})

    get_target_property(_extension_sources ${extension} INTERFACE_SOURCES)
    get_target_property(_extension_module_name ${extension} mlir_python_EXTENSION_MODULE_NAME)

    mlir_generate_type_stubs(
      MODULE_NAME "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs.${_extension_module_name}"
      DEPENDS_TARGETS "${NB_LIBRARY_TARGET_NAME};${ARG_DEPENDS_TARGETS}"
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
      OUTPUTS "${ARG_OUTPUTS}"
      DEPENDS_TARGET_SRC_DEPS "${_extension_sources}"
      IMPORT_PATHS "${KTIR_PYTHON_PACKAGE_DIR}/..;${ARG_IMPORT_PATHS}"
    )

    set(_generated_outputs "${ARG_OUTPUTS}")
    list(TRANSFORM _generated_outputs PREPEND "_mlir_libs/")

    declare_mlir_python_sources(
      ${extension}.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT ${ARG_ADD_TO_PARENT}
      SOURCES "${_generated_outputs}"
    )
  endfunction()
endif()
