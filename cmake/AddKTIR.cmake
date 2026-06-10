# Build all CAPI libraries.
add_custom_target(ktir-capi)
# Build all documentation targets.
add_custom_target(ktir-doc)
# Build all generated include targets.
add_custom_target(ktir-headers)

#
# TableGen related helpers
#

# Creates a TableGen target and adds it to ktir-headers.
function(add_ktir_tablegen_target name)
  add_public_tablegen_target(${name})
  set(TABLEGEN_OUTPUT "")
  add_dependencies(ktir-headers ${name})
endfunction()

# Creates a canonical MLIR dialect TableGen target and adds it to ktir-headers.
function(add_ktir_dialect dialect dialect_namespace)
  # TODO: Examine if we want to stick to the .hpp convention or use .h
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.hpp.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.hpp.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.hpp.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_ktir_tablegen_target(KTIR${dialect}IncGen)
endfunction()

# Creates an MLIR TableGen documentation target and adds it to ktir-doc.
function(add_ktir_doc doc_filename output_file output_directory command)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  set(TABLEGEN_OUTPUT "")
  tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
  set(_output "${KTIR_BINARY_DIR}/doc/${output_directory}${output_file}.md")
  add_custom_command(
    OUTPUT "${_output}"
    COMMAND ${CMAKE_COMMAND} -E copy "${TABLEGEN_OUTPUT}" "${_output}"
    DEPENDS "${TABLEGEN_OUTPUT}"
  )
  set_source_files_properties(${_output} PROPERTIES GENERATED TRUE)
  add_custom_target("KTIR${output_file}DocGen" DEPENDS "${_output}")
  add_dependencies(ktir-doc "KTIR${output_file}DocGen")
endfunction()

#
# Target creation helpers
#

# Creates an executable target.
macro(add_ktir_executable name)
  # The caller is responsible for determining when this should be built and
  # how it is installed. See add_ktir_tool for tool executables.
  add_llvm_executable(${ARGV})
endmacro()

# Creates an executable target and adds it to the install target.
macro(add_ktir_tool name)
  cmake_parse_arguments(ARG "DISABLE_INSTALL" "EXPORT_NAME" "" ${ARGN})

  if(NOT KTIR_BUILD_TOOLS)
    # Tools are always included, but not necessarily built.
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_ktir_executable(${name} ${ARG_UNPARSED_ARGUMENTS})

  if(KTIR_BUILD_TOOLS AND NOT ARG_DISABLE_INSTALL AND NOT DISABLE_INSTALL)
    if (DEFINED ARG_EXPORT_NAME)
      set_target_properties(${name} PROPERTIES EXPORT_NAME ${ARG_EXPORT_NAME})
      add_executable("${PROJECT_NAME}::${ARG_EXPORT_NAME}" ALIAS ${name})
    endif()

    install(TARGETS ${name}
      COMPONENT ${name}
      EXPORT ${PROJECT_NAME}
    )

    set_property(GLOBAL APPEND PROPERTY KTIR_EXPORTS ${name})
  endif()
endmacro()

# Creates a library target and adds it to the install target.
function(add_ktir_library name)
  cmake_parse_arguments(ARG "DISABLE_INSTALL" "EXPORT_NAME" "" ${ARGN})

  add_mlir_library(${name} 
    ${ARG_UNPARSED_ARGUMENTS} 
    DISABLE_INSTALL 
    EXCLUDE_FROM_LIBMLIR
  )

  if(NOT ARG_DISABLE_INSTALL)
    add_ktir_library_install(${name} ${ARGN})
  endif()
endfunction()

# Creates a library target and adds it to the install and ktir-capi targets.
function(add_ktir_public_c_api_library name)
  cmake_parse_arguments(ARG "DISABLE_INSTALL" "EXPORT_NAME" "" ${ARGN})

  add_mlir_public_c_api_library(${name}
    ${ARG_UNPARSED_ARGUMENTS} 
    DISABLE_INSTALL
  )

  if(NOT ARG_DISABLE_INSTALL AND NOT DISABLE_INSTALL)
    add_ktir_library_install(${name} ${ARGN})
  endif()

  add_dependencies(ktir-capi ${name})
endfunction()

# Creates a library target and appends it to the KTIR_CONVERSION_LIBS property.
function(add_ktir_conversion_library name)
  add_ktir_library(${ARGV})
  set_property(GLOBAL APPEND PROPERTY KTIR_CONVERSION_LIBS ${name})
endfunction()

# Creates a library target and appends it to the KTIR_DIALECT_LIBS property.
function(add_ktir_dialect_library name)
  add_ktir_library(${ARGV})
  set_property(GLOBAL APPEND PROPERTY KTIR_DIALECT_LIBS ${name})
endfunction()

# Creates a library target and appends it to the KTIR_EXTENSION_LIBS property.
function(add_ktir_extension_library name)
  add_ktir_library(${ARGV})
  set_property(GLOBAL APPEND PROPERTY KTIR_EXTENSION_LIBS ${name})
endfunction()

# Creates an install target for the given library target.
function(add_ktir_library_install name)
  cmake_parse_arguments(ARG "" "EXPORT_NAME" "" ${ARGN})

  if (DEFINED ARG_EXPORT_NAME)
    set_target_properties(${name} PROPERTIES EXPORT_NAME ${ARG_EXPORT_NAME})
    add_library("${PROJECT_NAME}::${ARG_EXPORT_NAME}" ALIAS ${name})
  endif()

  install(TARGETS ${name}
    EXPORT ${PROJECT_NAME}
    COMPONENT ${name}
  )
  
  set_property(GLOBAL APPEND PROPERTY KTIR_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY KTIR_EXPORTS ${name})
endfunction()
