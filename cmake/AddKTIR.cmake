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
  # This is a copy from AddMLIR.cmake, which uses the right targets.
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
  set(GEN_DOC_FILE ${KTIR_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
  )
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(ktir-doc ${output_file}DocGen)
endfunction()

#
# Target creation helpers
#

# Creates an executable target.
macro(add_ktir_executable name)
  # The caller is responsible for determining when this should be built and
  # how it is installed. See add_ktir_tool for tool executables.
  add_llvm_executable(${name} ${ARGN})
endmacro()

# Creates an executable target and adds it to the install target.
macro(add_ktir_tool name)
  cmake_parse_arguments(ARG "DISABLE_INSTALL" "" "" ${ARGN})

  if(NOT KTIR_BUILD_TOOLS)
    # Tools are always included, but not necessarily built.
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_ktir_executable(${name} ${ARG_UNPARSED_ARGUMENTS})

  if(KTIR_BUILD_TOOLS AND NOT ARG_DISABLE_INSTALL AND NOT DISABLE_INSTALL)
    # This is a copy from AddMLIR.cmake, which uses generic LLVM CMake.
    get_target_export_arg(${name} KTIR export_to_ktirtargets)
    install(TARGETS ${name}
      COMPONENT ${name}
      ${export_to_ktirtargets}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

    if(NOT CMAKE_CONFIGURATION_TYPES)
      add_llvm_install_targets(install-${name}
        DEPENDS ${name}
        COMPONENT ${name}
      )
    endif()
    set_property(GLOBAL APPEND PROPERTY KTIR_EXPORTS ${name})
  endif()
endmacro()

# Creates a library target and adds it to the install target.
function(add_ktir_library name)
  cmake_parse_arguments(ARG "DISABLE_INSTALL" "" "" ${ARGN})

  add_mlir_library(${ARGV} DISABLE_INSTALL EXCLUDE_FROM_LIBMLIR)

  if(NOT ARG_DISABLE_INSTALL)
    add_ktir_library_install(${name})
  endif()
endfunction()

# Creates a library target and adds it to the install and ktir-capi targets.
function(add_ktir_public_c_api_library name)
  cmake_parse_arguments(ARG "DISABLE_INSTALL" "" "" ${ARGN})

  add_mlir_public_c_api_library(${ARGV} DISABLE_INSTALL)

  if(NOT ARG_DISABLE_INSTALL AND NOT DISABLE_INSTALL)
    add_ktir_library_install(${name})
  endif()

  add_dependencies(ktir-capi ${name})
endfunction()

# Creates a library target and appends it to the KTIR_CONVERSION_LIBS property.
function(add_ktir_conversion_library name)
  add_ktir_library(${ARGV} DEPENDS ktir-headers)
  set_property(GLOBAL APPEND PROPERTY KTIR_CONVERSION_LIBS ${name})
endfunction()

# Creates a library target and appends it to the KTIR_DIALECT_LIBS property.
function(add_ktir_dialect_library name)
  add_ktir_library(${ARGV} DEPENDS ktir-headers)
  set_property(GLOBAL APPEND PROPERTY KTIR_DIALECT_LIBS ${name})
endfunction()

# Creates a library target and appends it to the KTIR_EXTENSION_LIBS property.
function(add_ktir_extension_library name)
  add_ktir_library(${ARGV} DEPENDS ktir-headers)
  set_property(GLOBAL APPEND PROPERTY KTIR_EXTENSION_LIBS ${name})
endfunction()

# Creates an install target for the given library target.
function(add_ktir_library_install name)
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    # This is a copy from AddMLIR.cmake, which uses generic LLVM CMake.
    get_target_export_arg(${name} KTIR export_to_ktirtargets UMBRELLA ktir-libraries)
    install(TARGETS ${name}
      COMPONENT ${name}
      ${export_to_ktirtargets}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    )

    if (NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
        DEPENDS ${name}
        COMPONENT ${name}
      )
    endif()
  set_property(GLOBAL APPEND PROPERTY KTIR_LIBS ${name})
  endif()
  set_property(GLOBAL APPEND PROPERTY KTIR_EXPORTS ${name})
endfunction()
