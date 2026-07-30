llvm_canonicalize_cmake_booleans(KTIR_ENABLE_PYTHON_BINDINGS)

# LLVMConfig.cmake hard-codes LLVM_LIT_ARGS to "-sv"; clear it so the
# ARGS passed explicitly to add_lit_testsuite are the only flags used.
set(LLVM_LIT_ARGS "")

configure_lit_site_cfg(
        lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(KTIR_TEST_DEPENDS
  ktir-opt
  FileCheck
)
if(KTIR_ENABLE_PYTHON_BINDINGS)
  list(APPEND KTIR_TEST_DEPENDS KTIRPythonModules)
endif()

add_lit_testsuite(check-ktir "Running the KTIR regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${KTIR_TEST_DEPENDS}
)
set_target_properties(check-ktir PROPERTIES FOLDER "Tests")

add_lit_testsuites(KTIR
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${KTIR_TEST_DEPENDS}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
)
