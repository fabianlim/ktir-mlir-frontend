configure_lit_site_cfg(
        lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(KTDP_TEST_DEPENDS
  ktdp-opt)

add_lit_testsuite(check-ktdp "Running the KTDP regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${KTDP_TEST_DEPENDS})
set_target_properties(check-ktdp PROPERTIES FOLDER "Tests")

add_dependencies(check-ktir check-ktdp)

add_lit_testsuites(KTDP ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${KTDP_TEST_DEPENDS})
