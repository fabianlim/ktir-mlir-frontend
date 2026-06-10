# Get the names of all relevant exported KTIR targets.
get_property(ktir_dialect_libs GLOBAL PROPERTY KTIR_DIALECT_LIBS)

add_ktir_tool(ktir-lsp-server ktir-lsp-server.cpp)
target_link_libraries(ktir-lsp-server PRIVATE
  ${ktir_dialect_libs}
)

llvm_update_compile_flags(ktir-lsp-server)
mlir_target_link_libraries(ktir-lsp-server PRIVATE 
  MLIRLspServerLib
  MLIRRegisterAllDialects
)
