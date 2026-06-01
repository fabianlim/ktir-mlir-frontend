add_ktir_tool(ktir-lsp-server ktir-lsp-server.cpp)
target_link_libraries(ktir-lsp-server PRIVATE
  KTIRKtdpDialect
)

llvm_update_compile_flags(ktir-lsp-server)
mlir_target_link_libraries(ktir-lsp-server PRIVATE 
  MLIRLspServerLib
  MLIRRegisterAllDialects
)
