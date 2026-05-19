add_ktir_tool(ktdp-lsp-server ktdp-lsp-server.cpp)
target_link_libraries(ktdp-lsp-server PRIVATE
  KTIRKtdpDialect)

llvm_update_compile_flags(ktdp-lsp-server)
mlir_target_link_libraries(ktdp-lsp-server PRIVATE 
  MLIRLspServerLib
  MLIRRegisterAllDialects)
