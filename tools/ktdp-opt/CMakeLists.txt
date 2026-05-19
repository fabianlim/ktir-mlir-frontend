add_ktir_tool(ktdp-opt ktdp-opt.cpp)
target_link_libraries(ktdp-opt PRIVATE
  KTIRKtdpDialect)

llvm_update_compile_flags(ktdp-opt)
mlir_target_link_libraries(ktdp-opt PRIVATE 
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions)
