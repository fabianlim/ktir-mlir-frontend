add_ktir_tool(ktir-opt ktir-opt.cpp)
target_link_libraries(ktir-opt PRIVATE
  KTIRKtdpDialect
)

llvm_update_compile_flags(ktir-opt)
mlir_target_link_libraries(ktir-opt PRIVATE 
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
