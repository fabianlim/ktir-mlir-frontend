# Get the names of all relevant exported KTIR targets.
get_property(ktir_dialect_libs GLOBAL PROPERTY KTIR_DIALECT_LIBS)
get_property(ktir_extension_libs GLOBAL PROPERTY KTIR_EXTENSION_LIBS)

add_ktir_tool(ktir-opt ktir-opt.cpp)
target_link_libraries(ktir-opt PRIVATE
  ${ktir_dialect_libs}
  ${ktir_extension_libs}
)

llvm_update_compile_flags(ktir-opt)
mlir_target_link_libraries(ktir-opt PRIVATE 
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
