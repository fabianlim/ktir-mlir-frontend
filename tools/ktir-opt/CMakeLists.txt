# Get the names of all relevant exported KTIR targets.
get_property(ktir_dialect_libs GLOBAL PROPERTY KTIR_DIALECT_LIBS)
get_property(ktir_extension_libs GLOBAL PROPERTY KTIR_EXTENSION_LIBS)
get_property(ktir_conversion_libs GLOBAL PROPERTY KTIR_CONVERSION_LIBS)

add_ktir_tool(ktir-opt 
    ktir-opt.cpp

  SUPPORT_PLUGINS
)
target_link_libraries(ktir-opt PRIVATE
  ${ktir_dialect_libs}
  ${ktir_extension_libs}
  ${ktir_conversion_libs}
)

mlir_target_link_libraries(ktir-opt PRIVATE
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
llvm_update_compile_flags(ktir-opt)
export_executable_symbols_for_plugins(ktir-opt)
