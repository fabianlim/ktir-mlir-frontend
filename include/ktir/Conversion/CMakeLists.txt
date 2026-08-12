set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name KTIRConversion)
add_ktir_tablegen_target(KTIRConversionPassesIncGen)
