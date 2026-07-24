add_ktir_dialect(KTDP ktdp)
add_ktir_doc(KTDP KTDPDialect Dialects/ -gen-dialect-doc -dialect=ktdp)

set(LLVM_TARGET_DEFINITIONS KTDPAttrInterfaces.td)
mlir_tablegen(KTDPAttrInterfaces.hpp.inc -gen-attr-interface-decls)
mlir_tablegen(KTDPAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_ktir_tablegen_target(KTIRKTDPAttrInterfacesIncGen)
add_ktir_doc(KTDPAttrs
  KTDPAttrInterfaces
  Dialects/
  -gen-attr-interface-docs)

set(LLVM_TARGET_DEFINITIONS KTDPAttrs.td)
mlir_tablegen(KTDPEnums.hpp.inc -gen-enum-decls)
mlir_tablegen(KTDPEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(KTDPAttrs.hpp.inc -gen-attrdef-decls -attrdefs-dialect=ktdp)
mlir_tablegen(KTDPAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ktdp)
add_ktir_tablegen_target(KTIRKTDPAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS KTDPPasses.td)
mlir_tablegen(KTDPPasses.hpp.inc -gen-pass-decls -name Ktdp)
add_ktir_tablegen_target(KTIRKTDPPassesIncGen)
