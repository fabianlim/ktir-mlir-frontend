//===- ktir-lsp-server.cpp - KTIR LSP server --------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "Ktdp/KtdpDialect.hpp"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::ktdp::KtdpDialect>();
  mlir::registerAllDialects(registry);

  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}
