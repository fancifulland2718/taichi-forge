#include "taichi/runtime/llvm/llvm_module_codec.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace taichi::lang {

llvm::Expected<std::unique_ptr<llvm::Module>> decode_llvm_module(
    llvm::MemoryBufferRef buffer,
    llvm::LLVMContext &context,
    LLVMModuleEncoding encoding) {
  if (encoding == LLVMModuleEncoding::bitcode) {
    return llvm::parseBitcodeFile(buffer, context);
  }
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssembly(buffer, diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("LLVM module", stream, false);
    return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
  }
  return std::move(module);
}

void encode_llvm_module(const llvm::Module &module,
                        llvm::raw_ostream &output,
                        LLVMModuleEncoding encoding) {
  if (encoding == LLVMModuleEncoding::bitcode) {
    llvm::WriteBitcodeToFile(module, output);
  } else {
    module.print(output, nullptr);
  }
}

}  // namespace taichi::lang
