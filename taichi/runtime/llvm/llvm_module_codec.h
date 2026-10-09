#pragma once

#include <memory>

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace llvm {
class raw_ostream;
}

namespace taichi::lang {

enum class LLVMModuleEncoding { assembly, bitcode };

// Recognize LLVM's raw/wrapped bitcode signature. This is not validation;
// malformed payloads are rejected by decode_llvm_module.
LLVMModuleEncoding detect_llvm_module_encoding(llvm::MemoryBufferRef buffer);

// Decode eagerly so the returned module does not retain the input buffer.
// The caller owns the context and must keep it alive for the module's lifetime.
// Container/schema checks and module verification remain with the caller.
llvm::Expected<std::unique_ptr<llvm::Module>> decode_llvm_module(
    llvm::MemoryBufferRef buffer,
    llvm::LLVMContext &context,
    LLVMModuleEncoding encoding);

// Does not mutate the module or change its target and optimization metadata.
void encode_llvm_module(const llvm::Module &module,
                        llvm::raw_ostream &output,
                        LLVMModuleEncoding encoding);

}  // namespace taichi::lang
