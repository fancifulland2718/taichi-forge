#pragma once

namespace llvm {
class Module;
}

namespace taichi::lang::LLVM {

// Promote private aggregate words exposed by late CPU helper inlining.
void promote_cpu_private_frames(llvm::Module &module);

}  // namespace taichi::lang::LLVM
