#pragma once

#include "taichi/aot/module_loader.h"
#include "taichi/common/virtual_dir.h"

class Runtime;
namespace taichi::lang {
namespace LLVM {
class KernelLauncher;
}
namespace gfx {
class GfxRuntime;
}
}

namespace capi {
bool is_unified_library(const taichi::io::VirtualDir &source);
std::unique_ptr<taichi::lang::aot::Module> load_unified_library(
    ::Runtime &runtime, std::shared_ptr<const taichi::io::VirtualDir> source,
    std::unique_ptr<taichi::lang::LLVM::KernelLauncher> launcher,
    taichi::lang::gfx::GfxRuntime *graphics);
}
