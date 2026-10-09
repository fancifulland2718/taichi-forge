#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include <sstream>
#include "llvm/Support/raw_ostream.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/llvm_module_codec.h"
#include "taichi/runtime/llvm/llvm_module_options.h"

namespace taichi::lang {
namespace {

TEST(LLVMKernelPayload, WritesBitcodeAndReadsLegacyText) {
  for (auto arch : {Arch::x64, Arch::cuda}) {
    llvm::LLVMContext context;
    LLVM::CompiledKernelData::InternalData data;
    data.compiled_data.module =
        std::make_unique<llvm::Module>("kernel", context);
    CompileConfig config;
    config.arch = arch;
    config.compile_tier = "full";
    config.fast_math = false;
    LLVMModuleOptions::from_config(config).write(*data.compiled_data.module);
    LLVM::CompiledKernelData original(arch, std::move(data));
    std::stringstream output;
    ASSERT_EQ(original.dump(output), CompiledKernelData::Err::kNoError);
    CompiledKernelDataFile file;
    ASSERT_EQ(file.load(output), CompiledKernelDataFile::Err::kNoError);
    ASSERT_EQ(detect_llvm_module_encoding(
                  llvm::MemoryBufferRef(file.src_code(), "cache")),
              LLVMModuleEncoding::bitcode);
    for (auto encoding :
         {LLVMModuleEncoding::bitcode, LLVMModuleEncoding::assembly}) {
      std::string bytes;
      llvm::raw_string_ostream stream(bytes);
      encode_llvm_module(*original.get_internal_data().compiled_data.module,
                         stream, encoding);
      file.set_src_code(std::move(bytes));
      std::stringstream serialized;
      ASSERT_EQ(file.dump(serialized), CompiledKernelDataFile::Err::kNoError);
      LLVM::CompiledKernelData loaded;
      ASSERT_EQ(loaded.load(serialized), CompiledKernelData::Err::kNoError);
      EXPECT_EQ(loaded.arch(), arch);
      auto options = LLVMModuleOptions::read(
          *loaded.get_internal_data().compiled_data.module, CompileConfig{});
      EXPECT_EQ(options.opt_level, 3);
      EXPECT_FALSE(options.fast_math);
    }
  }
}

TEST(LLVMKernelPayload, RejectsInvalidBitcodeWithValidContainerChecksum) {
  llvm::LLVMContext context;
  LLVM::CompiledKernelData::InternalData data;
  data.compiled_data.module = std::make_unique<llvm::Module>("kernel", context);
  LLVM::CompiledKernelData original(Arch::x64, std::move(data));
  std::stringstream output;
  ASSERT_EQ(original.dump(output), CompiledKernelData::Err::kNoError);
  CompiledKernelDataFile file;
  ASSERT_EQ(file.load(output), CompiledKernelDataFile::Err::kNoError);
  // Keep the signature, remove the bitstream, and recalculate the container
  // checksum. Decoding must fail rather than accepting a header-only module.
  file.set_src_code(file.src_code().substr(0, 4));
  std::stringstream serialized;
  ASSERT_EQ(file.dump(serialized), CompiledKernelDataFile::Err::kNoError);
  LLVM::CompiledKernelData loaded;
  EXPECT_EQ(loaded.load(serialized),
            CompiledKernelData::Err::kParseSrcCodeFailed);
}

}  // namespace
}  // namespace taichi::lang
#endif
