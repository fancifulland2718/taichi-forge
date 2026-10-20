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

TEST(LLVMKernelPayload, LoadedClonesOutliveSourceReloadAndDestruction) {
  for (auto arch : {Arch::x64, Arch::cuda}) {
    llvm::LLVMContext context;
    LLVM::CompiledKernelData::InternalData original_data;
    original_data.compiled_data.module =
        std::make_unique<llvm::Module>("retained", context);
    LLVM::CompiledKernelData original(arch, std::move(original_data));
    std::stringstream serialized;
    ASSERT_EQ(original.dump(serialized), CompiledKernelData::Err::kNoError);
    const auto bytes = serialized.str();
    std::weak_ptr<llvm::LLVMContext> lifetime;
    std::unique_ptr<CompiledKernelData> clone;
    std::unique_ptr<LLVM::CompiledKernelData::InternalData> copied;
    {
      LLVM::CompiledKernelData loaded;
      ASSERT_EQ(loaded.load(serialized), CompiledKernelData::Err::kNoError);
      const auto &data = loaded.get_internal_data().compiled_data;
      ASSERT_TRUE(data.module_context_owner);
      lifetime = data.module_context_owner;
      clone = loaded.clone();
      copied = std::make_unique<LLVM::CompiledKernelData::InternalData>(
          loaded.get_internal_data());
      EXPECT_EQ(copied->compiled_data.module_context_owner,
                data.module_context_owner);
      std::stringstream reloaded(bytes);
      ASSERT_EQ(loaded.load(reloaded), CompiledKernelData::Err::kNoError);
      EXPECT_NE(data.module_context_owner,
                copied->compiled_data.module_context_owner);
      EXPECT_FALSE(lifetime.expired());
      EXPECT_EQ(clone->check(), CompiledKernelData::Err::kNoError);
    }
    EXPECT_FALSE(lifetime.expired());
    EXPECT_EQ(clone->check(), CompiledKernelData::Err::kNoError);
    std::stringstream exported;
    EXPECT_EQ(clone->dump(exported), CompiledKernelData::Err::kNoError);
    EXPECT_EQ(exported.str(), bytes);
    clone.reset();
    EXPECT_FALSE(lifetime.expired());
    copied.reset();
    EXPECT_TRUE(lifetime.expired());
  }
}

}  // namespace
}  // namespace taichi::lang
#endif
