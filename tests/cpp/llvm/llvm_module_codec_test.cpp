#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "taichi/runtime/llvm/llvm_module_codec.h"
#include "taichi/runtime/llvm/llvm_module_options.h"

namespace taichi::lang {
namespace {

std::string assembly(const llvm::Module &module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  encode_llvm_module(module, stream, LLVMModuleEncoding::assembly);
  return result;
}

TEST(LLVMModuleCodec, PreservesModuleAndOptionsWithoutBorrowingInput) {
  for (auto encoding :
       {LLVMModuleEncoding::assembly, LLVMModuleEncoding::bitcode}) {
    llvm::LLVMContext source_context;
    const char *source = R"(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@values = private constant [3 x i32] [i32 1, i32 7, i32 11]
define i32 @run(i32 %value) {
entry:
  %result = add i32 %value, 7
  ret i32 %result
}
)";
    auto original =
        decode_llvm_module(llvm::MemoryBufferRef(source, "fixture"),
                           source_context, LLVMModuleEncoding::assembly);
    ASSERT_TRUE(bool(original));
    CompileConfig config;
    config.arch = Arch::cuda;
    config.compile_tier = "full";
    config.fast_math = false;
    LLVMModuleOptions::from_config(config).write(**original);
    const auto before = assembly(**original);
    std::string bytes;
    llvm::raw_string_ostream output(bytes);
    encode_llvm_module(**original, output, encoding);
    EXPECT_EQ(assembly(**original), before);

    llvm::LLVMContext destination_context;
    auto loaded = decode_llvm_module(llvm::MemoryBufferRef(bytes, "fixture"),
                                     destination_context, encoding);
    ASSERT_TRUE(bool(loaded));
    bytes.assign(bytes.size(), '\0');
    bytes.clear();
    bytes.shrink_to_fit();
    EXPECT_FALSE(llvm::verifyModule(**loaded, &llvm::errs()));
    EXPECT_EQ(assembly(**loaded), before);
    auto options = LLVMModuleOptions::read(**loaded, CompileConfig{});
    EXPECT_EQ(options.opt_level, 3);
    EXPECT_FALSE(options.fast_math);
    EXPECT_NE((*loaded)->getFunction("run"), nullptr);
  }
}

TEST(LLVMModuleCodec, ReturnsParseErrorsForBothEncodings) {
  for (auto encoding :
       {LLVMModuleEncoding::assembly, LLVMModuleEncoding::bitcode}) {
    llvm::LLVMContext context;
    auto loaded = decode_llvm_module(
        llvm::MemoryBufferRef("this is not LLVM IR", "broken"), context,
        encoding);
    ASSERT_FALSE(bool(loaded));
    EXPECT_FALSE(llvm::toString(loaded.takeError()).empty());
  }
}

}  // namespace
}  // namespace taichi::lang
#endif
