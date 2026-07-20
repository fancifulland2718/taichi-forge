#include "gtest/gtest.h"

#include "taichi/common/dynamic_loader.h"

namespace taichi {

TEST(DynamicLoaderTest, MissingOptionalSymbolDoesNotFailLibraryLoad) {
#ifdef _WIN32
  constexpr const char *kLibrary = "kernel32.dll";
  constexpr const char *kKnownSymbol = "GetCurrentProcessId";
#elif defined(__APPLE__)
  constexpr const char *kLibrary = "/usr/lib/libSystem.B.dylib";
  constexpr const char *kKnownSymbol = "getpid";
#else
  constexpr const char *kLibrary = "libm.so.6";
  constexpr const char *kKnownSymbol = "cos";
#endif
  DynamicLoader loader(kLibrary);
  ASSERT_TRUE(loader.loaded());
  EXPECT_NE(loader.load_function_optional(kKnownSymbol), nullptr);
  EXPECT_EQ(loader.load_function_optional(
                "taichi_deliberately_missing_optional_symbol"),
            nullptr);
  EXPECT_NE(loader.load_function(kKnownSymbol), nullptr);
}

}  // namespace taichi
