#include <algorithm>
#include "gtest/gtest.h"
#include "taichi/common/miniz.h"
#include "taichi/common/virtual_dir.h"

namespace taichi::io {
namespace {

std::vector<uint8_t> archive_bytes(int level = 0) {
  mz_zip_archive zip{};
  if (!mz_zip_writer_init_heap(&zip, 0, 0))
    return {};
  const std::string payload(8192, 'x');
  mz_zip_writer_add_mem(&zip, "large", payload.data(), payload.size(), level);
  mz_zip_writer_add_mem(&zip, "small", "hello", 5, level);
  void *data = nullptr;
  size_t size = 0;
  mz_zip_writer_finalize_heap_archive(&zip, &data, &size);
  auto *begin = static_cast<uint8_t *>(data);
  std::vector<uint8_t> result(begin, begin + size);
  mz_free(data);
  mz_zip_writer_end(&zip);
  return result;
}

TEST(VirtualDir, ZipOwnsCompressedSourceAndSupportsPrefixReads) {
  for (int level : {0, int(MZ_BEST_COMPRESSION)}) {
    auto bytes = archive_bytes(level);
    ASSERT_FALSE(bytes.empty());
    auto dir = VirtualDir::from_zip(bytes.data(), bytes.size());
    ASSERT_NE(dir, nullptr);
    std::vector<uint8_t>().swap(bytes);
    size_t size = 0;
    ASSERT_TRUE(dir->get_file_size("large", size));
    EXPECT_EQ(size, 8192);
    std::vector<char> full;
    ASSERT_TRUE(dir->load_file("small", full));
    EXPECT_EQ(std::string(full.begin(), full.end()), "hello");
    char prefix[3]{};
    EXPECT_EQ(dir->load_file("small", prefix, sizeof(prefix)), 3);
    EXPECT_EQ(std::string(prefix, 3), "hel");
    ASSERT_TRUE(dir->load_file("large", full));
    EXPECT_EQ(std::string(full.begin(), full.end()), std::string(8192, 'x'));
  }
}

TEST(VirtualDir, ZipValidatesOnlyRequestedPayloads) {
  auto bytes = archive_bytes();
  const std::string marker(32, 'x');
  auto it =
      std::search(bytes.begin(), bytes.end(), marker.begin(), marker.end());
  ASSERT_NE(it, bytes.end());
  *it =
      'y';  // Keep the central directory intact but invalidate one entry's CRC.
  auto dir = VirtualDir::from_zip(bytes.data(), bytes.size());
  ASSERT_NE(dir, nullptr);
  size_t size;
  EXPECT_TRUE(dir->get_file_size("large", size));
  std::vector<char> out;
  EXPECT_TRUE(dir->load_file("small", out));
  EXPECT_FALSE(dir->load_file("large", out));
  EXPECT_FALSE(dir->load_file("missing", out));
}

}  // namespace
}  // namespace taichi::io
