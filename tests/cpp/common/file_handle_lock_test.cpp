#include <cstdio>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"
#include "taichi/util/lock.h"

namespace taichi {

TEST(FileHandleLock, ReusesPersistentLockFile) {
  const std::string lock_path = std::string(std::tmpnam(nullptr)) + ".lock";
  std::ofstream(lock_path).close();

  EXPECT_TRUE(std::filesystem::exists(lock_path));
  EXPECT_TRUE(try_lock_with_file_handle(lock_path));
  EXPECT_FALSE(try_lock_with_file_handle(lock_path));
  EXPECT_TRUE(unlock_file_handle(lock_path));
  EXPECT_TRUE(std::filesystem::exists(lock_path));

  EXPECT_TRUE(try_lock_with_file_handle(lock_path));
  EXPECT_TRUE(unlock_file_handle(lock_path));
  EXPECT_FALSE(unlock_file_handle(lock_path));

  EXPECT_TRUE(std::filesystem::remove(lock_path));
}

}  // namespace taichi
