#include "taichi/common/virtual_dir.h"
#include "taichi/common/miniz.h"

#include <limits>
#include <filesystem>
#include <mutex>
#include <unordered_map>

namespace taichi {
namespace io {

struct FilesystemVirtualDir : public VirtualDir {
  std::string base_dir_;

  explicit FilesystemVirtualDir(const std::string &base_dir)
      : base_dir_(base_dir) {
  }

  static std::unique_ptr<VirtualDir> create(const std::string &base_dir) {
    // Deferred reads must not change target when the caller changes cwd.
    std::string base_dir2 = std::filesystem::absolute(
        base_dir.empty() ? "." : base_dir).lexically_normal().generic_string();
    if (base_dir2.back() != '/') base_dir2 += '/';

    return std::unique_ptr<VirtualDir>(new FilesystemVirtualDir(base_dir2));
  }

  bool get_file_size(const std::string &path, size_t &size) const override {
    std::fstream f(base_dir_ + path,
                   std::ios::in | std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      return false;
    }
    size = f.tellg();
    return true;
  }
  size_t load_file(const std::string &path,
                   void *data,
                   size_t size) const override {
    std::fstream f(base_dir_ + path, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
      return false;
    }

    f.read((char *)data, size);
    size_t n = f.gcount();
    return n;
  }
};
struct ZipArchiveVirtualDir : public VirtualDir {
  // Own compressed bytes, not an eagerly expanded copy of every entry. The
  // caller's .tcm buffer may be released immediately after from_zip().
  std::vector<uint8_t> bytes_;
  mutable mz_zip_archive archive_{};
  mutable std::mutex reader_mutex_;
  std::unordered_map<std::string, std::pair<mz_uint, size_t>> entries_;

  explicit ZipArchiveVirtualDir(std::vector<uint8_t> bytes)
      : bytes_(std::move(bytes)) {
  }
  ~ZipArchiveVirtualDir() override {
    if (archive_.m_pState)
      mz_zip_reader_end(&archive_);
  }

  static std::unique_ptr<VirtualDir> create(const std::string &archive_path) {
    std::fstream f(archive_path,
                   std::ios::in | std::ios::binary | std::ios::ate);
    if (!f.is_open() || f.tellg() < 0)
      return nullptr;
    std::vector<uint8_t> archive_data(static_cast<size_t>(f.tellg()));
    f.seekg(std::ios::beg);
    f.read((char *)archive_data.data(), archive_data.size());
    if (static_cast<size_t>(f.gcount()) != archive_data.size())
      return nullptr;
    return from_bytes(std::move(archive_data));
  }
  static std::unique_ptr<VirtualDir> from_zip(const void *data, size_t size) {
    if (!data || size == 0)
      return nullptr;
    const auto *begin = static_cast<const uint8_t *>(data);
    return from_bytes(std::vector<uint8_t>(begin, begin + size));
  }
  static std::unique_ptr<VirtualDir> from_bytes(std::vector<uint8_t> bytes) {
    auto dir = std::make_unique<ZipArchiveVirtualDir>(std::move(bytes));
    auto *archive = &dir->archive_;
    if (!mz_zip_reader_init_mem(archive, dir->bytes_.data(), dir->bytes_.size(),
                                0)) {
      return nullptr;
    }
    for (mz_uint i = 0; i < mz_zip_reader_get_num_files(archive); ++i) {
      if (mz_zip_reader_is_file_a_directory(archive, i))
        continue;
      mz_zip_archive_file_stat stat;
      if (!mz_zip_reader_file_stat(archive, i, &stat) ||
          stat.m_uncomp_size > std::numeric_limits<size_t>::max())
        return nullptr;
      const auto length = mz_zip_reader_get_filename(archive, i, nullptr, 0);
      if (length == 0)
        return nullptr;
      std::string name(length, '\0');
      if (mz_zip_reader_get_filename(archive, i, name.data(), length) != length)
        return nullptr;
      name.resize(length - 1);
      if (!dir->entries_
               .emplace(std::move(name),
                        std::make_pair(i, size_t(stat.m_uncomp_size)))
               .second)
        return nullptr;
    }
    return dir;
  }

  bool get_file_size(const std::string &path, size_t &size) const override {
    auto it = entries_.find(path);
    if (it == entries_.end()) {
      return false;
    }

    size = it->second.second;
    return true;
  }
  size_t load_file(const std::string &path,
                   void *data,
                   size_t size) const override {
    auto it = entries_.find(path);
    if (it == entries_.end()) {
      return 0;
    }

    const size_t n = std::min(size, it->second.second);
    std::lock_guard<std::mutex> lock(reader_mutex_);
    if (n == it->second.second) {
      return mz_zip_reader_extract_to_mem(&archive_, it->second.first, data, n,
                                          0)
                 ? n
                 : 0;
    }
    struct Prefix {
      void *data;
      size_t size;
    } prefix{data, n};
    // Preserve VirtualDir's prefix-read contract while checking the complete
    // entry's CRC. No temporary uncompressed entry buffer is retained.
    auto write = [](void *opaque, mz_uint64 offset, const void *src,
                    size_t count) -> size_t {
      auto &dst = *static_cast<Prefix *>(opaque);
      if (offset < dst.size) {
        std::memcpy(static_cast<uint8_t *>(dst.data) + size_t(offset), src,
                    std::min(count, dst.size - size_t(offset)));
      }
      return count;
    };
    return mz_zip_reader_extract_to_callback(&archive_, it->second.first, write,
                                             &prefix, 0)
               ? n
               : 0;
  }
};

inline bool is_zip_file(const std::string &path) {
  std::fstream f(path, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    return false;
  }

  // Ensure the file magic matches the Zip format.
  char magic[2];
  f.read(magic, 2);
  size_t n = f.gcount();
  if (n == 2 && magic[0] == 'P' && magic[1] == 'K') {
    return true;
  }

  return false;
}

std::unique_ptr<VirtualDir> VirtualDir::open(const std::string &path) {
  if (is_zip_file(path)) {
    return ZipArchiveVirtualDir::create(path);
  } else {
    // (penguinliong) I wanted to use `std::filesyste::is_directory`. But it
    // seems `<filesystem>` is only supported in MSVC.
    return FilesystemVirtualDir::create(path);
  }
}

std::unique_ptr<VirtualDir> VirtualDir::from_zip(const void *data,
                                                 size_t size) {
  return ZipArchiveVirtualDir::from_zip(data, size);
}
std::unique_ptr<VirtualDir> VirtualDir::from_fs_dir(
    const std::string &base_dir) {
  return FilesystemVirtualDir::create(base_dir);
}

}  // namespace io
}  // namespace taichi
