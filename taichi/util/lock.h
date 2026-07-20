#pragma once

#include "taichi/common/cleanup.h"
#include "taichi/common/core.h"
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(TI_PLATFORM_WINDOWS)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else  // POSIX
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace taichi {

namespace detail {

inline std::mutex &file_handle_lock_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::unordered_map<std::string, int> &file_handle_locks() {
  static std::unordered_map<std::string, int> locks;
  return locks;
}

inline int open_file_handle_lock(const std::string &path) {
#if defined(TI_PLATFORM_WINDOWS)
  int fd{-1};
  ::_sopen_s(&fd, path.c_str(),
             _O_CREAT | _O_RDWR | _O_BINARY | _O_NOINHERIT, _SH_DENYNO,
             _S_IREAD | _S_IWRITE);
  return fd;
#else
  int fd = ::open(path.c_str(), O_CREAT | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
  if (fd != -1) {
    ::fcntl(fd, F_SETFD, FD_CLOEXEC);
  }
  return fd;
#endif
}

inline bool try_lock_file_descriptor(int fd) {
#if defined(TI_PLATFORM_WINDOWS)
  const auto handle = reinterpret_cast<HANDLE>(::_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  OVERLAPPED overlapped{};
  return ::LockFileEx(handle,
                      LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0,
                      1, 0, &overlapped) != 0;
#else
  return ::flock(fd, LOCK_EX | LOCK_NB) == 0;
#endif
}

inline bool unlock_file_descriptor(int fd) {
#if defined(TI_PLATFORM_WINDOWS)
  const auto handle = reinterpret_cast<HANDLE>(::_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  OVERLAPPED overlapped{};
  return ::UnlockFileEx(handle, 0, 1, 0, &overlapped) != 0;
#else
  return ::flock(fd, LOCK_UN) == 0;
#endif
}

inline bool close_file_descriptor(int fd) {
#if defined(TI_PLATFORM_WINDOWS)
  return ::_close(fd) == 0;
#else
  return ::close(fd) == 0;
#endif
}

}  // namespace detail

inline bool try_lock_with_file(const std::string &path) {
  int fd{-1};
#if defined(TI_PLATFORM_WINDOWS)
  // See
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/sopen-s-wsopen-s
  ::_sopen_s(&fd, path.c_str(), _O_CREAT | _O_EXCL, _SH_DENYNO,
             _S_IREAD | _S_IWRITE);
  if (fd != -1)
    ::_close(fd);
#else
  // See https://www.man7.org/linux/man-pages/man2/open.2.html
  fd = ::open(path.c_str(), O_CREAT | O_EXCL,
              S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
  if (fd != -1)
    ::close(fd);
#endif
  return fd != -1;
}

inline bool unlock_with_file(const std::string &path) {
  return std::remove(path.c_str()) == 0;
}

// Metadata locks must survive process crashes without becoming permanently
// busy. Keep the descriptor open while the lock is owned so the operating
// system releases the advisory lock automatically when a process terminates.
// The lock file itself is intentionally persistent; its existence does not
// indicate ownership.
inline bool try_lock_with_file_handle(const std::string &path) {
  {
    std::lock_guard<std::mutex> guard(detail::file_handle_lock_mutex());
    if (detail::file_handle_locks().count(path) != 0) {
      return false;
    }
  }

  const int fd = detail::open_file_handle_lock(path);
  if (fd == -1) {
    return false;
  }
  if (!detail::try_lock_file_descriptor(fd)) {
    detail::close_file_descriptor(fd);
    return false;
  }

  {
    std::lock_guard<std::mutex> guard(detail::file_handle_lock_mutex());
    const auto [_, inserted] = detail::file_handle_locks().emplace(path, fd);
    if (inserted) {
      return true;
    }
  }

  detail::unlock_file_descriptor(fd);
  detail::close_file_descriptor(fd);
  return false;
}

inline bool unlock_file_handle(const std::string &path) {
  int fd{-1};
  {
    std::lock_guard<std::mutex> guard(detail::file_handle_lock_mutex());
    auto &locks = detail::file_handle_locks();
    const auto found = locks.find(path);
    if (found == locks.end()) {
      return false;
    }
    fd = found->second;
    locks.erase(found);
  }
  const bool unlocked = detail::unlock_file_descriptor(fd);
  const bool closed = detail::close_file_descriptor(fd);
  return unlocked && closed;
}

inline bool lock_with_file_handle(const std::string &path,
                                  int ms_delay = 50,
                                  int try_count = 5) {
  if (try_lock_with_file_handle(path)) {
    return true;
  }
  for (int i = 1; i < try_count; ++i) {
    std::chrono::milliseconds delay{ms_delay};
    std::this_thread::sleep_for(delay);
    if (try_lock_with_file_handle(path)) {
      return true;
    }
  }
  return false;
}

inline bool lock_with_file(const std::string &path,
                           int ms_delay = 50,
                           int try_count = 5) {
  if (try_lock_with_file(path)) {
    return true;
  }
  for (int i = 1; i < try_count; ++i) {
    std::chrono::milliseconds delay{ms_delay};
    std::this_thread::sleep_for(delay);
    if (try_lock_with_file(path)) {
      return true;
    }
  }
  return false;
}

inline RaiiCleanup make_unlocker(const std::string &path) {
  return make_cleanup([path]() {
    if (!unlock_with_file(path)) {
      TI_WARN(
          "Unlock {} failed. You can remove this .lock file manually and try "
          "again.",
          path);
    }
  });
}

}  // namespace taichi
