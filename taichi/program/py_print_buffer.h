#pragma once

#include <sstream>
#include <iostream>

#include "taichi/common/platform_macros.h"

namespace taichi {

// PythonPrintBuffer holds the logs printed from kernel before sending them back
// to python. The name could be a bit misleading, as it is really just a string
// buffer, and can be used without Python.
struct PythonPrintBuffer {
  std::stringstream ss;
  bool enabled{false};

  template <typename T>
  PythonPrintBuffer &operator<<(const T &t) {
    if (enabled)
      ss << t;
    else
      std::cout << t;
    return *this;
  }
  std::string pop_content() {
    auto ret = ss.str();
    ss = std::stringstream();
    return ret;
  }
};

extern TI_DLL_EXPORT PythonPrintBuffer py_cout;

}  // namespace taichi
