#pragma once

#include "taichi/ir/pass.h"

namespace taichi::lang {

class Program;

class MakeBlockLocalPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
    bool verbose;
    Program *program{nullptr};
  };
};

}  // namespace taichi::lang
