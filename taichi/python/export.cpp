/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/python/export.h"
#include <cstring>
#include "taichi/common/runtime_contract.h"
#include "taichi/common/interface.h"
#include "taichi/util/io.h"

namespace taichi {

namespace {

void validate_split_runtime_bootstrap() {
#if defined(TI_WITH_SPLIT_PYTHON_RUNTIME)
  ForgeRuntimeBootstrapV1 contract;
  const int status = taichi_forge_runtime_bootstrap_v1(&contract,
                                                        sizeof(contract));
  if (status != 0 || contract.struct_size != sizeof(contract)) {
    throw py::import_error(
        "Taichi Forge native runtime bootstrap failed before pybind "
        "initialization");
  }
  if (contract.manifest_schema_version !=
          kForgeContractManifestSchemaVersion ||
      contract.native_abi_revision != kForgeNativeAbiRevision) {
    throw py::import_error(
        "Taichi Forge native runtime ABI does not match this Python shim");
  }
  const auto *compiler_abi_end = static_cast<const char *>(
      std::memchr(contract.compiler_abi, '\0', sizeof(contract.compiler_abi)));
  if (compiler_abi_end == nullptr) {
    throw py::import_error(
        "Taichi Forge native runtime returned an invalid compiler ABI");
  }
  const std::string runtime_compiler_abi(
      contract.compiler_abi,
      static_cast<std::size_t>(compiler_abi_end - contract.compiler_abi));
  const std::string shim_compiler_abi = forge_compiler_abi_identity();
  if (runtime_compiler_abi != shim_compiler_abi) {
    throw py::import_error(
        "Taichi Forge native runtime and Python shim use incompatible C++ "
        "compiler ABIs");
  }
#endif
}

}  // namespace

PYBIND11_MODULE(taichi_python, m) {
  validate_split_runtime_bootstrap();
  m.doc() = "taichi_python";

  for (auto &kv : InterfaceHolder::get_instance()->methods) {
    kv.second(&m);
  }

  export_lang(m);
  export_math(m);
  export_misc(m);
  export_visual(m);
  export_ggui(m);
}

}  // namespace taichi
