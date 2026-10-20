#pragma once

#include <map>
#include <string>
#include <vector>

#include "taichi/ir/type_utils.h"

namespace taichi::io {
struct VirtualDir;
}

namespace taichi::lang {
class CallableBase;
}

namespace taichi::lang::aot {

using ValueParameterTypes = std::map<std::vector<int>, DataType>;
using NamedValueParameters = std::map<std::string, ValueParameterTypes>;

// Logical shapes are host binding metadata. They do not replace the compiled
// argument types, offsets or device ABI. The type table is shared by aliases.
struct ValueParameterReference {
  std::vector<int> indices;
  std::uint64_t type{0};
  TI_IO_DEF(indices, type);
};

struct ValueParameterArtifact {
  std::uint32_t version{1};
  std::vector<DataType> types;
  std::map<std::string, std::vector<ValueParameterReference>> kernels;
  TI_IO_DEF(version, types, kernels);
};

// Check the logical-to-lowered relationship without deriving offsets or
// guessing a tensor shape from generated structure member names.
void validate_value_parameter_type(DataType logical, DataType lowered);
DataType compiled_value_parameter_type(const CallableBase &kernel,
                                      const std::vector<int> &indices);
ValueParameterArtifact make_value_parameter_artifact(
    const NamedValueParameters &parameters);
NamedValueParameters parse_value_parameter_artifact(
    const liong::json::JsonValue &json);
NamedValueParameters load_value_parameter_artifact(const io::VirtualDir *source);
void write_value_parameter_artifact(const NamedValueParameters &parameters,
                                    const std::string &directory);

}  // namespace taichi::lang::aot
