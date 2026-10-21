#pragma once

#include <map>
#include <string>
#include <vector>

#include "taichi/program/argument_origin.h"

namespace taichi::io {
struct VirtualDir;
}
namespace taichi::lang {
class CallableBase;
}

namespace taichi::lang::aot {

using NamedArgumentOrigins = std::map<std::string, ArgumentOrigins>;

struct ArgumentOriginArtifact {
  std::uint32_t version{1};
  std::vector<ArgumentOrigins> layouts;
  std::map<std::string, std::uint64_t> kernels;
  TI_IO_DEF(version, layouts, kernels);
};

// Metadata only: compiled types, offsets, and value shapes remain authoritative
// in the existing executable and value-parameter artifacts.
ArgumentOrigins canonical_argument_origins(ArgumentOrigins origins);
void validate_argument_origins(const CallableBase &kernel,
                               const ArgumentOrigins &origins);
ArgumentOriginArtifact make_argument_origin_artifact(
    const NamedArgumentOrigins &origins);
NamedArgumentOrigins parse_argument_origin_artifact(
    const liong::json::JsonValue &json);
NamedArgumentOrigins load_argument_origin_artifact(const io::VirtualDir *source);
void write_argument_origin_artifact(const NamedArgumentOrigins &origins,
                                     const std::string &directory);

}  // namespace taichi::lang::aot
