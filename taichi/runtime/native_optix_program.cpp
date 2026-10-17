#include "taichi/runtime/native_optix_program.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/common/operation_json.h"
#include "picosha2.h"
#include <algorithm>
#include <set>
#include "taichi/optix/forge_optix_runtime_provider.h"
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
using Json = liong::json::JsonValue;
using namespace operation_json;

uint32_t u32(const Json &v) {
  const auto value = integer(v);
  if (value > UINT32_MAX)
    throw std::invalid_argument("OptiX program integer exceeds uint32 ABI");
  return uint32_t(value);
}
std::string sha256(const Json &v) {
  const auto value = name(v);
  if (value.size() != 64 ||
      value.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument(
        "OptiX program requires a lowercase SHA256 contract");
  return value;
}
struct Entry {
  uint32_t module{0};
  std::string name;
  TiForgeOptixProgramEntry native() const {
    return {module, name.empty() ? nullptr : name.c_str()};
  }
};
struct Group {
  std::string name;
  uint32_t kind;
  Entry entry, any_hit;
};
struct ProgramDescription {
  static constexpr const char *kind = "optix_program";
  static constexpr uint32_t version = 1;
  std::shared_ptr<NativeOptixContext> context;
  std::vector<std::shared_ptr<NativeHostBlobResource>> modules;
  std::vector<std::string> digests;
  std::vector<Group> groups;
  std::string parameter_name, expected;
  uint32_t parameter_size, payload_count, attribute_count, max_trace_depth;
  bool micromaps;
};

Entry entry(const Json &v, uint32_t modules, const char *prefix) {
  if (v.ty == liong::json::L_JSON_NULL)
    return {};
  fields(v, {"module", "name"});
  Entry result{u32(v["module"]), name(v["name"])};
  if (result.module >= modules || result.name.rfind(prefix, 0) != 0)
    throw std::invalid_argument(
        "OptiX program entry module or stage prefix is invalid");
  return result;
}

std::string program_contract(const ProgramDescription &d) {
  picosha2::hash256_one_by_one hash;
  auto number = [&](uint64_t value) {
    uint8_t bytes[8];
    for (size_t i = 0; i < 8; ++i)
      bytes[i] = uint8_t(value >> (8 * i));
    hash.process(bytes, bytes + 8);
  };
  auto text = [&](const std::string &value) {
    number(value.size());
    hash.process(value.begin(), value.end());
  };
  text("forge-optix-program-v1");
  number(d.digests.size());
  for (const auto &digest : d.digests)
    text(digest);
  number(d.groups.size());
  for (const auto &group : d.groups) {
    text(group.name);
    number(group.kind);
    for (const auto &e : {group.entry, group.any_hit}) {
      number(e.module);
      text(e.name);
    }
  }
  text(d.parameter_name);
  for (uint32_t value : {d.parameter_size, d.payload_count, d.attribute_count,
                         d.max_trace_depth, uint32_t(d.micromaps)})
    number(value);
  hash.finish();
  return picosha2::get_hash_hex_string(hash);
}

ProgramDescription parameters(const Json &p) {
  fields(p, {"module_count", "module_sha256", "groups", "parameter_name",
             "parameter_size", "payload_count", "attribute_count",
             "max_trace_depth", "allow_opacity_micromaps", "expected_program"});
  ProgramDescription d;
  const auto count = u32(p["module_count"]);
  if (!count || !p["module_sha256"].is_arr() ||
      p["module_sha256"].arr.inner.size() != count || !p["groups"].is_arr() ||
      p["groups"].arr.inner.empty() ||
      p["groups"].arr.inner.size() > UINT32_MAX)
    throw std::invalid_argument(
        "OptiX program requires complete modules and groups");
  for (const auto &v : p["module_sha256"].arr.inner)
    d.digests.push_back(sha256(v));
  std::set<std::string> names;
  bool raygen = false;
  for (const auto &v : p["groups"].arr.inner) {
    fields(v, {"name", "kind", "entry", "any_hit"});
    Group group;
    group.name = name(v["name"]);
    group.kind = u32(v["kind"]);
    if (!names.insert(group.name).second || group.kind > 2)
      throw std::invalid_argument(
          "OptiX program group name or kind is invalid");
    const char *prefix = group.kind == 0   ? "__raygen__"
                         : group.kind == 1 ? "__miss__"
                                           : "__closesthit__";
    group.entry = entry(v["entry"], count, prefix);
    if (v.obj.inner.count("any_hit"))
      group.any_hit = entry(v["any_hit"], count, "__anyhit__");
    if (group.kind < 2
            ? (group.entry.name.empty() || !group.any_hit.name.empty())
            : (group.entry.name.empty() && group.any_hit.name.empty()))
      throw std::invalid_argument(
          "OptiX program group has incompatible entries");
    raygen |= group.kind == 0;
    d.groups.push_back(std::move(group));
  }
  if (!raygen)
    throw std::invalid_argument("OptiX program requires a raygen group");
  d.parameter_name = name(p["parameter_name"]);
  d.parameter_size = u32(p["parameter_size"]);
  d.payload_count = u32(p["payload_count"]);
  d.attribute_count = u32(p["attribute_count"]);
  d.max_trace_depth = u32(p["max_trace_depth"]);
  d.micromaps = optional_bool(p, "allow_opacity_micromaps");
  // Existing OptiX public ABI register limits, not an application-size cap.
  if (d.payload_count > 32 || d.attribute_count < 2 || d.attribute_count > 8)
    throw std::invalid_argument(
        "OptiX program payload/attribute register counts are invalid");
  if (p.obj.inner.count("expected_program"))
    d.expected = sha256(p["expected_program"]);
  if (!d.expected.empty() && d.expected != program_contract(d))
    throw std::invalid_argument(
        "OptiX program code or layout contract mismatch");
  return d;
}

#ifdef TI_WITH_CUDA
TiForgeOptixProgramApi query_program_api(const TiForgeOptixProviderApi &api) {
  if (!(api.info.features & TI_FORGE_OPTIX_FEATURE_PROGRAMMABLE_PIPELINE) ||
      api.struct_size < offsetof(TiForgeOptixProviderApi, get_program_api) +
                            sizeof(api.get_program_api) ||
      !api.get_program_api)
    throw std::invalid_argument("OptiX adapter lacks programmable pipelines");
  TiForgeOptixProgramApi result{};
  native_optix::check(api, api.get_program_api(sizeof(result), &result),
                      "query program API");
  if (result.struct_size < sizeof(result) || result.revision != 1 ||
      !result.create || !result.destroy || !result.prepare ||
      !result.initialize || !result.launch || !result.destroy_launch ||
      !result.memory || !result.scene_info)
    throw std::invalid_argument(
        "OptiX program API is truncated or incompatible");
  return result;
}

class PreparedProgram final : public NativeOptixProgram {
  struct Lease {
    std::shared_ptr<NativeOptixContext> context;
    TiForgeOptixProgramApi api;
    TiForgeOptixProgram handle{nullptr};
  };

 public:
  explicit PreparedProgram(const ProgramDescription &d)
      : lease_(std::make_unique<Lease>()),
        contract_(program_contract(d)),
        parameter_size_(d.parameter_size) {
    lease_->context = d.context;
    lease_->api = query_program_api(d.context->provider_api());
    for (const auto &group : d.groups)
      kinds_.push_back(group.kind);
  }
  void initialize(const ProgramDescription &d) {
    std::vector<TiForgeOptixProgramModule> modules;
    for (const auto &module : d.modules) {
      const auto &bytes = module->bytes();
      modules.push_back(
          {reinterpret_cast<const char *>(bytes.data()), bytes.size()});
    }
    std::vector<TiForgeOptixProgramGroup> groups;
    for (const auto &g : d.groups)
      groups.push_back({g.kind, g.entry.native(), g.any_hit.native()});
    const TiForgeOptixProgramDesc description{
        sizeof(description),  uint32_t(modules.size()),
        modules.data(),       uint32_t(groups.size()),
        groups.data(),        d.parameter_name.c_str(),
        d.parameter_size,     d.payload_count,
        d.attribute_count,    d.max_trace_depth,
        uint32_t(d.micromaps)};
    native_optix::check(d.context->provider_api(),
                        lease_->api.create(d.context->context_handle(),
                                           &description, &lease_->handle),
                        "create program");
    if (!lease_->handle)
      throw std::runtime_error("OptiX returned a null program");
  }
  ~PreparedProgram() override {
    if (!lease_->handle)
      return;
    if (!native_optix::calls_safe(lease_->context->fault_domain())) {
      lease_.release();
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      const auto result = lease_->api.destroy(lease_->handle);
      if (result != TI_FORGE_OPTIX_SUCCESS) {
        TI_WARN("OptiX program cleanup failed (result {}).", int(result));
        lease_.release();
      }
    } catch (...) {
      lease_.release();
    }
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return lease_->context->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  bool retains_input(const std::string &port) const override {
    return port == "context";
  }
  std::shared_ptr<NativeOptixContext> context() const override {
    return lease_->context;
  }
  const TiForgeOptixProgramApi &program_api() const override {
    return lease_->api;
  }
  TiForgeOptixProgram program_handle() const override {
    return lease_->handle;
  }
  const std::string &contract() const override {
    return contract_;
  }
  uint32_t parameter_bytes() const override {
    return parameter_size_;
  }
  uint32_t group_kind(uint32_t index) const override {
    if (index >= kinds_.size())
      throw std::invalid_argument("OptiX SBT group index is out of range");
    return kinds_[index];
  }

 private:
  std::unique_ptr<Lease> lease_;
  std::string contract_;
  uint32_t parameter_size_;
  std::vector<uint32_t> kinds_;
};

NativeOperationAvailability probe(NativeProviderSession &providers) {
  try {
    const auto runtime = providers.runtime(native_optix::provider_definition());
    const auto api =
        runtime->execution_api<TiForgeOptixExecutionApi>().provider;
    if (!(api.info.features & TI_FORGE_OPTIX_FEATURE_PROGRAMMABLE_PIPELINE) ||
        !api.get_program_api)
      return {};
    query_program_api(api);
    return {true, 0};
  } catch (const NativeProviderError &error) {
    if (error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE ||
        error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE)
      return {};
    throw;
  }
}
#endif
}  // namespace

void register_optix_program_operations(NativeOperationRegistry &registry) {
  registry.define<ProgramDescription>(
      [](Device &device, const ProgramDescription &d) {
        if (d.context->arch() != Arch::cuda || &d.context->device() != &device)
          throw std::invalid_argument(
              "OptiX program requires a local CUDA context");
        for (size_t i = 0; i < d.modules.size(); ++i) {
          const auto &module = d.modules[i];
          const auto &bytes = module->bytes();
          if (module->arch() != Arch::cuda || &module->device() != &device ||
              bytes.empty() ||
              std::find(bytes.begin(), bytes.end(), 0) != bytes.end())
            throw std::invalid_argument(
                "OptiX PTX requires a nonempty local blob without NUL bytes");
          if (picosha2::hash256_hex_string(bytes.begin(), bytes.end()) !=
              d.digests[i])
            throw std::invalid_argument("OptiX program PTX digest mismatch");
        }
      },
      NativeObjectKind::resource);
  registry.register_schema<ProgramDescription>(
      "optix", "program", {"context"},
      [](const Json &p) {
        const auto description = parameters(p);
        return [description](const NativeOperationBindings &bindings) {
          auto result = description;
          result.context =
              bindings.at("context").resource<NativeOptixContext>();
          for (size_t i = 0; i < result.digests.size(); ++i)
            result.modules.push_back(bindings.at("module_" + std::to_string(i))
                                         .resource<NativeHostBlobResource>());
          return result;
        };
      },
      {}, {}, {{"module_count", {"module"}}});
#ifdef TI_WITH_CUDA
  registry.register_context_factory<ProgramDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &c, const ProgramDescription &d) {
        if (!c.fault_domain || !dynamic_cast<cuda::CudaDevice *>(&c.device))
          throw std::invalid_argument(
              "OptiX program requires its CUDA owner and fault domain");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        auto result = std::make_shared<PreparedProgram>(d);
        try {
          result->initialize(d);
          return result;
        } catch (const BackendRuntimeError &error) {
          c.fault_domain->report_backend_error(error, 0);
          throw;
        }
      },
      probe);
#endif
}
}  // namespace taichi::lang
