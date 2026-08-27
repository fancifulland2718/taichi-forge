#include "taichi/runtime/cuda/cuda_artifact_provider.h"
#include "taichi/runtime/cuda/cuda_compileiq_protocol.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SHA256.h"

#include "taichi/common/cleanup.h"
#include "taichi/common/version.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/system/timer.h"
#include "taichi/util/environ_config.h"
#include "taichi/util/io.h"
#include "taichi/util/lock.h"

namespace taichi::lang::cuda {

namespace {

constexpr int kArtifactCacheSchema = 2;
constexpr std::size_t kMaxDiagnosticBytes = 64 * 1024;
constexpr int kArtifactLockDelayMs = 50;
constexpr int kArtifactLockTryCount = 1200;

struct ExternalPtxasIdentity {
  std::string path;
  std::string binary_sha256;
  std::string version;
  int version_major{-1};
  int version_minor{-1};
};

struct CUDAArtifactProviderTelemetry {
  std::atomic<std::uint64_t> external_requests{0};
  std::atomic<std::uint64_t> cache_hits{0};
  std::atomic<std::uint64_t> cache_misses{0};
  std::atomic<std::uint64_t> compile_calls{0};
  std::atomic<std::uint64_t> compile_failures{0};
  std::atomic<std::uint64_t> compile_wall_ns{0};
  std::atomic<std::uint64_t> cubin_loads{0};
  std::atomic<std::uint64_t> cubin_unloads{0};
  std::atomic<std::uint64_t> cubin_bytes{0};
  std::atomic<std::uint64_t> cubin_current_bytes{0};
  std::atomic<std::uint64_t> cubin_peak_bytes{0};
  std::atomic<std::uint64_t> entry_points_loaded{0};
  std::atomic<std::uint64_t> multi_entry_artifacts{0};
  std::atomic<std::uint64_t> advanced_controls_skipped_non_user{0};
  std::atomic<std::uint64_t> advanced_controls_fallbacks{0};
  std::atomic<std::uint64_t> driver_ptx_fallbacks{0};
};

CUDAArtifactProviderTelemetry &provider_telemetry() {
  // Process-lifetime diagnostics must remain valid while CUDA programs are
  // destroyed during static teardown.
  static auto *telemetry = new CUDAArtifactProviderTelemetry();
  return *telemetry;
}

void update_peak(std::atomic<std::uint64_t> &peak, std::uint64_t value) {
  auto observed = peak.load(std::memory_order_relaxed);
  while (observed < value && !peak.compare_exchange_weak(
                                 observed, value, std::memory_order_relaxed)) {
  }
}

std::string env_string(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::string normalized_mode() {
  auto mode = env_string("TI_CUDA_PTXAS_MODE");
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return mode;
}

const char *artifact_role_name(JITModuleRole role) {
  switch (role) {
    case JITModuleRole::runtime:
      return "runtime";
    case JITModuleRole::user_kernel:
      return "user_kernel";
  }
  TI_NOT_IMPLEMENTED
}

std::vector<char> read_binary_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return {};
  }
  const auto end = input.tellg();
  if (end <= 0) {
    return {};
  }
  std::vector<char> bytes(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    return {};
  }
  return bytes;
}

std::string read_diagnostic_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::string text(kMaxDiagnosticBytes, '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(input.gcount()));
  return text;
}

bool write_binary_file(const std::filesystem::path &path,
                       const char *data,
                       std::size_t size) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  output.write(data, static_cast<std::streamsize>(size));
  output.flush();
  return static_cast<bool>(output);
}

std::string sha256_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  TI_ERROR_IF(!input, "Cannot read external ptxas binary '{}'.", path.string());
  llvm::SHA256 hash;
  std::array<char, 256 * 1024> buffer;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      hash.update(
          llvm::StringRef(buffer.data(), static_cast<std::size_t>(count)));
    }
  }
  TI_ERROR_IF(!input.eof(), "Failed while hashing external ptxas binary '{}'.",
              path.string());
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

std::string sha256_bytes(const std::vector<char> &bytes) {
  llvm::SHA256 hash;
  hash.update(llvm::StringRef(bytes.data(), bytes.size()));
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

std::vector<char> read_cached_cubin(const std::filesystem::path &cubin_path) {
  auto cubin = read_binary_file(cubin_path);
  if (cubin.empty()) {
    return {};
  }
  auto hash_path = cubin_path;
  hash_path += ".sha256";
  std::ifstream hash_file(hash_path);
  std::string expected_hash;
  hash_file >> expected_hash;
  if (!hash_file || expected_hash != sha256_bytes(cubin)) {
    TI_WARN("Ignoring incomplete or corrupted CUDA artifact cache entry '{}'.",
            cubin_path.string());
    return {};
  }
  return cubin;
}

void atomic_install(const std::filesystem::path &source,
                    const std::filesystem::path &destination) {
  std::error_code ec;
  std::filesystem::rename(source, destination, ec);
  if (ec) {
    std::filesystem::remove(destination, ec);
    ec.clear();
    std::filesystem::rename(source, destination, ec);
  }
  TI_ERROR_IF(ec, "Cannot install CUDA artifact cache entry '{}': {}",
              destination.string(), ec.message());
}

void install_cached_cubin(const std::filesystem::path &cubin_path,
                          const std::vector<char> &cubin) {
  auto hash_path = cubin_path;
  hash_path += ".sha256";
  auto cubin_install_path = cubin_path;
  cubin_install_path += ".install.tmp";
  auto hash_install_path = hash_path;
  hash_install_path += ".install.tmp";
  auto cleanup = make_cleanup([&]() {
    std::error_code ignored;
    std::filesystem::remove(cubin_install_path, ignored);
    std::filesystem::remove(hash_install_path, ignored);
  });
  const auto hash = sha256_bytes(cubin);
  TI_ERROR_IF(
      !write_binary_file(cubin_install_path, cubin.data(), cubin.size()),
      "Cannot write CUDA artifact cache entry '{}'.",
      cubin_install_path.string());
  TI_ERROR_IF(!write_binary_file(hash_install_path, hash.data(), hash.size()),
              "Cannot write CUDA artifact cache checksum '{}'.",
              hash_install_path.string());
  // Install the payload first. Readers require the matching checksum, so a
  // crash between these renames is observed as a miss rather than a bad cubin.
  atomic_install(cubin_install_path, cubin_path);
  atomic_install(hash_install_path, hash_path);
}

std::tuple<std::string, int, int> query_ptxas_version(
    const std::filesystem::path &ptxas,
    const std::filesystem::path &cache_root) {
  static std::atomic<std::uint64_t> counter{0};
  const auto suffix =
      std::to_string(llvm::sys::Process::getProcessId()) + "." +
      std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
  const auto stdout_path = cache_root / ("ptxas-version." + suffix + ".out");
  const auto stderr_path = cache_root / ("ptxas-version." + suffix + ".err");
  auto cleanup = make_cleanup([&]() {
    std::error_code ignored;
    std::filesystem::remove(stdout_path, ignored);
    std::filesystem::remove(stderr_path, ignored);
  });
  const std::vector<std::string> storage{ptxas.string(), "--version"};
  const std::vector<llvm::StringRef> arguments{storage[0], storage[1]};
  const auto stdout_string = stdout_path.string();
  const auto stderr_string = stderr_path.string();
  std::array<std::optional<llvm::StringRef>, 3> redirects{
      llvm::StringRef(""), llvm::StringRef(stdout_string),
      llvm::StringRef(stderr_string)};
  std::string execution_error;
  bool execution_failed = false;
  const int return_code = llvm::sys::ExecuteAndWait(
      ptxas.string(), arguments, std::nullopt, redirects,
      /*SecondsToWait=*/10, /*MemoryLimit=*/0, &execution_error,
      &execution_failed);
  auto text =
      read_diagnostic_file(stdout_path) + read_diagnostic_file(stderr_path);
  if (execution_failed || return_code != 0) {
    TI_WARN("Could not query external ptxas version: exit={}, error='{}'.",
            return_code, execution_error);
    return {"unknown", -1, -1};
  }
  std::smatch match;
  const std::regex release_pattern(R"(release\s+([0-9]+)\.([0-9]+))",
                                   std::regex::icase);
  if (!std::regex_search(text, match, release_pattern)) {
    TI_WARN("Could not parse external ptxas version output '{}'.", text);
    return {text, -1, -1};
  }
  return {text, std::stoi(match[1].str()), std::stoi(match[2].str())};
}

ExternalPtxasIdentity resolve_ptxas(const std::filesystem::path &cache_root) {
  std::string path = env_string("TI_CUDA_PTXAS_PATH");
  if (path.empty()) {
    auto discovered = llvm::sys::findProgramByName("ptxas");
#if defined(_WIN32)
    if (!discovered) {
      discovered = llvm::sys::findProgramByName("ptxas.exe");
    }
#endif
    TI_ERROR_IF(!discovered,
                "TI_CUDA_PTXAS_MODE=external requires ptxas. Set "
                "TI_CUDA_PTXAS_PATH or add ptxas to PATH.");
    path = *discovered;
  }

  std::error_code ec;
  auto canonical = std::filesystem::weakly_canonical(path, ec);
  std::error_code type_error;
  const bool regular = std::filesystem::is_regular_file(canonical, type_error);
  TI_ERROR_IF(ec || type_error || !regular,
              "TI_CUDA_PTXAS_PATH '{}' is not a regular file.", path);
  const auto canonical_path = canonical.string();

  static std::mutex identity_mutex;
  static std::unordered_map<std::string, ExternalPtxasIdentity> identities;
  std::lock_guard<std::mutex> guard(identity_mutex);
  const auto found = identities.find(canonical_path);
  if (found != identities.end()) {
    return found->second;
  }
  auto [version, version_major, version_minor] =
      query_ptxas_version(canonical, cache_root);
  ExternalPtxasIdentity identity{canonical_path, sha256_file(canonical),
                                 version, version_major, version_minor};
  identities.emplace(canonical_path, identity);
  return identity;
}

std::filesystem::path artifact_cache_root(const CompileConfig &config) {
  auto configured = env_string("TI_CUDA_ARTIFACT_CACHE_PATH");
  std::filesystem::path root =
      configured.empty()
          ? std::filesystem::path(config.offline_cache_file_path) /
                "cuda_artifacts"
          : std::filesystem::path(configured);
  std::error_code ec;
  std::filesystem::create_directories(root, ec);
  TI_ERROR_IF(ec, "Cannot create CUDA artifact cache '{}': {}", root.string(),
              ec.message());
  return root;
}

std::string artifact_cache_key(const CUDAKernelArtifact &artifact,
                               const ExternalPtxasIdentity &ptxas,
                               const CUDAAdvancedControls *controls) {
  llvm::SHA256 hash;
  auto add = [&hash](llvm::StringRef value) {
    hash.update(value);
    hash.update(llvm::StringRef("\0", 1));
  };
  add("taichi-cuda-artifact");
  add(std::to_string(kArtifactCacheSchema));
  add(std::to_string(TI_VERSION_MAJOR));
  add(std::to_string(TI_VERSION_MINOR));
  add(std::to_string(TI_VERSION_PATCH));
  add(artifact_role_name(artifact.role));
  add(artifact.target_identity);
  add(std::to_string(artifact.max_registers));
  add(artifact.fast_math ? "fast_math=1" : "fast_math=0");
  add(std::to_string(artifact.llvm_opt_level));
  add(ptxas.binary_sha256);
  add(ptxas.version);
  if (controls != nullptr) {
    add(controls->sha256);
    add(controls->source_identity);
  } else {
    add("no_advanced_controls");
  }
  hash.update(llvm::StringRef(artifact.payload.data(), artifact.code_size()));
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

std::string target_arch(const CUDAKernelArtifact &artifact) {
  const auto separator = artifact.target_identity.find('|');
  return artifact.target_identity.substr(0, separator);
}

std::vector<char> invoke_ptxas(const CUDAKernelArtifact &artifact,
                               const ExternalPtxasIdentity &ptxas,
                               const CUDAAdvancedControls *controls,
                               const std::filesystem::path &cache_root,
                               const std::string &cache_key) {
  static std::atomic<std::uint64_t> temp_counter{0};
  const auto suffix =
      std::to_string(llvm::sys::Process::getProcessId()) + "." +
      std::to_string(temp_counter.fetch_add(1, std::memory_order_relaxed));
  const auto input_path = cache_root / (cache_key + "." + suffix + ".ptx");
  const auto output_path =
      cache_root / (cache_key + "." + suffix + ".cubin.tmp");
  const auto stdout_path =
      cache_root / (cache_key + "." + suffix + ".stdout.tmp");
  const auto stderr_path =
      cache_root / (cache_key + "." + suffix + ".stderr.tmp");
  auto cleanup = make_cleanup([&]() {
    std::error_code ignored;
    std::filesystem::remove(input_path, ignored);
    std::filesystem::remove(output_path, ignored);
    std::filesystem::remove(stdout_path, ignored);
    std::filesystem::remove(stderr_path, ignored);
  });

  TI_ERROR_IF(!write_binary_file(input_path, artifact.payload.data(),
                                 artifact.code_size()),
              "Cannot write temporary PTX input '{}'.", input_path.string());

  std::vector<std::string> argument_storage{
      ptxas.path,
      input_path.string(),
      "-o",
      output_path.string(),
      "--gpu-name=" + target_arch(artifact),
  };
  if (artifact.max_registers != 0) {
    argument_storage.push_back("--maxrregcount=" +
                               std::to_string(artifact.max_registers));
  }
  if (controls != nullptr) {
    argument_storage.push_back("--apply-controls=" + controls->path.string());
  }
  std::vector<llvm::StringRef> arguments;
  arguments.reserve(argument_storage.size());
  for (const auto &argument : argument_storage) {
    arguments.emplace_back(argument);
  }
  const auto stdout_path_string = stdout_path.string();
  const auto stderr_path_string = stderr_path.string();
  std::array<std::optional<llvm::StringRef>, 3> redirects{
      llvm::StringRef(""), llvm::StringRef(stdout_path_string),
      llvm::StringRef(stderr_path_string)};
  const auto timeout_seconds = static_cast<unsigned>(std::clamp(
      get_environ_config("TI_CUDA_PTXAS_TIMEOUT_SECONDS", 60), 1, 3600));
  std::string execution_error;
  bool execution_failed = false;
  const auto started = Time::get_time();
  provider_telemetry().compile_calls.fetch_add(1, std::memory_order_relaxed);
  const int return_code = llvm::sys::ExecuteAndWait(
      ptxas.path, arguments, std::nullopt, redirects, timeout_seconds,
      /*MemoryLimit=*/0, &execution_error, &execution_failed);
  provider_telemetry().compile_wall_ns.fetch_add(
      static_cast<std::uint64_t>((Time::get_time() - started) * 1.0e9),
      std::memory_order_relaxed);

  const auto stdout_text = read_diagnostic_file(stdout_path);
  const auto stderr_text = read_diagnostic_file(stderr_path);
  if (execution_failed || return_code != 0) {
    provider_telemetry().compile_failures.fetch_add(1,
                                                    std::memory_order_relaxed);
    TI_ERROR(
        "External ptxas failed (exit={}, execution_error='{}', "
        "stdout='{}', stderr='{}').",
        return_code, execution_error, stdout_text, stderr_text);
  }
  auto cubin = read_binary_file(output_path);
  if (cubin.empty()) {
    provider_telemetry().compile_failures.fetch_add(1,
                                                    std::memory_order_relaxed);
    TI_ERROR(
        "External ptxas produced an empty cubin (stdout='{}', "
        "stderr='{}').",
        stdout_text, stderr_text);
  }
  TI_DEBUG(
      "External ptxas compiled CUDA artifact: target={}, bytes={}, "
      "stdout='{}', stderr='{}'",
      target_arch(artifact), cubin.size(), stdout_text, stderr_text);
  return cubin;
}

CUDAKernelArtifact select_external_ptxas_artifact(CUDAKernelArtifact artifact,
                                                  const CompileConfig &config) {
  TI_ERROR_IF(artifact.kind != CUDAArtifactKind::ptx,
              "External ptxas expects a canonical PTX artifact.");
  TI_ERROR_IF(CUDADriver::get_instance().is_musa(),
              "External NVIDIA ptxas is unavailable for the MUSA provider.");
  auto &telemetry = provider_telemetry();
  telemetry.external_requests.fetch_add(1, std::memory_order_relaxed);

  const auto cache_root = artifact_cache_root(config);
  const auto ptxas = resolve_ptxas(cache_root);
  const auto base_cache_key =
      artifact_cache_key(artifact, ptxas, /*controls=*/nullptr);
  const auto advanced_controls_configuration =
      cuda_advanced_controls_configuration_from_environment();
  std::optional<CUDAAdvancedControls> controls;
  if (artifact.role == JITModuleRole::user_kernel) {
    controls = resolve_cuda_advanced_controls(CUDACompileIQProtocolRequest{
        artifact, base_cache_key, cache_root, ptxas.path, ptxas.binary_sha256,
        ptxas.version},
                                               advanced_controls_configuration);
  } else if (advanced_controls_configuration.mode !=
                 CUDAAdvancedControlsMode::baseline ||
             advanced_controls_configuration.nested_tuning_request_rejected) {
    telemetry.advanced_controls_skipped_non_user.fetch_add(
        1, std::memory_order_relaxed);
  }
  if (controls) {
    const bool supports_acf =
        ptxas.version_major > 13 ||
        (ptxas.version_major == 13 && ptxas.version_minor >= 3);
    if (!supports_acf) {
      telemetry.advanced_controls_fallbacks.fetch_add(
          1, std::memory_order_relaxed);
      TI_WARN(
          "CUDA Advanced Controls require ptxas 13.3 or newer; resolved "
          "version is '{}'. Compiling this artifact with baseline ptxas.",
          ptxas.version);
      controls.reset();
    }
  }

  auto materialize = [&](const CUDAAdvancedControls *selected_controls) {
    const auto cache_key =
        selected_controls != nullptr
            ? artifact_cache_key(artifact, ptxas, selected_controls)
            : base_cache_key;
    const auto cubin_path = cache_root / (cache_key + ".cubin");
    auto cubin = read_cached_cubin(cubin_path);
    if (!cubin.empty()) {
      telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
      return cubin;
    }
    const auto lock_path = cache_root / (cache_key + ".lock");
    TI_ERROR_IF(!lock_with_file_handle(lock_path.string(), kArtifactLockDelayMs,
                                       kArtifactLockTryCount),
                "Timed out waiting for CUDA artifact cache lock '{}'.",
                lock_path.string());
    auto unlocker = make_cleanup([&]() {
      if (!unlock_file_handle(lock_path.string())) {
        TI_WARN("Failed to release CUDA artifact cache lock '{}'.",
                lock_path.string());
      }
    });

    cubin = read_cached_cubin(cubin_path);
    if (!cubin.empty()) {
      telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
      telemetry.cache_misses.fetch_add(1, std::memory_order_relaxed);
      cubin = invoke_ptxas(artifact, ptxas, selected_controls, cache_root,
                           cache_key);
      install_cached_cubin(cubin_path, cubin);
    }
    return cubin;
  };

  std::vector<char> cubin;
  if (controls) {
    try {
      cubin = materialize(&*controls);
    } catch (const std::exception &error) {
      telemetry.advanced_controls_fallbacks.fetch_add(
          1, std::memory_order_relaxed);
      TI_WARN(
          "CUDA Advanced Controls failed open for role={} target={}: {}. "
          "Retrying with baseline ptxas.",
          artifact_role_name(artifact.role), artifact.target_identity,
          error.what());
      controls.reset();
    } catch (...) {
      telemetry.advanced_controls_fallbacks.fetch_add(
          1, std::memory_order_relaxed);
      TI_WARN(
          "CUDA Advanced Controls failed open for role={} target={}; "
          "retrying with baseline ptxas.",
          artifact_role_name(artifact.role), artifact.target_identity);
      controls.reset();
    }
  }

  if (cubin.empty()) {
    try {
      cubin = materialize(/*selected_controls=*/nullptr);
    } catch (const std::exception &error) {
      telemetry.driver_ptx_fallbacks.fetch_add(1,
                                               std::memory_order_relaxed);
      TI_WARN(
          "External ptxas failed open for role={} target={}: {}. Loading "
          "canonical PTX with the CUDA driver.",
          artifact_role_name(artifact.role), artifact.target_identity,
          error.what());
      return artifact;
    } catch (...) {
      telemetry.driver_ptx_fallbacks.fetch_add(1,
                                               std::memory_order_relaxed);
      TI_WARN(
          "External ptxas failed open for role={} target={}; loading "
          "canonical PTX with the CUDA driver.",
          artifact_role_name(artifact.role), artifact.target_identity);
      return artifact;
    }
  }

  artifact.kind = CUDAArtifactKind::cubin;
  artifact.payload = std::move(cubin);
  artifact.provider_identity =
      "external_ptxas:" + ptxas.binary_sha256.substr(0, 16);
  if (controls) {
    artifact.provider_identity += ":" + controls->source_identity;
  }
  return artifact;
}

}  // namespace

CUDAKernelArtifact select_cuda_kernel_artifact(CUDAKernelArtifact artifact,
                                               const CompileConfig &config) {
  const auto mode = normalized_mode();
  if (mode.empty() || mode == "driver") {
    return artifact;
  }
  TI_ERROR_IF(mode != "external",
              "Unsupported TI_CUDA_PTXAS_MODE '{}'; expected 'driver' or "
              "'external'.",
              mode);
  return select_external_ptxas_artifact(std::move(artifact), config);
}

std::string cuda_artifact_provider_configuration_identity() {
  llvm::SHA256 hash;
  auto mode = normalized_mode();
  if (mode.empty()) {
    mode = "driver";
  }
  hash.update(llvm::StringRef("TI_CUDA_PTXAS_MODE"));
  hash.update(llvm::StringRef("\0", 1));
  hash.update(llvm::StringRef(mode));
  hash.update(llvm::StringRef("\0", 1));
  if (mode != "external") {
    return llvm::toHex(hash.final(), /*LowerCase=*/true);
  }
  for (const char *name :
       {"TI_CUDA_PTXAS_PATH", "TI_CUDA_ARTIFACT_CACHE_PATH",
        "TI_CUDA_PTXAS_TIMEOUT_SECONDS"}) {
    hash.update(llvm::StringRef(name));
    hash.update(llvm::StringRef("\0", 1));
    hash.update(llvm::StringRef(env_string(name)));
    hash.update(llvm::StringRef("\0", 1));
  }
  const auto controls = cuda_advanced_controls_configuration_from_environment();
  hash.update(llvm::StringRef("CUDA_ADVANCED_CONTROLS_MODE"));
  hash.update(llvm::StringRef("\0", 1));
  hash.update(llvm::StringRef(cuda_advanced_controls_mode_name(controls.mode)));
  hash.update(llvm::StringRef("\0", 1));
  auto add_environment = [&](const char *name, const std::string &value) {
    hash.update(llvm::StringRef(name));
    hash.update(llvm::StringRef("\0", 1));
    hash.update(llvm::StringRef(value));
    hash.update(llvm::StringRef("\0", 1));
  };
  if (controls.mode == CUDAAdvancedControlsMode::apply_explicit_acf) {
    add_environment("TI_CUDA_PTXAS_ACF_PATH", controls.explicit_acf_path);
  } else if (controls.mode == CUDAAdvancedControlsMode::request_tuning) {
    add_environment("TI_CUDA_COMPILEIQ_WORKER", controls.worker_path);
    add_environment("TI_CUDA_COMPILEIQ_PYTHON", controls.python_path);
    add_environment("TI_CUDA_COMPILEIQ_TIMEOUT_SECONDS",
                    env_string("TI_CUDA_COMPILEIQ_TIMEOUT_SECONDS"));
  }
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

void record_cuda_artifact_load(std::size_t entry_count,
                               bool is_cubin,
                               std::size_t bytes) noexcept {
  auto &telemetry = provider_telemetry();
  telemetry.entry_points_loaded.fetch_add(entry_count,
                                          std::memory_order_relaxed);
  if (entry_count > 1) {
    telemetry.multi_entry_artifacts.fetch_add(1, std::memory_order_relaxed);
  }
  if (is_cubin) {
    telemetry.cubin_loads.fetch_add(1, std::memory_order_relaxed);
    telemetry.cubin_bytes.fetch_add(bytes, std::memory_order_relaxed);
    const auto current = telemetry.cubin_current_bytes.fetch_add(
                             bytes, std::memory_order_relaxed) +
                         bytes;
    update_peak(telemetry.cubin_peak_bytes, current);
  }
}

void record_cuda_artifact_unload(bool is_cubin, std::size_t bytes) noexcept {
  if (!is_cubin) {
    return;
  }
  auto &telemetry = provider_telemetry();
  telemetry.cubin_unloads.fetch_add(1, std::memory_order_relaxed);
  const auto before =
      telemetry.cubin_current_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  TI_ASSERT(before >= bytes);
}

CUDAArtifactProviderTelemetrySnapshot
get_cuda_artifact_provider_telemetry_snapshot() {
  auto &telemetry = provider_telemetry();
  return {
      telemetry.external_requests.load(std::memory_order_relaxed),
      telemetry.cache_hits.load(std::memory_order_relaxed),
      telemetry.cache_misses.load(std::memory_order_relaxed),
      telemetry.compile_calls.load(std::memory_order_relaxed),
      telemetry.compile_failures.load(std::memory_order_relaxed),
      telemetry.compile_wall_ns.load(std::memory_order_relaxed),
      telemetry.cubin_loads.load(std::memory_order_relaxed),
      telemetry.cubin_unloads.load(std::memory_order_relaxed),
      telemetry.cubin_bytes.load(std::memory_order_relaxed),
      telemetry.cubin_current_bytes.load(std::memory_order_relaxed),
      telemetry.cubin_peak_bytes.load(std::memory_order_relaxed),
      telemetry.entry_points_loaded.load(std::memory_order_relaxed),
      telemetry.multi_entry_artifacts.load(std::memory_order_relaxed),
      telemetry.advanced_controls_skipped_non_user.load(
          std::memory_order_relaxed),
      telemetry.advanced_controls_fallbacks.load(std::memory_order_relaxed),
      telemetry.driver_ptx_fallbacks.load(std::memory_order_relaxed),
  };
}

}  // namespace taichi::lang::cuda
