#include "taichi/runtime/cuda/cuda_compileiq_protocol.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cwchar>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"

#include "taichi/common/cleanup.h"
#include "taichi/common/json.h"
#include "taichi/common/platform_macros.h"
#if defined(TI_PLATFORM_WINDOWS)
#include "taichi/platform/windows/windows.h"
#elif defined(TI_PLATFORM_OSX)
#include <crt_externs.h>
#else
extern char **environ;
#endif
#include "taichi/system/timer.h"
#include "taichi/util/environ_config.h"
#include "taichi/util/lock.h"

namespace taichi::lang::cuda {

namespace {

constexpr int kCompileIQProtocolSchema = 2;
constexpr int kWorkerLockDelayMs = 50;
constexpr int kWorkerLockTryCount = 1200;
constexpr std::size_t kMaxWorkerLogBytes = 64 * 1024;
constexpr char kCompileIQActiveRequestEnvironment[] =
    "TI_CUDA_COMPILEIQ_ACTIVE_REQUEST";

struct ProtocolTelemetry {
  std::atomic<std::uint64_t> requests{0};
  std::atomic<std::uint64_t> cache_hits{0};
  std::atomic<std::uint64_t> worker_calls{0};
  std::atomic<std::uint64_t> worker_failures{0};
  std::atomic<std::uint64_t> worker_wall_ns{0};
  std::atomic<std::uint64_t> acf_responses{0};
  std::atomic<std::uint64_t> pass_responses{0};
  std::atomic<std::uint64_t> fail_open_responses{0};
  std::atomic<std::uint64_t> nested_requests_rejected{0};
};

ProtocolTelemetry &protocol_telemetry() {
  static auto *telemetry = new ProtocolTelemetry();
  return *telemetry;
}

std::string env_string(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::filesystem::path regular_file(const std::string &configured,
                                   const char *setting_name) {
  std::error_code ec;
  auto canonical = std::filesystem::weakly_canonical(configured, ec);
  std::error_code type_error;
  const bool regular = std::filesystem::is_regular_file(canonical, type_error);
  TI_ERROR_IF(ec || type_error || !regular, "{} '{}' is not a regular file.",
              setting_name, configured);
  return canonical;
}

std::vector<char> read_binary(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return {};
  }
  const auto end = input.tellg();
  if (end <= 0) {
    return {};
  }
  std::vector<char> result(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  return input ? result : std::vector<char>{};
}

std::string read_text(const std::filesystem::path &path,
                      std::size_t maximum_bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::string result(maximum_bytes, '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  result.resize(static_cast<std::size_t>(input.gcount()));
  return result;
}

bool write_binary(const std::filesystem::path &path,
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

std::string sha256_bytes(const std::vector<char> &bytes) {
  llvm::SHA256 hash;
  hash.update(llvm::StringRef(bytes.data(), bytes.size()));
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

std::string sha256_file(const std::filesystem::path &path) {
  auto bytes = read_binary(path);
  TI_ERROR_IF(bytes.empty(), "Cannot hash empty or unreadable file '{}'.",
              path.string());
  return sha256_bytes(bytes);
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
  TI_ERROR_IF(ec, "Cannot install CompileIQ protocol cache entry '{}': {}",
              destination.string(), ec.message());
}

void install_verified_blob(const std::filesystem::path &destination,
                           const std::vector<char> &bytes) {
  auto checksum_path = destination;
  checksum_path += ".sha256";
  auto blob_tmp = destination;
  blob_tmp += ".install.tmp";
  auto checksum_tmp = checksum_path;
  checksum_tmp += ".install.tmp";
  auto cleanup = make_cleanup([&]() {
    std::error_code ignored;
    std::filesystem::remove(blob_tmp, ignored);
    std::filesystem::remove(checksum_tmp, ignored);
  });
  const auto checksum = sha256_bytes(bytes);
  TI_ERROR_IF(!write_binary(blob_tmp, bytes.data(), bytes.size()),
              "Cannot write CompileIQ protocol cache entry '{}'.",
              blob_tmp.string());
  TI_ERROR_IF(!write_binary(checksum_tmp, checksum.data(), checksum.size()),
              "Cannot write CompileIQ protocol cache checksum '{}'.",
              checksum_tmp.string());
  atomic_install(blob_tmp, destination);
  atomic_install(checksum_tmp, checksum_path);
}

std::optional<CUDAAdvancedControls> read_cached_acf(
    const std::filesystem::path &acf_path,
    const std::string &source_identity) {
  auto bytes = read_binary(acf_path);
  if (bytes.empty()) {
    return std::nullopt;
  }
  auto checksum_path = acf_path;
  checksum_path += ".sha256";
  std::ifstream checksum_file(checksum_path);
  std::string expected;
  checksum_file >> expected;
  const auto actual = sha256_bytes(bytes);
  if (!checksum_file || expected != actual) {
    return std::nullopt;
  }
  return CUDAAdvancedControls{acf_path, actual, source_identity};
}

bool read_cached_pass(const std::filesystem::path &pass_path) {
  return read_text(pass_path, 16) == "pass\n";
}

bool environment_entry_has_name(const std::string &entry, const char *name) {
  const auto separator = entry.find('=');
  if (separator == std::string::npos || separator != std::strlen(name)) {
    return false;
  }
#if defined(TI_PLATFORM_WINDOWS)
  return std::equal(entry.begin(), entry.begin() + separator, name,
                    [](unsigned char lhs, unsigned char rhs) {
                      return std::tolower(lhs) == std::tolower(rhs);
                    });
#else
  return entry.compare(0, separator, name) == 0;
#endif
}

std::vector<std::string> worker_environment() {
  std::vector<std::string> result;
#if defined(TI_PLATFORM_WINDOWS)
  auto *block = GetEnvironmentStringsW();
  TI_ERROR_IF(block == nullptr,
              "Cannot capture the environment for the CompileIQ worker.");
  auto release = make_cleanup([&]() { FreeEnvironmentStringsW(block); });
  for (auto *entry = block; *entry != L'\0'; entry += std::wcslen(entry) + 1) {
    llvm::SmallVector<char, 256> utf8;
    const auto length = std::wcslen(entry);
    TI_ERROR_IF(llvm::sys::windows::UTF16ToUTF8(entry, length, utf8),
                "Cannot encode a CompileIQ worker environment entry.");
    result.emplace_back(utf8.begin(), utf8.end());
  }
#elif defined(TI_PLATFORM_OSX)
  for (auto **entry = *_NSGetEnviron(); entry != nullptr && *entry != nullptr;
       ++entry) {
    result.emplace_back(*entry);
  }
#else
  for (auto **entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    result.emplace_back(*entry);
  }
#endif
  result.erase(
      std::remove_if(result.begin(), result.end(), [](const std::string &entry) {
        return environment_entry_has_name(
            entry, kCompileIQActiveRequestEnvironment);
      }),
      result.end());
  result.emplace_back(std::string(kCompileIQActiveRequestEnvironment) + "=1");
  return result;
}

bool read_cached_fail_open(const std::filesystem::path &path) {
  return read_text(path, 32) == "fail_open\n";
}

void install_marker(const std::filesystem::path &path,
                    llvm::StringRef marker) {
  auto temporary = path;
  temporary += ".install.tmp";
  auto cleanup = make_cleanup([&]() {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
  });
  TI_ERROR_IF(!write_binary(temporary, marker.data(), marker.size()),
              "Cannot write CompileIQ marker '{}'.", temporary.string());
  atomic_install(temporary, path);
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

std::string worker_cache_key(const CUDACompileIQProtocolRequest &request,
                             const std::filesystem::path &worker,
                             const std::filesystem::path &python) {
  llvm::SHA256 hash;
  auto add = [&hash](llvm::StringRef value) {
    hash.update(value);
    hash.update(llvm::StringRef("\0", 1));
  };
  add("taichi-compileiq-worker");
  add(std::to_string(kCompileIQProtocolSchema));
  add(request.base_artifact_key);
  add(worker.string());
  add(sha256_file(worker));
  if (!python.empty()) {
    add(python.string());
    add(sha256_file(python));
  }
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

std::string make_request_json(const CUDACompileIQProtocolRequest &request,
                              const std::filesystem::path &ptx_path) {
  liong::json::JsonObject root;
  root.inner.emplace("schema_version",
                     liong::json::JsonValue(kCompileIQProtocolSchema));
  root.inner.emplace("kind",
                     liong::json::JsonValue("taichi_cuda_compileiq_request"));
  root.inner.emplace("artifact_key",
                     liong::json::JsonValue(request.base_artifact_key));
  root.inner.emplace("ptx_path", liong::json::JsonValue(ptx_path.string()));
  root.inner.emplace("target",
                     liong::json::JsonValue(request.artifact.target_identity));
  root.inner.emplace("provider",
                     liong::json::JsonValue("taichi_forge_cuda_artifact_v1"));
  root.inner.emplace(
      "artifact_role",
      liong::json::JsonValue(artifact_role_name(request.artifact.role)));

  liong::json::JsonArray entries;
  for (const auto &entry : request.artifact.entry_names) {
    entries.inner.emplace_back(entry);
  }
  root.inner.emplace("entry_names", liong::json::JsonValue(std::move(entries)));

  liong::json::JsonObject options;
  options.inner.emplace("max_registers",
                        liong::json::JsonValue(request.artifact.max_registers));
  options.inner.emplace("fast_math",
                        liong::json::JsonValue(request.artifact.fast_math));
  options.inner.emplace("llvm_opt_level", liong::json::JsonValue(
                                              request.artifact.llvm_opt_level));
  root.inner.emplace("options", liong::json::JsonValue(std::move(options)));

  liong::json::JsonObject ptxas;
  ptxas.inner.emplace("path", liong::json::JsonValue(request.ptxas_path));
  ptxas.inner.emplace("sha256", liong::json::JsonValue(request.ptxas_sha256));
  ptxas.inner.emplace("version", liong::json::JsonValue(request.ptxas_version));
  root.inner.emplace("ptxas", liong::json::JsonValue(std::move(ptxas)));
  return liong::json::print(liong::json::JsonValue(std::move(root)));
}

const liong::json::JsonValue &required_field(
    const liong::json::JsonValue &object,
    const char *name) {
  TI_ERROR_IF(!object.is_obj(), "CompileIQ worker response must be an object.");
  const auto found = object.obj.inner.find(name);
  TI_ERROR_IF(found == object.obj.inner.end(),
              "CompileIQ worker response is missing '{}'.", name);
  return found->second;
}

std::optional<CUDAAdvancedControls> invoke_worker(
    const CUDACompileIQProtocolRequest &request,
    const std::filesystem::path &worker,
    const std::filesystem::path &python,
    const std::string &cache_key) {
  static std::atomic<std::uint64_t> temp_counter{0};
  const auto suffix =
      std::to_string(llvm::sys::Process::getProcessId()) + "." +
      std::to_string(temp_counter.fetch_add(1, std::memory_order_relaxed));
  const auto prefix = request.cache_root / (cache_key + "." + suffix);
  auto ptx_path = prefix;
  ptx_path += ".ptx";
  auto request_path = prefix;
  request_path += ".request.json";
  auto response_path = prefix;
  response_path += ".response.json";
  auto stdout_path = prefix;
  stdout_path += ".stdout.tmp";
  auto stderr_path = prefix;
  stderr_path += ".stderr.tmp";
  auto cleanup = make_cleanup([&]() {
    std::error_code ignored;
    std::filesystem::remove(ptx_path, ignored);
    std::filesystem::remove(request_path, ignored);
    std::filesystem::remove(response_path, ignored);
    std::filesystem::remove(stdout_path, ignored);
    std::filesystem::remove(stderr_path, ignored);
  });
  TI_ERROR_IF(!write_binary(ptx_path, request.artifact.payload.data(),
                            request.artifact.code_size()),
              "Cannot write CompileIQ worker PTX '{}'.", ptx_path.string());
  const auto request_json = make_request_json(request, ptx_path);
  TI_ERROR_IF(
      !write_binary(request_path, request_json.data(), request_json.size()),
      "Cannot write CompileIQ worker request '{}'.", request_path.string());

  const auto program = python.empty() ? worker.string() : python.string();
  std::vector<std::string> argument_storage{program};
  if (!python.empty()) {
    argument_storage.push_back(worker.string());
  }
  argument_storage.insert(argument_storage.end(),
                          {"--request", request_path.string(), "--response",
                           response_path.string()});
  std::vector<llvm::StringRef> arguments;
  arguments.reserve(argument_storage.size());
  for (const auto &argument : argument_storage) {
    arguments.emplace_back(argument);
  }
  const auto environment_storage = worker_environment();
  std::vector<llvm::StringRef> environment;
  environment.reserve(environment_storage.size());
  for (const auto &entry : environment_storage) {
    environment.emplace_back(entry);
  }
  const llvm::ArrayRef<llvm::StringRef> environment_ref(environment);
  const auto stdout_string = stdout_path.string();
  const auto stderr_string = stderr_path.string();
  std::array<std::optional<llvm::StringRef>, 3> redirects{
      llvm::StringRef(""), llvm::StringRef(stdout_string),
      llvm::StringRef(stderr_string)};
  const auto timeout = static_cast<unsigned>(std::clamp(
      get_environ_config("TI_CUDA_COMPILEIQ_TIMEOUT_SECONDS", 3600), 1, 86400));
  std::string execution_error;
  bool execution_failed = false;
  auto &telemetry = protocol_telemetry();
  telemetry.worker_calls.fetch_add(1, std::memory_order_relaxed);
  const auto started = Time::get_time();
  const int return_code = llvm::sys::ExecuteAndWait(
      program, arguments, environment_ref, redirects, timeout,
      /*MemoryLimit=*/0, &execution_error, &execution_failed);
  telemetry.worker_wall_ns.fetch_add(
      static_cast<std::uint64_t>((Time::get_time() - started) * 1.0e9),
      std::memory_order_relaxed);
  const auto stdout_text = read_text(stdout_path, kMaxWorkerLogBytes);
  const auto stderr_text = read_text(stderr_path, kMaxWorkerLogBytes);
  TI_ERROR_IF(execution_failed || return_code != 0,
              "CompileIQ worker failed (exit={}, execution_error='{}', "
              "stdout='{}', stderr='{}').",
              return_code, execution_error, stdout_text, stderr_text);

  const auto response_text = read_text(response_path, kMaxWorkerLogBytes);
  TI_ERROR_IF(response_text.empty(),
              "CompileIQ worker did not write a response manifest.");
  auto response = liong::json::parse(response_text);
  const auto &schema = required_field(response, "schema_version");
  TI_ERROR_IF(
      !schema.is_num() || static_cast<int>(schema) != kCompileIQProtocolSchema,
      "CompileIQ worker response schema mismatch.");
  const auto &status_value = required_field(response, "status");
  TI_ERROR_IF(!status_value.is_str(),
              "CompileIQ worker response status must be a string.");
  const auto &status = static_cast<const std::string &>(status_value);
  if (status == "pass") {
    return std::nullopt;
  }
  TI_ERROR_IF(status != "ok",
              "CompileIQ worker returned unsupported status '{}'.", status);
  const auto &acf_value = required_field(response, "acf_path");
  TI_ERROR_IF(!acf_value.is_str(),
              "CompileIQ worker acf_path must be a string.");
  const auto acf_path = regular_file(
      static_cast<const std::string &>(acf_value), "worker acf_path");
  auto bytes = read_binary(acf_path);
  TI_ERROR_IF(bytes.empty(), "CompileIQ worker returned an empty ACF '{}'.",
              acf_path.string());
  const auto actual_hash = sha256_bytes(bytes);
  const auto &declared = required_field(response, "acf_sha256");
  TI_ERROR_IF(!declared.is_str() ||
                  static_cast<const std::string &>(declared) != actual_hash,
              "CompileIQ worker ACF checksum mismatch.");
  const auto cached_path = request.cache_root / (cache_key + ".compileiq.acf");
  install_verified_blob(cached_path, bytes);
  return CUDAAdvancedControls{cached_path, actual_hash,
                              "compileiq_worker:" + cache_key.substr(0, 16)};
}

std::optional<CUDAAdvancedControls> resolve_worker_controls(
    const CUDACompileIQProtocolRequest &request,
    const std::string &configured_worker,
    const std::string &configured_python) {
  const auto worker =
      regular_file(configured_worker, "TI_CUDA_COMPILEIQ_WORKER");
  const auto python =
      configured_python.empty()
          ? std::filesystem::path()
          : regular_file(configured_python, "TI_CUDA_COMPILEIQ_PYTHON");
  const auto key = worker_cache_key(request, worker, python);
  const auto acf_path = request.cache_root / (key + ".compileiq.acf");
  const auto pass_path = request.cache_root / (key + ".compileiq.pass");
  const auto fail_open_path =
      request.cache_root / (key + ".compileiq.fail_open");
  const auto source_identity = "compileiq_worker:" + key.substr(0, 16);
  auto &telemetry = protocol_telemetry();
  if (auto cached = read_cached_acf(acf_path, source_identity)) {
    telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
    telemetry.acf_responses.fetch_add(1, std::memory_order_relaxed);
    return cached;
  }
  if (read_cached_pass(pass_path)) {
    telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
    telemetry.pass_responses.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }
  if (read_cached_fail_open(fail_open_path)) {
    telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
    telemetry.fail_open_responses.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }

  const auto lock_path = request.cache_root / (key + ".compileiq.lock");
  TI_ERROR_IF(!lock_with_file_handle(lock_path.string(), kWorkerLockDelayMs,
                                     kWorkerLockTryCount),
              "Timed out waiting for CompileIQ worker cache lock '{}'.",
              lock_path.string());
  auto unlocker = make_cleanup([&]() {
    if (!unlock_file_handle(lock_path.string())) {
      TI_WARN("Failed to release CompileIQ worker cache lock '{}'.",
              lock_path.string());
    }
  });
  if (auto cached = read_cached_acf(acf_path, source_identity)) {
    telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
    telemetry.acf_responses.fetch_add(1, std::memory_order_relaxed);
    return cached;
  }
  if (read_cached_pass(pass_path)) {
    telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
    telemetry.pass_responses.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }

  if (read_cached_fail_open(fail_open_path)) {
    telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
    telemetry.fail_open_responses.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }

  std::optional<CUDAAdvancedControls> controls;
  try {
    controls = invoke_worker(request, worker, python, key);
  } catch (const std::exception &error) {
    telemetry.worker_failures.fetch_add(1, std::memory_order_relaxed);
    telemetry.fail_open_responses.fetch_add(1, std::memory_order_relaxed);
    TI_WARN("CompileIQ worker failed open for artifact {}: {}",
            request.base_artifact_key, error.what());
    install_marker(fail_open_path, "fail_open\n");
    return std::nullopt;
  } catch (...) {
    telemetry.worker_failures.fetch_add(1, std::memory_order_relaxed);
    telemetry.fail_open_responses.fetch_add(1, std::memory_order_relaxed);
    TI_WARN("CompileIQ worker failed open for artifact {}",
            request.base_artifact_key);
    install_marker(fail_open_path, "fail_open\n");
    return std::nullopt;
  }
  if (controls) {
    telemetry.acf_responses.fetch_add(1, std::memory_order_relaxed);
    return controls;
  }
  install_marker(pass_path, "pass\n");
  telemetry.pass_responses.fetch_add(1, std::memory_order_relaxed);
  return std::nullopt;
}

}  // namespace

CUDAAdvancedControlsConfiguration
cuda_advanced_controls_configuration_from_environment() {
  CUDAAdvancedControlsConfiguration configuration;
  configuration.explicit_acf_path = env_string("TI_CUDA_PTXAS_ACF_PATH");
  configuration.worker_path = env_string("TI_CUDA_COMPILEIQ_WORKER");
  configuration.python_path = env_string("TI_CUDA_COMPILEIQ_PYTHON");
  const bool active_request =
      env_string(kCompileIQActiveRequestEnvironment) == "1";

  if (!active_request) {
    TI_ERROR_IF(!configuration.explicit_acf_path.empty() &&
                    !configuration.worker_path.empty(),
                "TI_CUDA_PTXAS_ACF_PATH and TI_CUDA_COMPILEIQ_WORKER are "
                "mutually exclusive.");
  } else if (!configuration.worker_path.empty()) {
    // A worker may compile baseline or explicit-ACF candidates, but it must
    // never recursively request another tuning worker. An explicit ACF wins
    // over the inherited worker setting for an outer search candidate.
    configuration.nested_tuning_request_rejected = true;
  }

  if (!configuration.explicit_acf_path.empty()) {
    configuration.mode = CUDAAdvancedControlsMode::apply_explicit_acf;
  } else if (!configuration.worker_path.empty() && !active_request) {
    configuration.mode = CUDAAdvancedControlsMode::request_tuning;
  }
  return configuration;
}

const char *cuda_advanced_controls_mode_name(
    CUDAAdvancedControlsMode mode) noexcept {
  switch (mode) {
    case CUDAAdvancedControlsMode::baseline:
      return "baseline";
    case CUDAAdvancedControlsMode::apply_explicit_acf:
      return "apply_explicit_acf";
    case CUDAAdvancedControlsMode::request_tuning:
      return "request_tuning";
  }
  return "unknown";
}

std::optional<CUDAAdvancedControls> resolve_cuda_advanced_controls(
    const CUDACompileIQProtocolRequest &request,
    const CUDAAdvancedControlsConfiguration &configuration) {
  auto &telemetry = protocol_telemetry();
  if (configuration.nested_tuning_request_rejected) {
    telemetry.nested_requests_rejected.fetch_add(1,
                                                  std::memory_order_relaxed);
  }
  if (configuration.mode == CUDAAdvancedControlsMode::baseline) {
    return std::nullopt;
  }
  telemetry.requests.fetch_add(1, std::memory_order_relaxed);
  try {
    if (configuration.mode ==
        CUDAAdvancedControlsMode::apply_explicit_acf) {
      const auto path = regular_file(configuration.explicit_acf_path,
                                     "TI_CUDA_PTXAS_ACF_PATH");
      auto bytes = read_binary(path);
      TI_ERROR_IF(bytes.empty(), "TI_CUDA_PTXAS_ACF_PATH '{}' is empty.",
                  path.string());
      const auto hash = sha256_bytes(bytes);
      const auto source_identity = "direct_acf:" + hash.substr(0, 16);
      const auto cached_path = request.cache_root / (hash + ".direct.acf");
      if (auto cached = read_cached_acf(cached_path, source_identity)) {
        telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
        telemetry.acf_responses.fetch_add(1, std::memory_order_relaxed);
        return cached;
      }
      const auto lock_path = request.cache_root / (hash + ".direct.acf.lock");
      TI_ERROR_IF(!lock_with_file_handle(lock_path.string(), kWorkerLockDelayMs,
                                         kWorkerLockTryCount),
                  "Timed out waiting for direct ACF cache lock '{}'.",
                  lock_path.string());
      auto unlocker = make_cleanup([&]() {
        if (!unlock_file_handle(lock_path.string())) {
          TI_WARN("Failed to release direct ACF cache lock '{}'.",
                  lock_path.string());
        }
      });
      if (auto cached = read_cached_acf(cached_path, source_identity)) {
        telemetry.cache_hits.fetch_add(1, std::memory_order_relaxed);
        telemetry.acf_responses.fetch_add(1, std::memory_order_relaxed);
        return cached;
      }
      install_verified_blob(cached_path, bytes);
      telemetry.acf_responses.fetch_add(1, std::memory_order_relaxed);
      return CUDAAdvancedControls{cached_path, hash, source_identity};
    }
    return resolve_worker_controls(request, configuration.worker_path,
                                   configuration.python_path);
  } catch (...) {
    if (configuration.mode == CUDAAdvancedControlsMode::request_tuning) {
      telemetry.worker_failures.fetch_add(1, std::memory_order_relaxed);
    }
    telemetry.fail_open_responses.fetch_add(1, std::memory_order_relaxed);
    TI_WARN(
        "CUDA Advanced Controls resolution failed open; using baseline ptxas");
    return std::nullopt;
  }
}

CUDACompileIQProtocolTelemetrySnapshot
get_cuda_compileiq_protocol_telemetry_snapshot() {
  auto &telemetry = protocol_telemetry();
  return {
      telemetry.requests.load(std::memory_order_relaxed),
      telemetry.cache_hits.load(std::memory_order_relaxed),
      telemetry.worker_calls.load(std::memory_order_relaxed),
      telemetry.worker_failures.load(std::memory_order_relaxed),
      telemetry.worker_wall_ns.load(std::memory_order_relaxed),
      telemetry.acf_responses.load(std::memory_order_relaxed),
      telemetry.pass_responses.load(std::memory_order_relaxed),
      telemetry.fail_open_responses.load(std::memory_order_relaxed),
      telemetry.nested_requests_rejected.load(std::memory_order_relaxed),
  };
}

}  // namespace taichi::lang::cuda
