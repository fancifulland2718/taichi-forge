#include "taichi/runtime/cuda/cuda_compileiq_protocol.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"

#include "taichi/common/cleanup.h"
#include "taichi/common/json.h"
#include "taichi/system/timer.h"
#include "taichi/util/environ_config.h"
#include "taichi/util/lock.h"

namespace taichi::lang::cuda {

namespace {

constexpr int kCompileIQProtocolSchema = 1;
constexpr int kWorkerLockDelayMs = 50;
constexpr int kWorkerLockTryCount = 1200;
constexpr std::size_t kMaxWorkerLogBytes = 64 * 1024;

struct ProtocolTelemetry {
  std::atomic<std::uint64_t> requests{0};
  std::atomic<std::uint64_t> cache_hits{0};
  std::atomic<std::uint64_t> worker_calls{0};
  std::atomic<std::uint64_t> worker_failures{0};
  std::atomic<std::uint64_t> worker_wall_ns{0};
  std::atomic<std::uint64_t> acf_responses{0};
  std::atomic<std::uint64_t> pass_responses{0};
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
      program, arguments, std::nullopt, redirects, timeout,
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

  auto controls = invoke_worker(request, worker, python, key);
  if (controls) {
    telemetry.acf_responses.fetch_add(1, std::memory_order_relaxed);
    return controls;
  }
  const std::string marker = "pass\n";
  auto pass_tmp = pass_path;
  pass_tmp += ".install.tmp";
  TI_ERROR_IF(!write_binary(pass_tmp, marker.data(), marker.size()),
              "Cannot write CompileIQ pass cache entry '{}'.",
              pass_tmp.string());
  auto cleanup = make_cleanup([&]() {
    std::error_code ignored;
    std::filesystem::remove(pass_tmp, ignored);
  });
  atomic_install(pass_tmp, pass_path);
  telemetry.pass_responses.fetch_add(1, std::memory_order_relaxed);
  return std::nullopt;
}

}  // namespace

std::optional<CUDAAdvancedControls> resolve_cuda_advanced_controls(
    const CUDACompileIQProtocolRequest &request) {
  const auto direct_acf = env_string("TI_CUDA_PTXAS_ACF_PATH");
  const auto worker = env_string("TI_CUDA_COMPILEIQ_WORKER");
  TI_ERROR_IF(!direct_acf.empty() && !worker.empty(),
              "TI_CUDA_PTXAS_ACF_PATH and TI_CUDA_COMPILEIQ_WORKER are "
              "mutually exclusive.");
  if (direct_acf.empty() && worker.empty()) {
    return std::nullopt;
  }
  auto &telemetry = protocol_telemetry();
  telemetry.requests.fetch_add(1, std::memory_order_relaxed);
  try {
    if (!direct_acf.empty()) {
      const auto path = regular_file(direct_acf, "TI_CUDA_PTXAS_ACF_PATH");
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
    return resolve_worker_controls(request, worker,
                                   env_string("TI_CUDA_COMPILEIQ_PYTHON"));
  } catch (...) {
    if (!worker.empty()) {
      telemetry.worker_failures.fetch_add(1, std::memory_order_relaxed);
    }
    throw;
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
  };
}

}  // namespace taichi::lang::cuda
