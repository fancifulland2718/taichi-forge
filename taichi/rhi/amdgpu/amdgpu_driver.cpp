#include "taichi/rhi/amdgpu/amdgpu_driver.h"

#include <filesystem>
#include <hip/hip_runtime_api.h>

#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/util/environ_config.h"

#if HIP_VERSION_MAJOR < 7
#error "The AMDGPU backend requires HIP 7 or newer SDK headers."
#endif

namespace taichi::lang {

static_assert(HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X ==
              hipDeviceAttributeMaxBlockDimX);
static_assert(HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT ==
              hipDeviceAttributeMultiprocessorCount);

std::string get_amdgpu_error_message(uint32 err) {
  auto &driver = AMDGPUDriver::get_instance_without_context();
  return fmt::format("AMDGPU Error {}: {}", driver.get_error_name(err),
                     driver.get_error_string(err));
}

AMDGPUDriverBase::AMDGPUDriverBase() {
  disabled_by_env_ = (get_environ_config("TI_ENABLE_AMDGPU", 1) == 0);
}

bool AMDGPUDriverBase::load_lib() {
  if (disabled_by_env_)
    return false;
  std::vector<std::string> paths;
  if (const auto *explicit_path = std::getenv("TI_HIP_RUNTIME_PATH")) {
    // An explicit runtime must not silently fall through to another SDK.
    paths.emplace_back(explicit_path);
  } else {
#if defined(TI_PLATFORM_WINDOWS)
    const std::vector<std::string> names = {"amdhip64_7.dll", "amdhip64.dll"};
    const auto subdir = "bin";
#else
    const std::vector<std::string> names = {"libamdhip64.so.7",
                                            "libamdhip64.so"};
    const auto subdir = "lib";
#endif
    for (const char *key : {"ROCM_PATH", "HIP_PATH"}) {
      if (const auto *root = std::getenv(key)) {
        for (const auto &name : names)
          paths.push_back(
              (std::filesystem::path(root) / subdir / name).string());
      }
    }
    paths.insert(paths.end(), names.begin(), names.end());
  }
  for (const auto &path : paths) {
    loader_ = std::make_unique<DynamicLoader>(path);
    if (loader_->loaded()) {
      TI_TRACE("Loaded HIP runtime: {}", path);
      return true;
    }
  }
  TI_TRACE(
      "No HIP runtime found; configure ROCM_PATH/HIP_PATH or "
      "TI_HIP_RUNTIME_PATH.");
  return false;
}

bool AMDGPUDriver::detected() {
  return available_;
}

AMDGPUDriver::AMDGPUDriver() {
  if (!load_lib())
    return;
  auto required = [&](const char *symbol) {
    auto *address = loader_->load_function_optional(symbol);
    if (!address)
      TI_WARN("HIP runtime missing required symbol: {}", symbol);
    return address;
  };
  get_error_name =
      reinterpret_cast<decltype(get_error_name)>(required("hipGetErrorName"));
  get_error_string = reinterpret_cast<decltype(get_error_string)>(
      required("hipGetErrorString"));
  driver_get_version = reinterpret_cast<decltype(driver_get_version)>(
      required("hipDriverGetVersion"));
  runtime_get_version = reinterpret_cast<decltype(runtime_get_version)>(
      required("hipRuntimeGetVersion"));
  if (!get_error_name || !get_error_string || !driver_get_version ||
      !runtime_get_version)
    return;

  int version = 0;
  if (runtime_get_version(&version) != HIP_SUCCESS || version < 70000000) {
    TI_WARN("AMDGPU requires HIP 7 or newer; reported runtime version {}",
            version);
    return;
  }
  // Use the SDK's versioned ABI, never infer struct layout from its contents.
  device_properties_ = required("hipGetDevicePropertiesR0600");
  pointer_attributes_ = required("hipPointerGetAttributes");
  if (!device_properties_ || !pointer_attributes_)
    return;

#define PER_AMDGPU_FUNCTION(name, symbol_name, ...) \
  {                                                 \
    auto *address = required(#symbol_name);         \
    if (!address)                                   \
      return;                                       \
    name.set(address);                              \
    name.set_lock(&lock_);                          \
    name.set_names(#name, #symbol_name);            \
  }
#include "taichi/rhi/amdgpu/amdgpu_driver_functions.inc.h"
#undef PER_AMDGPU_FUNCTION

  // A loadable runtime without a supported GPU is not an available backend.
  // This happens once at discovery, not once per launch.
  int count = 0;
  if (init.call(0) != HIP_SUCCESS ||
      device_get_count.call(&count) != HIP_SUCCESS || count == 0) {
    TI_TRACE("HIP runtime loaded, but no supported AMD GPU is available.");
    return;
  }
  available_ = true;
}

AMDGPUDriver::DeviceInfo AMDGPUDriver::device_info(int device) {
  hipDeviceProp_t properties{};
  const auto get_properties =
      reinterpret_cast<decltype(&hipGetDeviceProperties)>(device_properties_);
  std::lock_guard<std::mutex> guard(lock_);
  auto result = get_properties(&properties, device);
  TI_ERROR_IF(result != hipSuccess, "HIP device properties failed: {}",
              int(result));
  return {properties.gcnArchName, properties.major, properties.minor,
          properties.warpSize};
}

bool AMDGPUDriver::pointer_is_device(void *ptr) {
  hipPointerAttribute_t attributes{};
  const auto get_attributes =
      reinterpret_cast<decltype(&hipPointerGetAttributes)>(pointer_attributes_);
  std::lock_guard<std::mutex> guard(lock_);
  return get_attributes(&attributes, ptr) == hipSuccess &&
         (attributes.type == hipMemoryTypeDevice ||
          attributes.type == hipMemoryTypeManaged);
}

AMDGPUDriver &AMDGPUDriver::get_instance_without_context() {
  static auto *instance = new AMDGPUDriver();
  return *instance;
}

AMDGPUDriver &AMDGPUDriver::get_instance() {
  AMDGPUContext::get_instance();
  return get_instance_without_context();
}

}  // namespace taichi::lang
