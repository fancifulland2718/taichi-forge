#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_fft.h"
#include "taichi/common/operation_json.h"

#include <limits>

namespace taichi::lang {
#ifdef TI_WITH_CUDA
// The existing cuFFT implementation owns both JIT and AOT vendor plans.
std::shared_ptr<NativeOperation> prepare_cuda_fft_operation(
    Arch arch,
    Device &device,
    const FftC2COperation &operation);
NativeOperationAvailability probe_cuda_fft_operation();
std::shared_ptr<NativeResource> prepare_cuda_fft_plan(
    Arch arch,
    Device &device,
    const FftPlanDescription &description);
std::shared_ptr<NativeOperation> prepare_cuda_fft_execution(
    Arch arch,
    Device &device,
    const FftExecuteOperation &operation);
#endif

namespace {

int positive_int(const liong::json::JsonValue &value) {
  const auto number = operation_json::integer(value);
  if (!number || number > std::numeric_limits<int>::max())
    throw std::invalid_argument("cuFFT PlanMany values must be positive int32");
  return static_cast<int>(number);
}

std::vector<int> fft_dimensions(const liong::json::JsonValue &value) {
  if (!value.is_arr() || value.arr.inner.empty() || value.arr.inner.size() > 3)
    throw std::invalid_argument("cuFFT PlanMany requires rank 1-3");
  std::vector<int> result;
  for (const auto &axis : value.arr.inner)
    result.push_back(positive_int(axis));
  return result;
}

FftLayoutDescription fft_layout(const liong::json::JsonValue &parameters,
                                const char *key,
                                std::vector<int> logical) {
  using namespace operation_json;
  auto found = parameters.obj.inner.find(key);
  FftLayoutDescription layout;
  if (found == parameters.obj.inner.end()) {
    layout.embed = std::move(logical);
    uint64_t distance = 1;
    for (int extent : layout.embed) {
      if (distance > std::numeric_limits<int>::max() / extent)
        throw std::invalid_argument(
            "cuFFT compact batch distance exceeds int32");
      distance *= extent;
    }
    layout.batch_distance = static_cast<int>(distance);
  } else {
    const auto &value = found->second;
    fields(value, {"embed", "stride", "batch_distance"});
    layout.embed = fft_dimensions(value["embed"]);
    layout.stride = positive_int(value["stride"]);
    layout.batch_distance = positive_int(value["batch_distance"]);
    if (layout.embed.size() != logical.size())
      throw std::invalid_argument(
          "cuFFT layout rank does not match dimensions");
    for (std::size_t i = 0; i < logical.size(); ++i)
      if (layout.embed[i] < logical[i])
        throw std::invalid_argument(
            "cuFFT embed must cover logical dimensions");
  }
  return layout;
}

void register_fft_plan(NativeOperationRegistry &registry) {
  registry.define<FftPlanDescription>(
      [](Device &, const FftPlanDescription &) {}, NativeObjectKind::resource);
  registry.register_schema<FftPlanDescription>(
      "cufft", "plan_f32", {},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<FftPlanDescription> {
        using namespace operation_json;
        fields(parameters, {"dimensions", "batch_count", "transform",
                            "input_layout", "output_layout"});
        FftPlanDescription result;
        result.dimensions = fft_dimensions(parameters["dimensions"]);
        auto batch = parameters.obj.inner.find("batch_count");
        if (batch != parameters.obj.inner.end())
          result.batch_count = positive_int(batch->second);
        const auto transform = name(parameters["transform"]);
        if (transform != "c2c" && transform != "r2c" && transform != "c2r")
          throw std::invalid_argument(
              "cuFFT transform must be c2c, r2c or c2r");
        result.transform = transform == "c2c"   ? FftTransformKind::c2c
                           : transform == "r2c" ? FftTransformKind::r2c
                                                : FftTransformKind::c2r;
        auto input = result.dimensions, output = result.dimensions;
        if (result.transform == FftTransformKind::r2c)
          output.back() = output.back() / 2 + 1;
        if (result.transform == FftTransformKind::c2r)
          input.back() = input.back() / 2 + 1;
        result.input_layout = fft_layout(parameters, "input_layout", input);
        result.output_layout = fft_layout(parameters, "output_layout", output);
        return [result](const NativeOperationBindings &) { return result; };
      });
  registry.define<FftExecuteOperation>(
      [](Device &device, const FftExecuteOperation &operation) {
        if (!operation.plan || &operation.plan->device() != &device ||
            operation.input.pointer.device != &device ||
            operation.output.pointer.device != &device)
          throw std::invalid_argument(
              "cuFFT plan or storage belongs to another device");
        const auto transform = operation.plan->description().transform;
        if ((transform == FftTransformKind::r2c && operation.inverse) ||
            (transform == FftTransformKind::c2r && !operation.inverse))
          throw std::invalid_argument(
              "cuFFT direction does not match real transform");
      });
  registry.register_schema<FftExecuteOperation>(
      "cufft", "execute_f32", {"plan", "input", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<FftExecuteOperation> {
        using namespace operation_json;
        fields(parameters, {"direction"});
        const auto direction = name(parameters["direction"]);
        if (direction != "forward" && direction != "inverse")
          throw std::invalid_argument(
              "cuFFT direction must be forward or inverse");
        return [inverse = direction ==
                          "inverse"](const NativeOperationBindings &bindings) {
          return FftExecuteOperation{
              bindings.at("plan").resource<FftPlanResource>(),
              bindings.at("input").array(), bindings.at("output").array(),
              inverse};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<FftPlanDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cuda_fft_plan,
      probe_cuda_fft_operation);
  registry.register_factory<FftExecuteOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cuda_fft_execution,
      probe_cuda_fft_operation);
#endif
}
}  // namespace

std::size_t fft_c2c_element_count(const FftC2COperation &operation) {
  if (operation.dimensions.empty() || !operation.batch_count)
    throw std::invalid_argument(
        "FFT requires positive dimensions and batch count");
  const auto limit =
      std::numeric_limits<std::size_t>::max() / (2 * sizeof(float));
  std::size_t count = operation.batch_count;
  for (auto dimension : operation.dimensions) {
    if (!dimension || count > limit / dimension)
      throw std::invalid_argument(
          "FFT extent is zero or overflows storage size");
    count *= dimension;
  }
  return count;
}

void register_fft_operation(NativeOperationRegistry &registry) {
  register_fft_plan(registry);
  registry.define<FftC2COperation>(
      [](Device &device, const FftC2COperation &op) {
        fft_c2c_element_count(op);
        if (op.input.device != &device || op.output.device != &device)
          throw std::invalid_argument("FFT storage belongs to another device");
      });
  registry.register_schema<FftC2COperation>(
      "cufft", "fft_c2c_f32", {"input", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<FftC2COperation> {
        using namespace operation_json;
        fields(parameters, {"dimensions", "batch_count", "direction"});
        FftC2COperation descriptor{{}, {}, {}, 1, false};
        const auto &dimensions = parameters["dimensions"];
        if (!dimensions.is_arr() || dimensions.arr.inner.empty())
          throw std::invalid_argument(
              "FFT dimensions must be a nonempty array");
        for (const auto &axis : dimensions.arr.inner) {
          const auto value = integer(axis);
          if (value > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(
                "FFT dimension overflows address space");
          descriptor.dimensions.push_back(static_cast<std::size_t>(value));
        }
        const auto batch =
            optional_integer(parameters, "batch_count").value_or(1);
        if (batch > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument(
              "FFT batch count overflows address space");
        descriptor.batch_count = static_cast<std::size_t>(batch);
        const auto direction = name(parameters["direction"]);
        if (direction != "forward" && direction != "inverse")
          throw std::invalid_argument(
              "FFT direction must be forward or inverse");
        descriptor.inverse = direction == "inverse";
        const auto bytes =
            fft_c2c_element_count(descriptor) * 2 * sizeof(float);
        return [descriptor, bytes](const NativeOperationBindings &bindings) {
          auto buffer = [&](const char *port) {
            const auto &value = bindings.at(port).array();
            if (value.scalar_type != PrimitiveType::f32 ||
                value.element_shape != std::vector<std::uint64_t>{2} ||
                value.base_offset % 8 || value.pointer.offset % 8)
              throw std::invalid_argument(
                  "FFT requires aligned interleaved complex f32 vector(2) "
                  "ndarrays");
            if (value.bytes != bytes)
              throw std::invalid_argument(
                  "FFT dimensions/batch do not match ndarray storage");
            return value.pointer;
          };
          auto result = descriptor;
          result.input = buffer("input");
          result.output = buffer("output");
          return result;
        };
      },
      "fft_c2c_f32");
#ifdef TI_WITH_CUDA
  registry.register_factory<FftC2COperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cuda_fft_operation,
      probe_cuda_fft_operation);
#endif
}

std::shared_ptr<NativeOperation> prepare_fft_operation(
    Arch arch,
    Device &device,
    const FftC2COperation &operation) {
  return native_operation_registry().prepare(arch, device, operation);
}
}  // namespace taichi::lang
