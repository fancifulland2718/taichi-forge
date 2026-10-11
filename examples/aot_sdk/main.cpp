#include <taichi/cpp/taichi.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: aot_runner cpu|cuda|vulkan MODULE_DIR SDK_RUNTIME_DIR\n";
    return 2;
  }
  try {
#if defined(_WIN32)
    _putenv_s("TI_LIB_DIR", argv[3]);
#else
    setenv("TI_LIB_DIR", argv[3], 1);
#endif
    const std::string backend = argv[1];
#if defined(__aarch64__) || defined(_M_ARM64)
    const TiArch cpu = TI_ARCH_ARM64;
#else
    const TiArch cpu = TI_ARCH_X64;
#endif
    const TiArch arch = backend == "cpu" ? cpu : backend == "cuda" ? TI_ARCH_CUDA : TI_ARCH_VULKAN;
    if (backend != "cpu" && backend != "cuda" && backend != "vulkan")
      throw std::runtime_error("Unknown backend");
    if (!ti::is_arch_available(arch))
      throw std::runtime_error("Backend unavailable in this SDK/device");
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    auto report = [&](const char *name) {
      const auto now = Clock::now();
      std::cout << name << "=" << std::chrono::duration<double>(now - start).count() << "s\n";
      start = now;
    };
    ti::Runtime runtime(arch);
    ti::check_last_error();
    report("runtime_init");
    auto module = runtime.load_aot_module(argv[2]);
    ti::check_last_error();
    report("module_metadata");
    auto forward = module.get_kernel("forward");
    auto backward = module.get_kernel("backward");
    auto graph = module.get_compute_graph("differentiate");
    ti::check_last_error();
    report("kernel_and_graph_load");
    auto x = runtime.allocate_ndarray<float>({4}, {}, true);
    auto y = runtime.allocate_ndarray<float>({4}, {}, true);
    auto dx = runtime.allocate_ndarray<float>({4}, {}, true);
    auto dy = runtime.allocate_ndarray<float>({4}, {}, true);
    x.write(std::vector<float>{1, 2, 3, 4});
    forward.push_arg(x);
    forward.push_arg(y);
    backward.push_arg(x);
    backward.push_arg(y);
    graph["x"] = x;
    graph["y"] = y;
    for (bool use_graph : {false, true}) {
      const float seed = use_graph ? -0.5f : 1.0f;
      y.write(std::vector<float>(4, -100));
      dx.write(std::vector<float>(4, 0));
      dy.write(std::vector<float>(4, seed));
      if (use_graph) {
        graph.launch_with_gradients({{"x", dx.ndarray()}, {"y", dy.ndarray()}});
      } else {
        const std::vector<TiNdArrayGradientBinding> bindings{{0, dx.ndarray()}, {1, dy.ndarray()}};
        forward.launch_with_gradients(bindings);
        backward.launch_with_gradients(bindings);
      }
      runtime.wait();
      ti::check_last_error();
      std::vector<float> values(4), gradients(4);
      y.read(values);
      dx.read(gradients);
      ti::check_last_error();
      for (int i = 0; i < 4; ++i) {
        const float value = i + 1;
        if (values[i] != value * value + 2 * value ||
            gradients[i] != seed * (2 * value + 2))
          throw std::runtime_error("Forward/reverse result mismatch");
      }
    }
    std::cout << "PASS: direct and ordinary Graph forward/reverse\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
