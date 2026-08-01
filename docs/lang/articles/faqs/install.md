---
sidebar_position: 2
---

# Installation Troubleshooting

## Linux issues

- If Taichi crashes and reports
  `` /usr/lib/libstdc++.so.6: version `CXXABI_1.3.11' not found ``:

  You might be using Ubuntu 16.04. Please try the solution in [this
  thread](https://github.com/tensorflow/serving/issues/819#issuecomment-377776784):

  ```bash
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  sudo apt-get update
  sudo apt-get install libstdc++6
  ```


## Windows issues

- If Taichi crashes and reports `ImportError` on Windows. Please
  consider installing [Microsoft Visual C++
  Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

## Python issues

- If `pip` could not find a satisfying package,
  i.e.,

  ```
  ERROR: Could not find a version that satisfies the requirement taichi-forge (from versions: none)
  ERROR: No matching distribution found for taichi-forge
  ```

  - Make sure you're using a supported 64-bit Python version, currently 3.10--3.14:

    ```bash
    python3 -c "import sys;print(sys.version[:sys.version.find('.', 2)])"
    # 3.10 through 3.14
    ```

  - Make sure your Python executable is 64-bit:

    ```bash
    python3 -c "print(__import__('platform').architecture()[0])"
    # 64bit
    ```

## CUDA issues

- If Taichi exits with message "Out of CUDA pre-allocated memory", e.g.,

  ```python
  import taichi_forge as ti

  ti.init(arch=ti.cuda)

  x = ti.field(dtype=ti.i16)

  ti.root.pointer(ti.i, 1024).dense(ti.i, 1024 * 1024).place(x)
  # A sparse array. Each dense block is 2MB in size.

  # Populate 1024 * 2MB = 2GB memory
  def populate():
    for k in range(1024):
      x[k * 1024 * 1024] = 1

  populate()
  ```

  ... may give you ...

  ```
  [Taichi] Starting on arch=cuda
  Taichi JIT:0: allocate_from_buffer: block: [0,0,0], thread: [0,0,0] Assertion `Out of CUDA pre-allocated memory.
  Consider using ti.init(device_memory_fraction=0.9) or ti.init(device_memory_GB=4) to allocate more GPU memory` failed.
  ```

  Current Forge releases derive the default CUDA sparse pool from the
  materialized SNode tree. This error therefore normally means that an
  explicitly selected fixed budget is too small, or that the active sparse
  topology exceeds its declared bound. Increase
  `cuda_sparse_pool_size_GB`, use an appropriate positive
  `device_memory_fraction`, or reduce the declared sparse capacity. Do not use
  `device_memory_GB` as a silent cap for the default auto-sized path. See
  [Forge options](../../../forge/forge_options.en.md#26-cuda-sparse-memory-pool)
  for the current precedence and memory contract.

- If you find other CUDA problems:

  - **Possible solution**: add `export TI_ENABLE_CUDA=0` to your
    `~/.bashrc`. This disables the CUDA backend completely and
    Taichi will fall back on other GPU backends such as OpenGL.

## OpenGL issues

- If Taichi crashes with a stack backtrace containing a line of
  `glfwCreateWindow` (see
  [\#958](https://github.com/taichi-dev/taichi/issues/958)):

  ```plaintext {9-11}
  [Taichi] mode=release
  [E 05/12/20 18.25:00.129] Received signal 11 (Segmentation Fault)
  ***********************************
  * Taichi Compiler Stack Traceback *
  ***********************************

  ... (many lines, omitted)

  /path/to/site-packages/taichi_forge/_lib/core/taichi_python.so: _glfwPlatformCreateWindow
  /path/to/site-packages/taichi_forge/_lib/core/taichi_python.so: glfwCreateWindow
  /path/to/site-packages/taichi_forge/_lib/core/taichi_python.so: taichi::lang::opengl::initialize_opengl(bool)

  ... (many lines, omitted)
  ```

  it is likely because you are running Taichi on a (virtual) machine
  with an old OpenGL API. Taichi requires OpenGL 4.3+ to work.

  - **Possible solution**: select a supported non-OpenGL backend explicitly,
    for example `ti.init(arch=ti.cpu)` or a qualified Vulkan backend, and do
    not create an OpenGL window in a headless process. Forge does not document
    the historical `TI_ENABLE_OPENGL` environment switch as a current runtime
    contract.

## Installation interrupted
If installation is interrupted by an `HTTPSConnection` error, retry with a
larger timeout. If you use a PyPI-compatible mirror, verify that it carries
both the `taichi-forge` shim and the exact matching `taichi-forge-runtime`
version.

```
python -m pip install -U taichi-forge --retries 10 --timeout 60
```

## Other issues

- If none of the above addresses your problem, report it in the
  [Taichi Forge issue tracker](https://github.com/fancifulland2718/taichi-forge/issues)
  with `ti diagnose`, the loaded package path, backend, driver, and wheel
  versions.
