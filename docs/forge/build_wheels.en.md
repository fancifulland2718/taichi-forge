# Building Forge Wheels

> Current source build contract: `0.6.3`. This is not a claim that every build
> profile or hardware combination is release-qualified. The split runtime/shim
> model first shipped in `0.4.23`; see [release notes](release_notes.en.md).

This document mirrors the public wheel build path used by
`.github/workflows/publish_runtime_pypi.yml` and
`.github/workflows/publish_pypi.yml`. It is intended for external developers who
want to reproduce the PyPI-style Windows or Ubuntu builds locally.

## Wheel Matrix

The publish workflow builds two wheel families:

- `taichi-forge-runtime`: platform-native runtime wheels tagged as
  `py3-none-win_amd64` and `py3-none-manylinux_2_35_x86_64`.
- `taichi-forge`: small CPython pybind shim wheels for Python 3.10, 3.11,
  3.12, 3.13, and 3.14 on Windows x86_64 and Linux x86_64.

`pip install taichi-forge` installs the shim wheel for the active Python
version and pulls the matching `taichi-forge-runtime` wheel through the package
dependency. The public import path remains `import taichi_forge`.

The pybind shim is still per-CPython-minor. `pyproject.toml` currently sets
`wheel.py-api = ""`; the project does not publish `abi3` shim wheels.

### Private native ABI boundary

The native link surface between the two wheel families is package-private. It
is not a public C++ SDK ABI. The runtime build derives the exact shim-reachable
symbol closure and installs `taichi_runtime.exports.json` beside the native
library. Windows applies that closure through a `.def` file, Linux through an
ELF version script, and split-runtime macOS source builds through an
exported-symbols list. Required Taichi RTTI/ODR identities are included only
when the shim actually imports them; definitions already owned by the shim are
not exported speculatively. ELF and Mach-O localize every other definition.
On Windows, the generated closure is combined with Taichi declarations that
are explicitly marked `dllexport`: MSVC uses those declarations to emit class
special members and vtables required by an independently compiled shim. The
post-link audit permits this bounded Taichi-owned set but rejects bundled
third-party owners and enforces the export safety cap. The public wheel workflow
currently qualifies Windows and Linux; this statement does not announce a
macOS wheel.

The Linux shim has an explicit `DT_NEEDED` edge to `libtaichi_runtime.so` and a
package-relative `RUNPATH` into `taichi-forge-runtime`. The loader keeps both
the runtime and an optional packaged CUDART in `RTLD_LOCAL` scope. This prevents
LLVM, SPIR-V, UI, allocator, and other implementation symbols from entering the
process-wide lookup domain while preserving the C++ identity shared by the shim
and its direct dependency.

Runtime and shim source commits need not be identical. A shim is built by
linking against an already-built or published runtime wheel of the same package version;
that link plus the private-ABI manifest is the compatibility gate. Validators
therefore compare package version, ABI revision, canonical export closure, and
the final binary audit rather than requiring equal Git identities.

On POSIX, the loader also checks a small manifest-selected set of Taichi-owned
private ABI symbols before loading the runtime. If an embedder has already made
an incompatible Taichi ABI process-global, import fails closed instead of
allowing the global definition to preempt the package-private dependency. This
check is import-time only and does not enter kernel or Graph launch paths.

## Common Rules

Use the scikit-build-core path:

```bash
python -I -m build --wheel --no-isolation
```

Important details:

- Install `packaging/constraints/release-build.txt` before building. Release
  runtime and shim jobs use the same pinned pybind11 generation and the
  file's Python-specific NumPy constraints; do not substitute old pybind11 2.x
  instructions. Project metadata separately declares supported version ranges.
- `cmake<4` is required because some vendored CMake projects still declare old
  policy versions rejected by CMake 4.
- Use `CMAKE_ARGS`, not `TAICHI_CMAKE_ARGS`. The publish workflow uses
  scikit-build-core, and `TAICHI_CMAKE_ARGS` is a legacy setup.py variable that
  will be ignored here.
- Set `LLVM_DIR` to a prebuilt LLVM 20 installation containing
  `lib/cmake/llvm/LLVMConfig.cmake`.
- The release wheel configuration enables Vulkan, OpenGL, CUDA, and LLVM,
  splits the native runtime out of the CPython shim, omits the C API package
  tree, and disables test binaries. C API artifacts are native distribution
  outputs and should be built or published separately when needed.
- The platform runtime wheel is built before the CPython shim wheels. Shim
  builds should consume the already-built or already-published runtime wheel
  instead of rebuilding the C++ runtime for every Python version.
- Shim builds extract the native library/import library and pass
  `TI_PREBUILT_PYTHON_RUNTIME_DIR`. Runtime bitcode and libdevice install rules
  remain owned by `taichi-forge-runtime`; a shim wheel must contain one pybind
  extension and no duplicate runtime, CUDART, or bitcode assets.
- After changing `version.txt`, run `python scripts/sync_runtime_dependency.py`
  before building release wheels. The publish workflow does this automatically.
- PyPI/TestPyPI publishing must be authorized for both project names:
  `taichi-forge` and `taichi-forge-runtime`.

Base release build `CMAKE_ARGS`:

```bash
-DTI_WITH_VULKAN:BOOL=ON
-DTI_WITH_OPENGL:BOOL=ON
-DTI_WITH_CUDA:BOOL=ON
-DTI_WITH_LLVM:BOOL=ON
-DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON
-DTI_WITH_C_API:BOOL=OFF
-DTI_BUILD_TESTS:BOOL=OFF
```

The `taichi-forge-runtime` workflow appends these runtime-only flags:

```bash
-DTI_WITH_CUDA_TOOLKIT:BOOL=OFF
-DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF
-DTI_WITH_CUPTI:BOOL=OFF
```

The workflow also sets:

```bash
TI_SKIP_VERSION_CHECK=ON
TI_CI=1
```

## Bundled libdevice compatibility version

The native runtime packages exactly one
`slim_libdevice.<major>.bc` compatibility asset. At configure time, the build
discovers that filename, installs the same file, and derives the internal
`TI_CUDA_LIBDEVICE_VERSION` string from its major version. This keeps a
repackaged runtime wheel internally consistent without tying it to the CUDA
Toolkit used to build it or to the GPU driver's version on an end-user system.

`taichi_forge._lib.core.cuda_version()` retains its historical string return
shape, but means the compatibility version of that bundled libdevice asset.
It is not a driver or Toolkit version probe. A release validation job should
assert that a built runtime wheel contains exactly one such asset and that its
major version matches this value after installation.

## CUDA dependency classes

Current source builds the standard CUDA runtime and native primitive providers
against Forge's dynamically loaded Driver API. Installing and using that core
does not require a user-side CUDA Toolkit, CUB, CUDART, CUPTI, or CUDA-versioned
Python package. Building the complete runtime wheel is different: its existing
bundled OptiX adapters compile PTX with CUDA Toolkit 12.5.1, and its cuDSS adapter
uses the build-only headers from `packaging/constraints/cudss-build.txt`.
Neither the Toolkit nor those vendor runtimes become wheel payloads. A release
still publishes exactly one
`taichi-forge-runtime` wheel for Windows and one for Linux; the distribution
name, dependency, extras, and wheel tag never carry a `cu11`, `cu12`, or
`cu13` suffix.

Application-installed cuDSS, OptiX, cuSPARSELt, cuTENSOR, AmgX, NCCL, and
similar vendor runtimes remain outside this release matrix. The runtime wheel
may carry thin Forge adapters that do not link those vendor runtimes. Their
installation, loader, version, lifecycle, and support-status boundaries are documented in
[Optional external hardware providers](external_hardware_providers.en.md).
The standard wheel also carries runtime-loaded execution adapters for
cuSPARSELt, cuTENSOR, and AmgX. They build without vendor headers or libraries,
load a user-installed runtime only when explicitly requested, and the release
gate rejects any implicit vendor-runtime dependency.

`scripts/validate_runtime_wheel.py --dependency-class driver-only` is the
standard release gate. It verifies project/version identity, one native runtime,
no `cuda_runtime_major.txt`, no bundled CUDART, and no native dependency on a
Toolkit runtime library. The Windows and Linux release workflows additionally
scan PE imports or ELF `DT_NEEDED` entries before upload. Auditwheel may bundle
ordinary non-system libraries, but it must not introduce CUDART into the
driver-only candidate.

Already-published 0.5.0 runtime wheels used the earlier bundled-CUDART layout.
The Python loader, repair helper, and validators continue to recognize those
artifacts for installation and repair compatibility. This does not relax the
driver-only requirement for a newly built standard upload candidate. The
validator's default `either` mode exists for compatibility tooling; release
workflows select `driver-only` explicitly.

The optional `.github/workflows/test_cuda_toolkit_reference.yml` workflow is a
non-publishing developer target. It may install a recent Toolkit (currently
13.2), compile CUB/CUDART comparison providers with
`TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE=ON`, and run differential or
performance tests. Those binaries are not standard runtime payloads and
explicit `cuda_cub*` methods are deprecated. CUPTI/NVPerf is likewise an
independent developer/profiler capability, not a primitive or release
dependency.

### Build profiles and optional addons

`portable-runtime` describes the standard runtime's user-side dependency
boundary; `driver-only` remains the binary validator's CUDA dependency-class
argument. It does not mean that every build input is driver-only.

`cuda-toolkit-addon` is a separately built provider binary, not another copy of
the native runtime or a CPython-specific provider. The CUB source-provider
builder produces a component manifest describing compiler/Toolkit/CCCL,
linkage, target code and declared driver requirements. Manifest schema 3 and
the source-provider C ABI describe compatibility; a matching repository HEAD
is not required. Actual component changes still affect physical identity and
reuse applicability. Static linkage is not proof of Toolkit independence.

The reset-monoid segmented-scan addon can contribute complete Graph recipes;
this is distinct from the deprecated `cuda_cub*` reference methods. Its explicit
execution/search boundary is documented in
[Optional external hardware providers](external_hardware_providers.en.md).
Build it with `python -m taichi_forge.hardware.source_providers.cub.build --help`
for the current compiler/target options. Do not copy its CUDART, compiler,
vendor libraries or profiler runtime into the portable wheel. A full
`cuda-toolkit-specialized-runtime` variant is not required or published merely
to use this addon.

After installing both distributions, `scripts/validate_installed_runtime.py`
requires equal shim/runtime versions. Install validation resolves the shim
wheel's declared Python dependencies, uses the local runtime wheel, runs
`pip check`, and imports from outside the checkout. Each CPython build runs
`scripts/validate_shim_wheel.py` to reject missing direct Python dependencies,
duplicated runtime payloads, or a mismatched runtime dependency. The prebuilt
Linux shim uses LLVM headers only and intentionally does not link LLVMSupport,
so it disables LLVM's link sentinel with the header-supported
`LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING` mode. Wheel validation rejects an
extension that still contains an LLVM Enable/Disable ABI sentinel; the full
native runtime retains normal LLVM ABI checks.

Removing CUDART avoids making the build Toolkit a user-side runtime dependency,
but does not by itself establish a lower NVIDIA driver floor. PTX/module loading
and the complete primitive matrix must pass on each claimed target driver.
Linux and older-driver evidence still pending is tracked in
[Linux revalidation](linux_revalidation.en.md).

## LLVM 20

The publish workflow expects prebuilt LLVM 20 archives:

- Windows: `LLVM20_WIN_URL`
- Linux: `LLVM20_LINUX_URL` or `LLVM20_LINUX_MANYLINUX_URL`

CI also requires `LLVM20_WIN_SHA256` and `LLVM20_LINUX_SHA256` for the selected
archives. These are artifact integrity checks, not runtime/shim HEAD locks.

These are produced by:

- `.github/workflows/build_llvm20_windows.yml`
- `.github/workflows/build_llvm20_linux.yml`

External builders can either run those workflows and download the release
assets, or build LLVM 20 locally with equivalent flags. On Windows, the helper
`scripts/build_llvm20_local.ps1` is the preferred local route.

After extraction, set:

```bash
LLVM_DIR=/path/to/dist/taichi-llvm-20/lib/cmake/llvm
```

On Windows PowerShell:

```powershell
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
```

## Ubuntu 22.04 Build

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ninja-build ccache libxrandr-dev \
  libxinerama-dev libxcursor-dev libxi-dev libx11-dev \
  libxrender-dev libgl-dev libtinfo-dev libxml2-dev \
  libvulkan-dev vulkan-tools xz-utils
```

Install the Vulkan SDK used by the workflow:

```bash
VULKAN_SDK_VERSION=1.4.304.1
sdk_root=$(mktemp -d "${TMPDIR:-/tmp}/forge-vulkan-sdk.XXXXXX")
sdk_archive="$sdk_root/sdk.tar.xz"
curl -fsSL \
  "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz" \
  -o "$sdk_archive"
echo "92d698f12a968b024e2b593037830262785b2b734553683457719d4da7c5b0d6  $sdk_archive" | sha256sum -c -
tar -xJf "$sdk_archive" -C "$sdk_root" --strip-components=1

export VULKAN_SDK="$sdk_root/x86_64"
export VK_SDK_PATH="$VULKAN_SDK"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VULKAN_SDK/bin:$PATH"
```

Install Python build packages:

```bash
python -m pip install --requirement packaging/constraints/release-build.txt
python -m pip install --require-hashes --no-deps --requirement packaging/constraints/cudss-build.txt

export BASE_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts/sync_runtime_dependency.py
```

Install CUDA Toolkit 12.5.1 for the existing bundled PTX build and make `nvcc`
discoverable, as in the runtime workflow. This is a build-machine requirement,
not an end-user Toolkit dependency. Build the platform runtime wheel once:

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS -DTI_WITH_CUDA_TOOLKIT:BOOL=OFF -DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF -DTI_WITH_CUPTI:BOOL=OFF" \
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

Repair and validate this runtime using the workflow's `repair_runtime_wheel.py`
and strict binary checks before using it as a shim input. Set `RUNTIME_WHEEL`
to the absolute path of that single validated wheel. Extract into a fresh
directory and build the CPython shim for the active interpreter:

```bash
python -m zipfile -e "$RUNTIME_WHEEL" runtime-unpacked
runtime_link="$PWD/runtime-unpacked/taichi_forge_runtime/_lib/runtime_native"
CMAKE_ARGS="$BASE_CMAKE_ARGS \"-DTI_PREBUILT_PYTHON_RUNTIME_DIR=$runtime_link\"" \
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

The workflow runs `auditwheel repair` on both wheel families using the release
constraints (currently 6.8.1): the shim repair deliberately excludes the separately
distributed `libtaichi_runtime.so`, while post-repair validation requires the
`DT_NEEDED` edge and relative `RUNPATH` and rejects any duplicated runtime
payload. It also requires the runtime wheel's primary ELF to remain at the
canonical `taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so` path;
only grafted dependencies may receive auditwheel hash names. This floor includes
preservation of valid additional RPATH entries and the repaired dependency
traversal needed by a wheel whose primary payload is a non-Python runtime ELF.

Release jobs pass `--strict-binary` to both wheel validators. This reopens the
actual upload candidate and compares its PE/ELF export or dynamic-dependency
table with the manifest after repair; byte-string marker checks alone are not a
release qualification.

```bash
mkdir -p wheelhouse
auditwheel repair dist/*.whl -w wheelhouse/ \
  --plat manylinux_2_35_x86_64 \
  --exclude libtaichi_runtime.so
mkdir -p wheelhouse-runtime
auditwheel repair dist-runtime/*.whl -w wheelhouse-runtime/ --plat manylinux_2_35_x86_64
```

Ubuntu 22.04 uses glibc 2.35. To produce a lower manylinux tag, build inside a
matching manylinux container instead of native Ubuntu 22.04.

## Windows Build

Required tools:

- Python 3.10 through 3.14, matching the wheel being built.
- MSVC x64 developer environment.
- Ninja.
- Vulkan SDK `1.4.304.1`.
- CUDA Toolkit `12.5.1` and a compatible PTX host compiler for the bundled
  OptiX artifacts. The workflow explicitly selects installed MSVC v142 for
  this PTX command; native runtime/shim compilation keeps its own MSVC toolset.
- Prebuilt LLVM 20 archive from `LLVM20_WIN_URL`, or a local LLVM 20 build.

Install Python build packages:

```powershell
python -m pip install --requirement packaging/constraints/release-build.txt
python -m pip install --require-hashes --no-deps --requirement packaging/constraints/cudss-build.txt
```

Set paths:

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"
$env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
$baseCmakeArgs = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts\sync_runtime_dependency.py
```

Set `$ptxHost` to the absolute path of the compatible `cl.exe` selected for
CUDA 12.5 PTX preprocessing (not an unsupported newer compiler). Build the
platform runtime wheel once:

```powershell
$env:CMAKE_ARGS = "$baseCmakeArgs -DTI_WITH_CUDA_TOOLKIT:BOOL=OFF -DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF -DTI_WITH_CUPTI:BOOL=OFF"
$env:CMAKE_ARGS += " `"-DCMAKE_CUDA_HOST_COMPILER:FILEPATH=$ptxHost`""
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

Repair and validate the runtime using the workflow's existing helpers. Set
`$runtimeWheel` to that validated wheel's absolute path, extract into a fresh
directory, and build the CPython shim against its DLL/import library:

```powershell
python -m zipfile -e $runtimeWheel runtime-unpacked
$runtimeLink = (Resolve-Path runtime-unpacked/taichi_forge_runtime/_lib/runtime_native).Path.Replace('\', '/')
$env:CMAKE_ARGS = "$baseCmakeArgs `"-DTI_PREBUILT_PYTHON_RUNTIME_DIR=$runtimeLink`""
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

For local Windows LLVM builds, prefer:

```powershell
powershell -File scripts\build_llvm20_local.ps1
```

Then set `LLVM_DIR` to the generated `dist\taichi-llvm-20\lib\cmake\llvm`.

## CPU-Only Smoke Build

For local smoke checks, it is acceptable to disable CUDA, Vulkan, OpenGL, and
the C API to reduce build time. That matches the lightweight smoke workflow, not
the PyPI release workflow.

```bash
export CMAKE_ARGS="-DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_VULKAN:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python -I -m build --wheel --no-isolation
```

Use the full release `CMAKE_ARGS` above when validating behavior intended to
match PyPI wheels.
