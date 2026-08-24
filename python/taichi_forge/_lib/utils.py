import os
import platform
import re
import sys
import warnings
import ctypes
import importlib.util
import glob
import json

from colorama import Fore, Style

_startup_profile_enabled = os.environ.get("TI_STARTUP_PROFILE", "").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)
_startup_profile_clock = None
_startup_profile_origin_ns = None
_startup_profile_events = []


def configure_startup_profile(enabled=True, *, clear=False):
    """Enable low-frequency import/init timing checkpoints."""

    global _startup_profile_enabled
    global _startup_profile_clock
    global _startup_profile_origin_ns
    if not isinstance(enabled, bool):
        raise TypeError("startup profile enabled must be bool")
    if not isinstance(clear, bool):
        raise TypeError("startup profile clear must be bool")
    _startup_profile_enabled = enabled
    if clear:
        _startup_profile_events.clear()
        _startup_profile_origin_ns = None
    if enabled and _startup_profile_clock is None:
        import time  # pylint: disable=import-outside-toplevel

        _startup_profile_clock = time.perf_counter_ns
    if enabled and _startup_profile_origin_ns is None:
        _startup_profile_origin_ns = _startup_profile_clock()
    return _startup_profile_enabled


def startup_profile_mark(name):
    if not _startup_profile_enabled:
        return
    if not isinstance(name, str) or not name:
        raise ValueError("startup profile event name must be non-empty")
    now = _startup_profile_clock()
    _startup_profile_events.append((name, now - _startup_profile_origin_ns))


def startup_profile_raw_snapshot(*, clear=False):
    global _startup_profile_origin_ns
    if not isinstance(clear, bool):
        raise TypeError("startup profile clear must be bool")
    events = tuple(_startup_profile_events)
    elapsed_ns = (
        _startup_profile_clock() - _startup_profile_origin_ns
        if _startup_profile_enabled and _startup_profile_origin_ns is not None
        else 0
    )
    result = {
        "enabled": _startup_profile_enabled,
        "elapsed_ns": elapsed_ns,
        "events": events,
    }
    if clear:
        _startup_profile_events.clear()
        if _startup_profile_enabled:
            _startup_profile_origin_ns = _startup_profile_clock()
    return result


if _startup_profile_enabled:
    configure_startup_profile(True)
    startup_profile_mark("python_import.total.begin")

if sys.version_info[0] < 3 or sys.version_info[1] <= 5:
    raise RuntimeError(
        "\nPlease restart with Python 3.6+\n" + "Current Python version:",
        sys.version_info,
    )


def in_docker():
    if os.environ.get("TI_IN_DOCKER", "") == "":
        return False
    return True


_dll_dir_handles = []
_native_library_handles = []
_native_runtime_loaded = False
_loaded_native_runtime_path = None
_CUDA_RUNTIME_MAJOR_MANIFEST = "cuda_runtime_major.txt"


def _native_load_trace(message):
    if os.environ.get("TI_NATIVE_RUNTIME_LOAD_TRACE", "") == "1":
        print(f"[taichi-forge native loader] {message}", file=sys.stderr, flush=True)


def _dedupe_existing_dirs(paths):
    seen = set()
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(path)
        key = os.path.normcase(path)
        if key in seen or not os.path.isdir(path):
            continue
        seen.add(key)
        yield path


def _external_runtime_roots():
    spec = importlib.util.find_spec("taichi_forge_runtime")
    if spec is None or spec.submodule_search_locations is None:
        return []
    return list(spec.submodule_search_locations)


def _native_runtime_dirs():
    package_lib = os.path.join(package_root, "_lib")
    candidates = [
        os.environ.get("TAICHI_NATIVE_RUNTIME_DIR", ""),
        os.path.join(package_lib, "runtime_native"),
        os.path.join(package_lib, "core"),
    ]
    for root in _external_runtime_roots():
        auditwheel_lib_dir = os.path.join(os.path.dirname(root), os.path.basename(root) + ".libs")
        candidates.extend(
            [
                os.path.join(root, "native"),
                os.path.join(root, "runtime_native"),
                os.path.join(root, "_lib", "runtime_native"),
                auditwheel_lib_dir,
            ]
        )
    return list(_dedupe_existing_dirs(candidates))


def _runtime_bitcode_dir():
    candidates = [os.environ.get("TAICHI_RUNTIME_DIR", ""), os.path.join(package_root, "_lib", "runtime")]
    for root in _external_runtime_roots():
        candidates.extend([os.path.join(root, "runtime"), os.path.join(root, "_lib", "runtime")])
    for path in _dedupe_existing_dirs(candidates):
        return path
    return os.path.join(package_root, "_lib", "runtime")


def _native_runtime_library_name():
    if get_os_name() == "win":
        return "taichi_runtime.dll"
    if get_os_name() == "osx":
        return "libtaichi_runtime.dylib"
    return "libtaichi_runtime.so"


def _native_runtime_library_candidates(directory):
    exact = os.path.join(directory, _native_runtime_library_name())
    yield exact
    if get_os_name() == "linux":
        for path in sorted(glob.glob(os.path.join(directory, "libtaichi_runtime-*.so"))):
            if os.path.abspath(path) != os.path.abspath(exact):
                yield path


def _cuda_runtime_major_from_name(name):
    name = os.path.basename(name).lower()
    if get_os_name() == "win":
        match = re.fullmatch(r"cudart64_(\d+)\.dll", name)
    elif get_os_name() == "linux":
        match = re.fullmatch(
            r"(?:libcudart\.so\.|libcudart-[^.]+\.so\.)(\d+)(?:\.\d+)*",
            name,
        )
    else:
        return None
    return int(match.group(1)) if match else None


def _cuda_runtime_patterns(major=None):
    if get_os_name() == "win":
        version = str(major) if major is not None else "*"
        return [f"cudart64_{version}.dll"]
    if get_os_name() == "linux":
        if major is None:
            return ["libcudart.so.*", "libcudart-*.so.*"]
        return [f"libcudart.so.{major}*", f"libcudart-*.so.{major}*"]
    return []


def _cuda_runtime_library_candidates(directory, major=None):
    patterns = _cuda_runtime_patterns(major)

    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            abspath = os.path.abspath(path)
            key = os.path.normcase(abspath)
            if key in seen or not os.path.isfile(abspath):
                continue
            seen.add(key)
            yield abspath


def _bundled_cuda_runtime_major(runtime_dirs):
    manifest_majors = set()
    for directory in runtime_dirs:
        manifest = os.path.join(directory, _CUDA_RUNTIME_MAJOR_MANIFEST)
        if not os.path.isfile(manifest):
            continue
        try:
            with open(manifest, "r", encoding="ascii") as f:
                major = int(f.read().strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Invalid CUDA runtime manifest: {manifest}") from exc
        if major <= 0:
            raise RuntimeError(f"Invalid CUDA runtime major {major} in {manifest}")
        manifest_majors.add(major)

    if len(manifest_majors) > 1:
        raise RuntimeError(
            "Conflicting bundled CUDA runtime manifests: "
            f"{sorted(manifest_majors)}"
        )
    discovered_majors = {
        major
        for directory in runtime_dirs
        for path in _cuda_runtime_library_candidates(directory)
        if (major := _cuda_runtime_major_from_name(path)) is not None
    }
    if manifest_majors:
        manifest_major = next(iter(manifest_majors))
        unexpected_majors = discovered_majors - {manifest_major}
        if unexpected_majors:
            raise RuntimeError(
                "Bundled CUDA runtime libraries conflict with manifest major "
                f"{manifest_major}: {sorted(unexpected_majors)}"
            )
        return manifest_major

    # Backward compatibility for existing runtime wheels and source builds
    # created before the manifest was introduced. Runtime search roots never
    # include arbitrary system CUDA directories.
    if len(discovered_majors) > 1:
        raise RuntimeError(
            "Multiple bundled CUDA runtime majors found without a manifest: "
            f"{sorted(discovered_majors)}"
        )
    return next(iter(discovered_majors)) if discovered_majors else None


def _prepare_bundled_cuda_runtime(runtime_dirs):
    os.environ.pop("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", None)
    os.environ.pop("TI_CUDA_CUB_SORT_BUNDLED_CUDART_MAJOR", None)
    major = _bundled_cuda_runtime_major(runtime_dirs)
    if major is None:
        return
    for path in runtime_dirs:
        for lib_path in _cuda_runtime_library_candidates(path, major):
            os.environ["TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH"] = lib_path
            os.environ["TI_CUDA_CUB_SORT_BUNDLED_CUDART_MAJOR"] = str(major)
            return
    raise RuntimeError(
        f"Bundled CUDA runtime major {major} was declared but no matching "
        "library was found."
    )


def _reject_global_private_abi_collisions(runtime_dirs):
    if get_os_name() not in {"linux", "osx"}:
        return
    manifests = [
        os.path.join(directory, "taichi_runtime.exports.json")
        for directory in runtime_dirs
    ]
    manifests = [path for path in manifests if os.path.isfile(path)]
    if not manifests:
        return
    if len(manifests) != 1:
        raise RuntimeError(
            "Multiple split-runtime private ABI manifests were found: "
            f"{manifests}"
        )
    try:
        with open(manifests[0], encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "The split-runtime private ABI manifest is invalid"
        ) from exc
    probes = manifest.get("private_abi_collision_probe_symbols", [])
    if not probes:
        return
    if (
        manifest.get("schema_version") != 2
        or not isinstance(probes, list)
        or probes != sorted(set(probes))
        or not all(isinstance(symbol, str) and symbol for symbol in probes)
    ):
        raise RuntimeError(
            "The split-runtime private ABI collision probes are invalid"
        )
    process = ctypes.CDLL(None)
    collisions = []
    for symbol in probes:
        try:
            getattr(process, symbol)
        except AttributeError:
            continue
        collisions.append(symbol)
    if collisions:
        raise RuntimeError(
            "A process-global Taichi private ABI is already loaded; refusing "
            "unsafe symbol interposition: "
            + ", ".join(collisions[:8])
        )


def _preload_cuda_runtime_for_native_runtime():
    if get_os_name() != "linux":
        return
    flags = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 2)
    candidate = os.environ.get("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", "")
    if not candidate:
        return
    try:
        _native_load_trace(f"preloading CUDA runtime: {candidate}")
        _native_library_handles.append(ctypes.CDLL(candidate, mode=flags))
        _native_load_trace("CUDA runtime preload passed")
    except OSError:
        _native_load_trace("CUDA runtime preload was unavailable")
        return


def _prepare_native_runtime():
    global _native_runtime_loaded  # pylint: disable=global-statement
    global _loaded_native_runtime_path  # pylint: disable=global-statement
    if _native_runtime_loaded:
        return

    startup_profile_mark("native_runtime.search.begin")
    runtime_dirs = _native_runtime_dirs()
    startup_profile_mark("native_runtime.search.end")
    _native_load_trace(f"runtime search directories: {runtime_dirs}")
    _reject_global_private_abi_collisions(runtime_dirs)
    startup_profile_mark("native_runtime.cuda_dependency.begin")
    _prepare_bundled_cuda_runtime(runtime_dirs)
    _preload_cuda_runtime_for_native_runtime()
    startup_profile_mark("native_runtime.cuda_dependency.end")
    if get_os_name() == "win":
        for path in runtime_dirs:
            if hasattr(os, "add_dll_directory"):
                _dll_dir_handles.append(os.add_dll_directory(path))
            os.environ["PATH"] += os.pathsep + path

    for path in runtime_dirs:
        for lib_path in _native_runtime_library_candidates(path):
            if not os.path.exists(lib_path):
                continue
            _native_load_trace(f"loading native runtime: {lib_path}")
            startup_profile_mark("native_runtime.load.begin")
            if get_os_name() == "win":
                handle = ctypes.WinDLL(lib_path)  # pylint: disable=no-member
            else:
                flags = getattr(os, "RTLD_LOCAL", 0) | getattr(
                    os, "RTLD_NOW", 2
                )
                handle = ctypes.CDLL(lib_path, mode=flags)
            # Keep the explicit loader reference alive for the complete Python
            # process lifetime. On POSIX the shim has a direct dependency edge
            # to this locally loaded runtime, so its private C++ ABI is
            # available without entering the process-global symbol scope.
            _native_library_handles.append(handle)
            _native_runtime_loaded = True
            _loaded_native_runtime_path = os.path.realpath(lib_path)
            startup_profile_mark("native_runtime.load.end")
            _native_load_trace("native runtime load passed")
            return
    _native_load_trace("no native runtime candidate was found")


def get_os_name():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith("darwin") or name.lower().startswith("macos"):
        return "osx"
    if name.lower().startswith("windows"):
        return "win"
    if name.lower().startswith("linux"):
        return "linux"
    if "bsd" in name.lower():
        return "unix"
    assert False, f"Unknown platform name {name}"


def _python_core_dlopen_flags():
    flags = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 2)
    if not _native_runtime_loaded:
        flags |= getattr(os, "RTLD_DEEPBIND", 8)
    return flags


def import_ti_python_core():
    _prepare_native_runtime()
    old_flags = None
    if get_os_name() != "win":
        # pylint: disable=E1101
        old_flags = sys.getdlopenflags()
        dlopen_flags = _python_core_dlopen_flags()
        _native_load_trace(f"pybind shim dlopen flags: {dlopen_flags}")
        # The split runtime is a direct local dependency of the shim.
        # RTLD_DEEPBIND would prefer duplicate weak/inline definitions from the
        # shim and can split C++ singleton and type state across the two DSOs.
        # Keep DEEPBIND only for the historical monolithic import path.
        sys.setdlopenflags(dlopen_flags)
    else:
        pyddir = os.path.dirname(os.path.realpath(__file__))
        os.environ["PATH"] += os.pathsep + pyddir
    try:
        _native_load_trace("importing pybind shim")
        startup_profile_mark("pybind_shim.import.begin")
        from taichi_forge._lib.core import taichi_python as core  # pylint: disable=C0415
        startup_profile_mark("pybind_shim.import.end")
        _native_load_trace("pybind shim import passed")
    except Exception as e:
        if isinstance(e, ImportError):
            print(
                Fore.YELLOW + "Share object taichi_python import failed, "
                "check this page for possible solutions:\n"
                "https://docs.taichi-lang.org/docs/install" + Fore.RESET
            )
            if get_os_name() == "win":
                # pylint: disable=E1101
                e.msg += "\nConsider installing Microsoft Visual C++ Redistributable: https://aka.ms/vs/16/release/vc_redist.x64.exe"
        raise e from None
    finally:
        if old_flags is not None:
            sys.setdlopenflags(old_flags)  # pylint: disable=E1101
    lib_dir = _runtime_bitcode_dir()
    startup_profile_mark("runtime_bitcode.configure.begin")
    core.set_lib_dir(locale_encode(lib_dir))
    startup_profile_mark("runtime_bitcode.configure.end")
    return core


def locale_encode(path):
    try:
        import locale  # pylint: disable=C0415

        return path.encode(locale.getdefaultlocale()[1])
    except (UnicodeEncodeError, TypeError):
        try:
            return path.encode(sys.getfilesystemencoding())
        except UnicodeEncodeError:
            try:
                return path.encode()
            except UnicodeEncodeError:
                return path


def is_ci():
    return os.environ.get("TI_CI", "") == "1"


package_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_core_shared_object():
    directory = os.path.join(package_root, "_lib")
    return os.path.join(directory, "libtaichi_python.so")


def print_red_bold(*args, **kwargs):
    print(Fore.RED + Style.BRIGHT, end="")
    print(*args, **kwargs)
    print(Style.RESET_ALL, end="")


def print_yellow_bold(*args, **kwargs):
    print(Fore.YELLOW + Style.BRIGHT, end="")
    print(*args, **kwargs)
    print(Style.RESET_ALL, end="")


def check_exists(src):
    if not os.path.exists(src):
        raise FileNotFoundError(f'File "{src}" not exist. Installation corrupted or build incomplete?')


ti_python_core = import_ti_python_core()
startup_profile_mark("python_import.native_core_ready")

ti_python_core.set_python_package_dir(package_root)

log_level = os.environ.get("TI_LOG_LEVEL", "")
if log_level:
    ti_python_core.set_logging_level(log_level)


def get_dll_name(name):
    if get_os_name() == "linux":
        return f"libtaichi_{name}.so"
    if get_os_name() == "osx":
        return f"libtaichi_{name}.dylib"
    if get_os_name() == "win":
        return f"taichi_{name}.dll"
    raise Exception(f"Unknown OS: {get_os_name()}")


def at_startup():
    ti_python_core.set_core_state_python_imported(True)


at_startup()


def compare_version(latest, current):
    latest_num = map(int, latest.split("."))
    current_num = map(int, current.split("."))
    return tuple(latest_num) > tuple(current_num)


def _print_taichi_header():
    header = "[Taichi] "
    header += f"version {ti_python_core.get_version_string()}, "

    try:
        timestamp_path = os.path.join(ti_python_core.get_repo_dir(), "timestamp")
        if os.path.exists(timestamp_path):
            latest_version = ""
            with open(timestamp_path, "r") as f:
                latest_version = f.readlines()[1].rstrip()
            if compare_version(latest_version, ti_python_core.get_version_string()):
                header += f"latest version {latest_version}, "
    except:
        pass

    llvm_target_support = ti_python_core.get_llvm_target_support()
    header += f"llvm {llvm_target_support}, "

    commit_hash = ti_python_core.get_commit_hash()
    commit_hash = commit_hash[:8]
    header += f"commit {commit_hash}, "

    header += f"{get_os_name()}, "

    py_ver = ".".join(str(x) for x in sys.version_info[:3])
    header += f"python {py_ver}"

    print(header)


if os.getenv("ENABLE_TAICHI_HEADER_PRINT", "True").lower() not in ("false", "0", "f"):
    _print_taichi_header()


def try_get_wheel_tag(module):
    try:
        from email.parser import Parser  # pylint: disable=import-outside-toplevel

        wheel_path = f'{module.__path__[0]}-{".".join(map(str, module.__version__))}.dist-info/WHEEL'
        with open(wheel_path, "r") as f:
            meta = Parser().parse(f)
        return meta.get("Tag")
    except Exception:
        return None


def try_get_loaded_libc_version():
    assert platform.system() == "Linux"
    with open("/proc/self/maps") as f:
        content = f.read()

    try:
        libc_path = next(v for v in content.split() if "libc-" in v)
        ver = re.findall(r"\d+\.\d+", libc_path)
        if not ver:
            return None
        return tuple([int(v) for v in ver[0].split(".")])
    except StopIteration:
        return None


def try_get_pip_version():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pip  # pylint: disable=import-outside-toplevel
        return tuple([int(v) for v in pip.__version__.split(".")])
    except ImportError:
        return None


def warn_restricted_version():
    if os.environ.get("TI_MANYLINUX2014_OK", ""):
        return

    if get_os_name() == "linux":
        try:
            import taichi_forge as ti  # pylint: disable=import-outside-toplevel

            wheel_tag = try_get_wheel_tag(ti)
            if wheel_tag and "manylinux2014" in wheel_tag:
                print_yellow_bold(
                    "You have installed a restricted version of taichi, certain features (e.g. Vulkan & GGUI) will not work."
                )
                libc_ver = try_get_loaded_libc_version()
                if libc_ver and libc_ver < (2, 27):
                    print_yellow_bold(
                        "!! Taichi requires glibc >= 2.27 to run, please try upgrading your OS to a recent one (e.g. Ubuntu 18.04 or later) if possible."
                    )

                pip_ver = try_get_pip_version()
                if pip_ver and pip_ver < (20, 3, 0):
                    print_yellow_bold(
                        f"!! Your pip (version {'.'.join(map(str, pip_ver))}) is outdated (20.3.0 or later required), "
                        "try upgrading pip and install taichi again."
                    )
                    print()
                    print_yellow_bold("    $ python3 -m pip install --upgrade pip")
                    print_yellow_bold("    $ python3 -m pip install --force-reinstall taichi")
                    print()

                print_yellow_bold(
                    "You can suppress this warning by setting the environment variable TI_MANYLINUX2014_OK=1."
                )
        except Exception:
            pass
