# Taichi Forge 0.7.1 · Install from GitHub Releases

[中文](install_release.zh.md)

Download two wheels from the [Forge 0.7.1 release assets](https://github.com/fancifulland2718/taichi-forge/releases/tag/forge-v0.7.1) into a folder named `wheels/` in your current directory.

| Download | How to choose |
| --- | --- |
| `taichi_forge-0.7.1-…` Python frontend | Use standard 64-bit CPython 3.10–3.14. Match both the interpreter tag (`cp310` through `cp314`) and the operating system. |
| `taichi_forge_runtime-0.7.1-1-py3-none-…` native Runtime | Match the operating system. Choose `win_amd64` for Windows or `manylinux_2_35_x86_64` for Linux (glibc ≥ 2.35). The `-1-` after the version identifies the build containing the Vulkan field layout fix. |

## New environment

The commands use Python 3.10. For another supported interpreter, change the venv command and both occurrences of `cp310` in the frontend filename. Pass both local wheels to pip in the same command. Pip fetches the other Python dependencies.

**Windows x64 · PowerShell**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ./wheels/taichi_forge-0.7.1-cp310-cp310-win_amd64.whl ./wheels/taichi_forge_runtime-0.7.1-1-py3-none-win_amd64.whl
```

**Linux x86_64 · Bash**
```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install ./wheels/taichi_forge-0.7.1-cp310-cp310-manylinux_2_35_x86_64.whl ./wheels/taichi_forge_runtime-0.7.1-1-py3-none-manylinux_2_35_x86_64.whl
```

## Existing 0.7.1 environment

Use the Python executable from your application's environment to reinstall the downloaded Runtime. The Linux example below also applies to Windows with the `win_amd64` Runtime filename shown above. Restart any running Python process after installation. PyPI serves the original Runtime build for this version.

```shell
python -m pip install --force-reinstall --no-deps ./wheels/taichi_forge_runtime-0.7.1-1-py3-none-manylinux_2_35_x86_64.whl
```

## Verify the installation

For a new environment, use `.\.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux) in place of `python` below. The second command should print `0.7.1 1`. The last command loads the native Runtime and initializes the CPU backend.

```shell
python -m pip check
python -c "from email import message_from_string as parse; from importlib.metadata import distribution; d=distribution('taichi-forge-runtime'); print(d.version, parse(d.read_text('WHEEL'))['Build'])"
python -c "import taichi_forge as ti; ti.init(arch=ti.cpu); ti.reset()"
```
