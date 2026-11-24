# Taichi Forge 0.7.1 · 从 GitHub Release 安装

[English](install_release.en.md)

在 [Forge 0.7.1 Release 的 Assets](https://github.com/fancifulland2718/taichi-forge/releases/tag/forge-v0.7.1) 下载两个 wheel，放进当前目录下的 `wheels/` 文件夹。

| 下载项 | 如何选择 |
| --- | --- |
| `taichi_forge-0.7.1-…` Python 前端 | 使用常规 64 位 CPython 3.10–3.14。按解释器选择 `cp310`、`cp311`、`cp312`、`cp313` 或 `cp314`，再匹配操作系统。 |
| `taichi_forge_runtime-0.7.1-1-py3-none-…` 原生 Runtime | 选择相同系统即可。Windows 选 `win_amd64`，Linux 选 `manylinux_2_35_x86_64`（glibc ≥ 2.35）。版本号后的 `-1-` 是包含 Vulkan 字段布局修复的构建号。 |

## 新环境

以下以 Python 3.10 为例。换解释器时，同时替换创建环境命令中的版本和前端文件名的两处 `cp310`。两个本地 wheel 一起交给 pip，其他 Python 依赖由 pip 获取。

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

## 已安装 0.7.1

在应用原来的虚拟环境中，只需把下载的 Runtime wheel 强制重装一次。以下为 Linux 示例，Windows 将文件名换为上方的 `win_amd64` Runtime。安装后重新启动正在运行的 Python 进程。直接从 PyPI 安装 0.7.1 会取得原始构建。

```shell
python -m pip install --force-reinstall --no-deps ./wheels/taichi_forge_runtime-0.7.1-1-py3-none-manylinux_2_35_x86_64.whl
```

## 验证安装

新环境中将下面的 `python` 换成 `.\.venv\Scripts\python.exe`（Windows）或 `.venv/bin/python`（Linux）。第二条应输出 `0.7.1 1`。最后一条验证前端可加载 Runtime 并初始化 CPU。

```shell
python -m pip check
python -c "from email import message_from_string as parse; from importlib.metadata import distribution; d=distribution('taichi-forge-runtime'); print(d.version, parse(d.read_text('WHEEL'))['Build'])"
python -c "import taichi_forge as ti; ti.init(arch=ti.cpu); ti.reset()"
```
