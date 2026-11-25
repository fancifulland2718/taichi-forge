# Taichi Forge 0.7.1 · 从 GitHub Release 安装

[English](install_release.en.md)

在 [Forge 0.7.1 Release 的 Assets](https://github.com/fancifulland2718/taichi-forge/releases/tag/forge-v0.7.1) 下载两个 wheel，放进当前目录下的 `wheels/` 文件夹。

| 下载项 | 如何选择 |
| --- | --- |
| `taichi_forge-0.7.1-2-…` Python 前端 | 使用常规 64 位 CPython 3.10–3.14。按解释器选择 `cp310`、`cp311`、`cp312`、`cp313` 或 `cp314`，再匹配操作系统。 |
| `taichi_forge_runtime-0.7.1-2-py3-none-…` 原生 Runtime | 选择相同系统即可。Windows 选 `win_amd64`，Linux 选 `manylinux_2_35_x86_64`（glibc ≥ 2.35）。构建 2 包含 Vulkan 字段布局修复和 OptiX 9.0/9.1 驱动适配。 |

## 新环境

以下以 Python 3.10 为例。换解释器时，同时替换创建环境命令中的版本和前端文件名的两处 `cp310`。两个本地 wheel 一起交给 pip，其他 Python 依赖由 pip 获取。

**Windows x64 · PowerShell**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ./wheels/taichi_forge-0.7.1-2-cp310-cp310-win_amd64.whl ./wheels/taichi_forge_runtime-0.7.1-2-py3-none-win_amd64.whl
```

**Linux x86_64 · Bash**
```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install ./wheels/taichi_forge-0.7.1-2-cp310-cp310-manylinux_2_35_x86_64.whl ./wheels/taichi_forge_runtime-0.7.1-2-py3-none-manylinux_2_35_x86_64.whl
```

## 已安装 0.7.1

在应用原来的虚拟环境中重装前端和 Runtime 两份构建 2 的 wheel。下面以 Linux、Python 3.10 为例，其他 Python 版本需替换前端标记，Windows 使用上方的两个 Windows 文件名。重启正在运行的 Python 进程。PyPI 提供原始 0.7.1 构建。

```shell
python -m pip install --force-reinstall --no-deps ./wheels/taichi_forge-0.7.1-2-cp310-cp310-manylinux_2_35_x86_64.whl ./wheels/taichi_forge_runtime-0.7.1-2-py3-none-manylinux_2_35_x86_64.whl
```

## 验证安装

新环境中将下面的 `python` 换成 `.\.venv\Scripts\python.exe`（Windows）或 `.venv/bin/python`（Linux）。第二条应输出 `[('0.7.1', '2'), ('0.7.1', '2')]`。最后一条验证前端可加载 Runtime 并初始化 CPU。

```shell
python -m pip check
python -c "from email import message_from_string as parse; from importlib.metadata import distribution; print([(d.version, parse(d.read_text('WHEEL'))['Build']) for d in map(distribution, ('taichi-forge','taichi-forge-runtime'))])"
python -c "import taichi_forge as ti; ti.init(arch=ti.cpu); ti.reset()"
```
