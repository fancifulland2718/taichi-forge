# PyPI 发行流程与权限排查

本项目使用两个独立 workflow 发布 PyPI 包：

- [`publish_runtime_pypi.yml`](../../.github/workflows/publish_runtime_pypi.yml)
  构建并发布平台级 `taichi-forge-runtime` wheel。
- [`publish_pypi.yml`](../../.github/workflows/publish_pypi.yml)
  构建并发布 Python/pybind shim `taichi-forge` wheel。

发布顺序必须是 **runtime 先发，shim 后发**。`publish_pypi.yml` 会从 PyPI/TestPyPI
下载同版本的 `taichi-forge-runtime` wheel，解包出 native runtime link artifacts，再构建
各 Python 版本的 shim wheel。

下文以 `0.5.0` 为发布示例。发布其它版本时必须统一替换 runtime workflow 输入、shim
workflow 输入或 tag，以及安装验证命令中的版本。runtime 与 shim 版本必须完全一致。
`publish_runtime_pypi.yml` 仅由 `workflow_dispatch` 触发；若仓库 `version.txt` 尚未更新，
正式发布时必须显式填写版本，不能留空依赖旧 fallback。

## 1. 一次成功发行需要的全部前置条件

### 1.1 GitHub 仓库设置

- **Settings → Actions → General → Workflow permissions** 必须设置为 "Read and write
  permissions"（默认只读），否则 `GITHUB_TOKEN` 无法创建 Release、无法 push tag。
  - 症状：Release step 报 `403 Resource not accessible by integration`。
- **Settings → Environments** 新建两个环境：
  - `testpypi` — 绑定到 TestPyPI 的 Trusted Publisher。
  - `pypi`     — 绑定到生产 PyPI 的 Trusted Publisher。
  - 可以为 `pypi` 配置 "Required reviewers" 做最后一道人工 gate。

### 1.2 GitHub → PyPI 的 Trusted Publishing 绑定（**推荐**，比 API token 更安全）

在 PyPI（或 test.pypi.org）上为该项目添加一个 Trusted Publisher：

| PyPI Project Name | Workflow filename | Owner | Repository name | Environment name |
| ---- | ---- | ---- | ---- | ---- |
| `taichi-forge-runtime` | `publish_runtime_pypi.yml` | `<仓库 owner>` | `taichi-forge` | `pypi` 或 `testpypi` |
| `taichi-forge` | `publish_pypi.yml` | `<仓库 owner>` | `taichi-forge` | `pypi` 或 `testpypi` |

绑定完成后，workflow 里的 `pypa/gh-action-pypi-publish@release/v1` 会通过 OIDC 向
PyPI 申请短期 token，**无需手动维护任何 secret**。

### 1.3 （备选）使用传统 API token

如果你的组织策略禁用了 OIDC / Trusted Publishing，需要：
- 在 PyPI 生成项目范围的 API token（`pypi-` 开头）。
- 作为 `PYPI_API_TOKEN`（以及 `TEST_PYPI_API_TOKEN`）保存到 GitHub Secrets。
- 修改对应 workflow 的 publish step，加上 `password: ${{ secrets.PYPI_API_TOKEN }}`。

### 1.4 仓库变量（Repo Variables）

在 Settings → Secrets and variables → **Variables** 标签页配置：

| 变量 | 必需 | 内容 |
| ---- | --- | ---- |
| `LLVM20_WIN_URL`               | ✅ Windows 发行必需 | LLVM 20 Windows zip 的公网 URL（由 `build_llvm20_windows.yml` 产出） |
| `LLVM20_LINUX_MANYLINUX_URL`   | ✅ Linux 发行必需   | LLVM 20 Linux zip 的公网 URL（manylinux 构建） |

这些 URL 可以指向同一个项目的 "LLVM 20" Release 下的 asset，例如：
`https://github.com/<owner>/taichi/releases/download/llvm20/taichi-llvm-20-msvc2026.zip`

### 1.5 （备选）PAT fallback

如果默认 `GITHUB_TOKEN` 即使开启了 "Read and write" 依然无法创建 Release（比如组织
级策略覆盖），把一个 fine-grained PAT（权限：Contents: write）存为 `RELEASE_PAT`：
`publish_pypi.yml` 已经用 `${{ secrets.RELEASE_PAT || secrets.GITHUB_TOKEN }}` 优先
使用它。

### 1.6 单一平台 runtime wheel 与 CUDA 构建基线

`publish_runtime_pypi.yml` 是唯一需要安装 CUDA Toolkit 的发布 workflow。它构建平台级
`taichi-forge-runtime` wheel，并启用：

```text
-DTI_WITH_CUDA_TOOLKIT:BOOL=ON
-DTI_CUDA_CUB_SORT_DYNAMIC_CUDART:BOOL=ON
```

完整的用户侧合同和版本边界见 [构建 Wheel](../forge/build_wheels.zh.md)。维护者在本 workflow
中只需守住以下发行不变量：

- 每个平台恰好一个 `taichi-forge-runtime` wheel，不创建 `cu11` / `cu12` / `cu13` 包、extra、
  版本后缀或 wheel tag；
- Windows/Linux wheel 都包含唯一 native runtime、唯一动态 CUDART 和
  `cuda_runtime_major.txt`，并通过 `scripts/validate_runtime_wheel.py`；Linux 必须验证
  auditwheel 后的最终候选。auditwheel 生成 hashed `.libs` CUDART 后，workflow 会删除
  已被替代的 raw CUDART、重写 `RECORD`，再执行 `auditwheel show` 和单副本校验；
- `CUDA_TOOLKIT_VERSION` 当前默认 `13.2.0`，只作为可替换的内部构建基线。降低前必须完成
  [Linux 复测清单](../forge/linux_revalidation.zh.md)和目标旧 driver 实测，不能只凭编译成功
  修改对外最低驱动声明。

`publish_pypi.yml` 不应重新编译 C++ runtime，也不应重新安装 CUDA Toolkit。它只从目标
PyPI/TestPyPI 下载指定版本的 `taichi-forge-runtime` wheel，解包 link artifacts，然后构建
各 Python 版本的 pybind shim wheel。

## 2. 触发方式

### 2.1 预演（不上传 PyPI）

```
Actions → Publish runtime wheels to PyPI → Run workflow
  version: 0.5.0.dev20260714
  publish: false
  target:  testpypi        (忽略，不会上传)

Actions → Publish wheels to PyPI → Run workflow
  version: 0.5.0.dev20260714
  publish: false
  target:  testpypi        (忽略，不会上传)
```

runtime workflow 会产出 2 个 runtime wheel artifacts（Windows + Linux）。shim workflow
会产出 10 个 Python wheel artifacts（2 OS × Python 3.10-3.14）。`publish=false` 时不会
上传 PyPI；但 shim workflow 仍然需要对应版本的 `taichi-forge-runtime` 已经存在于目标索引，
否则无法从 wheel 中解包 link artifacts。

### 2.2 TestPyPI（真正上传，但到沙箱）

```
Actions → Publish runtime wheels to PyPI → Run workflow
  version: 0.5.0rc1
  publish: true
  target:  testpypi

Actions → Publish wheels to PyPI → Run workflow
  version: 0.5.0rc1
  publish: true
  target:  testpypi
```

先把 runtime wheels 推到 test.pypi.org，然后 shim workflow 从 TestPyPI 下载 runtime wheel
并构建/上传 `taichi-forge` wheels。shim workflow 会创建 **draft** GitHub Release（因为不是
tag 触发）。

安装验证：
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ taichi-forge==0.5.0rc1
```

### 2.3 生产发行（推 tag）

创建正式 tag 前，tag 所指向的 commit 必须已经把 `version.txt` 更新为 `v0.5.0`，并运行
`python scripts/sync_runtime_dependency.py`，使 `pyproject.toml` 精确依赖
`taichi-forge-runtime==0.5.0`。workflow 会再次同步构建工作区，但不能用这一临时覆盖替代
正式源码 tag 中的版本一致性。

```
Actions → Publish runtime wheels to PyPI → Run workflow
  version: 0.5.0
  publish: true
  target:  pypi

git tag v0.5.0
git push origin v0.5.0
```

生产发布也要先跑 runtime workflow，确认 `taichi-forge-runtime==0.5.0` 已经在 PyPI
可下载；随后推 tag 触发 `publish_pypi.yml` 构建 shim wheels。tag 触发后 shim workflow 会：
1. 从 PyPI 下载并解包同版本 runtime wheel。
2. 10 个 shim wheel 并行构建（Windows + Linux × Python 3.10-3.14）。
3. 自动合成 GitHub Release（非 draft，包含自动生成的 release notes）。
4. 推到生产 PyPI（需要 `pypi` environment 的 Trusted Publisher 已绑定）。

## 3. 常见"无权限"问题速查

| 症状 | 原因 | 解决 |
| ---- | --- | ---- |
| `Error 403: Resource not accessible by integration` 在 `action-gh-release` | Workflow permissions 是只读 | Settings → Actions → General → Workflow permissions 改为 "Read and write" |
| `id-token: write not granted` | 工作流或作业级别缺 `permissions.id-token: write` | 已在顶层声明，检查是否在 job 里被覆盖 |
| PyPI 返回 `invalid-publisher` | Trusted Publisher 没绑定或环境名不匹配 | 按 §1.2 重新绑定，确认 `environment.name` 与 PyPI 侧配置一致 |
| PyPI 返回 `File already exists` | 重复上传同版本 | 使用 `skip-existing: true`（已启用），或改版本号 |
| Release step 成功但 asset 为空 | artifact download 失败 / path 不对 | 看 `Gather wheels` step 输出，确认 `dist/*.whl` 确实存在 |
| shim workflow 下载 runtime 失败 | 同版本 `taichi-forge-runtime` 尚未发布到目标索引 | 先运行 `publish_runtime_pypi.yml`，并确认目标是同一个 `pypi` / `testpypi` |
| fork 触发 workflow 没有 id-token | fork 的 `pull_request` 默认没 OIDC 权限 | 改用 `workflow_dispatch` 或从 canonical repo 发起 |
| Linux runtime 编译报缺少 CUDA/CCCL 头文件 | 所选 Toolkit 与 native 实现仍有未移除的版本专属依赖 | 保持单一包名不变；补齐兼容 adapter，并在该 Toolkit 的 Linux 构建中复测 |
| Windows CUDA 版本校验误报 | `nvcc -V` 是多行输出，或校验字符串被硬编码 | 把输出 join 成字符串，并从 `CUDA_TOOLKIT_VERSION` 推导 `major.minor` |
| runtime wheel 缺 CUDART 或 `cuda_runtime_major.txt` | runtime 构建没有启用动态 cudart，或 repair/打包路径错误 | 检查 `TI_WITH_CUDA_TOOLKIT`、`TI_CUDA_CUB_SORT_DYNAMIC_CUDART` 和 wheel contents validate step |
| manylinux wheel 出现 raw 与 hashed 两份 CUDART | auditwheel 已复制并改写依赖，但 raw 数据文件副本尚未规范化 | 确认 auditwheel 后运行 `repair_runtime_wheel.py --platform manylinux`，再执行 `auditwheel show` 和严格校验 |
| Linux shim 导入时报 `llvm::DisableABIBreakingChecks` 未定义 | prebuilt shim 只使用 LLVM headers 且不链接 LLVMSupport，却没有关闭 header 的 link sentinel | 保持 runtime/shim 分包边界；确认 Linux shim 定义 `LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1`，并让 `validate_shim_wheel.py` 拒绝残留 sentinel 的 wheel |

## 4. 和 LLVM 20 的关系

- Publish workflow 不会即时编译 LLVM 20（太慢，6 小时超时）。
- 改为先跑一次 [`build_llvm20_windows.yml`](../../.github/workflows/build_llvm20_windows.yml)
  产出 `dist/taichi-llvm-20-msvc2026.zip`（发到 `llvm20` tag 下），然后把该 asset 的
  URL 填到 `LLVM20_WIN_URL` repo variable 里。
- Linux 端同理，需要在 manylinux 容器里 build LLVM 20 并发到 Release，再
  设置 `LLVM20_LINUX_MANYLINUX_URL`。

## 5. 发布前验证

源码侧最低门槛：

```powershell
python -m pytest tests/python/test_runtime_packaging_cuda_version.py -q
python -m py_compile scripts/repair_runtime_wheel.py scripts/validate_runtime_wheel.py scripts/validate_shim_wheel.py scripts/validate_installed_runtime.py
```

workflow 产出后，必须对最终上传候选运行：

```text
python scripts/validate_runtime_wheel.py --wheel-dir <runtime-wheel-dir> --platform pair
python scripts/validate_shim_wheel.py --wheel-dir <one-shim-wheel-dir> --platform <windows-or-manylinux>
```

随后让 pip 按 shim wheel 的 `Requires-Dist` 安装其 Python 依赖和本地同版本 runtime wheel，
运行 `pip check`；不得在最终安装验证中使用 `--no-deps`。再到仓库目录之外运行
`scripts/validate_installed_runtime.py`，并确认实际 CUDART 来自 runtime package。正式发布
还必须完成 [Linux 复测清单](../forge/linux_revalidation.zh.md) 中适用于发布环境的 GPU、
sanitizer、GGUI/interop 和性能稳定性门槛；仅 `import` 或 smoke test 不足以替代这些检查。
