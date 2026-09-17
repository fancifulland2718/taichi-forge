# PyPI 发行流程与权限排查

> 当前源码/包版本为 `0.6.3`，处于发布准备阶段，尚未正式发布。
> 以下命令是维护者操作说明，不表示已经上传、创建 tag 或完成全部平台验证。

发布入口只有一个，runtime 构建由它委托给可复用 workflow：

- [`publish_runtime_pypi.yml`](../../.github/workflows/publish_runtime_pypi.yml)
  构建平台级 `taichi-forge-runtime` artifact，支持 `workflow_call` 和手动构建，**不上传包**。
- [`publish_pypi.yml`](../../.github/workflows/publish_pypi.yml)
  先调用 runtime 构建，再构建 Python/pybind shim `taichi-forge`，完成整组验证后可选发布两个项目。

顺序为 **runtime artifact → shim 构建与安装验证 → 完整 wheel 集合 → 上传 → GitHub Release**。
shim 从本次 workflow 的 runtime artifact 解包 link artifacts，不依赖提前发布到 PyPI/TestPyPI。
上传两个项目并非索引端的原子事务；失败后应核对实际已上传文件，不要盲目重试或覆盖同版本。

下文以 `0.6.3` 为示例。其它版本须同步 `version.txt`、包元数据、workflow 输入或 tag
以及安装命令。标准发行的 runtime 与 shim 包版本相同；兼容性不以 Git commit 相等为条件。

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
| `taichi-forge-runtime` | `publish_pypi.yml` | `<仓库 owner>` | `<实际仓库名>` | `pypi` 或 `testpypi` |
| `taichi-forge` | `publish_pypi.yml` | `<仓库 owner>` | `<实际仓库名>` | `pypi` 或 `testpypi` |

两个项目都绑定到实际执行上传的 `publish_pypi.yml`，不是只产出 artifact 的 runtime workflow。
绑定完成后，workflow 里的 `pypa/gh-action-pypi-publish` 会通过 OIDC 向
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
| `LLVM20_WIN_SHA256`            | ✅ Windows 发行必需 | 对应 Windows zip 的 SHA256（64 位十六进制，不带前缀） |
| `LLVM20_LINUX_URL`             | ✅ Linux 发行必需   | LLVM 20 Linux zip 的公网 URL（manylinux 构建）；也可使用兼容别名 `LLVM20_LINUX_MANYLINUX_URL` |
| `LLVM20_LINUX_SHA256`          | ✅ Linux 发行必需   | 对应 Linux zip 的 SHA256（64 位十六进制，不带前缀） |

这些 URL 可以指向同一个项目的 "LLVM 20" Release 下的 asset，例如：
`https://github.com/<owner>/taichi/releases/download/llvm20/taichi-llvm-20-msvc2026.zip`

校验值对应下载的 LLVM artifact，不绑定 Forge commit HEAD。变更 URL 时同步核对该产物的校验值；不要用另一平台
或另一构建的值。公开 artifact 的 digest/随附 checksum 应与实际下载文件一致。

手动运行 `publish_pypi.yml` 并设置 `publish=false`，会构建、安装验证并汇总校验完整的两个平台 runtime 和
Python 3.10–3.14 shim wheel 集合，保存为 `validated-wheel-set` artifact。该模式不进入发布 environment，
不上传 PyPI/TestPyPI，也不创建 GitHub Release。只有完整集合通过后，显式发布才消费这份已验证的 artifact。

### 1.5 （备选）PAT fallback

如果默认 `GITHUB_TOKEN` 即使开启了 "Read and write" 依然无法创建 Release（比如组织
级策略覆盖），把一个 fine-grained PAT（权限：Contents: write）存为 `RELEASE_PAT`：
`publish_pypi.yml` 已经用 `${{ secrets.RELEASE_PAT || secrets.GITHUB_TOKEN }}` 优先
使用它。

### 1.6 单一平台 driver-only runtime wheel

`publish_runtime_pypi.yml` 构建不依赖 CUDA Toolkit runtime 的平台级
`taichi-forge-runtime` wheel，并显式启用：

```text
-DTI_WITH_AMDGPU:BOOL=ON
-DTI_WITH_CUDA_TOOLKIT:BOOL=OFF
-DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF
-DTI_WITH_CUPTI:BOOL=OFF
```

完整的用户侧合同和版本边界见 [构建 Wheel](../forge/build_wheels.zh.md)。维护者在本 workflow
中只需守住以下发行不变量：

- 每个平台恰好一个 `taichi-forge-runtime` wheel，不创建 `cu11` / `cu12` / `cu13` 包、extra、
  版本后缀或 wheel tag；
- Windows/Linux wheel 都只包含唯一 native runtime，不包含
  `cuda_runtime_major.txt` 或 CUDART，并通过
  `scripts/validate_runtime_wheel.py --dependency-class driver-only`；Linux 必须验证
  auditwheel 后的最终候选没有重新引入 CUDART；
- `.github/workflows/test_cuda_toolkit_reference.yml` 可以用 Toolkit 13.2 构建 CUB/CUDART
  对照 provider，但它不上传 wheel，也不能改变标准发行产物；
- driver-only 依赖扫描不能替代旧 driver 真机执行。修改最低驱动声明前必须完成
  [Linux 复测清单](../forge/linux_revalidation.zh.md)和目标旧 driver 实测。

AMDGPU 编入标准 Windows/Linux runtime，但 HIP runtime、驱动和 linker 由用户安装。
构建获取固定 HIP host 头文件及 LLVM 匹配的设备 bitcode；不把 HIP shared library、driver
或 vendor 数学库放入 wheel，也不要求其他后端加载它们。详见 [ROCm 指南](../forge/rocm_backend.zh.md)。

修改 `taichi/rhi/cuda/primitives/`、Program primitive arena、Vulkan native cache 或 CPU
native scratch 的提交会改变 native runtime，必须先重建 runtime artifact，再构建对应 shim；
shim-only 构建无法携带这些二进制更新。只改 Python 包装、测试或文档时可复用 ABI/行为匹配的
runtime。若该版本已经正式发布，应使用新版本，不覆盖旧文件。不得把 native provider 复制进 shim。

`publish_pypi.yml` 的各 CPython shim job 不重新编译 C++ runtime，也不安装 CUDA Toolkit。
runtime 只由被调用的 runtime workflow 构建一次/平台；shim job 消费其 artifact。

## 2. 触发方式

### 2.1 预演（不上传 PyPI）

```
Actions → Publish wheels to PyPI → Run workflow
  version: 0.6.3
  validation_platform: all
  publish: false
  target:  testpypi        (忽略，不会上传)
```

入口会构建 2 个 runtime 和 10 个 shim wheel（2 OS × Python 3.10–3.14），保存
`validated-wheel-set`。`publish=false` 不进入发布 environment、不上传包、不创建 Release，
也不要求对应版本已在索引中存在。候选可使用正式目标版本号，不能仅凭文件名视为已经发布。

只验证 Windows 时使用 `validation_platform=windows`、`publish=false`。可通过
`runtime_run_id` 复用已有运行的 `wheel-windows-runtime` artifact；仍需满足版本、native/provider
合同和编译器 ABI，不能只比较源码 commit。该模式不构成完整发行集合，也不允许发布。

只需构建 runtime 时，可手动运行 `Build runtime wheel artifacts`，传入 `version` 与
`platform=windows|linux|all`。此 workflow 没有 `publish` 或 `target` 输入。

### 2.2 TestPyPI（真正上传，但到沙箱）

```
Actions → Publish wheels to PyPI → Run workflow
  version: 0.6.3rc1
  validation_platform: all
  publish: true
  target:  testpypi
```

先在本次运行中完成整组构建与验证，再上传两个项目到 test.pypi.org。上传成功后创建
**draft** GitHub Release（非 tag 触发）。两个项目都需要对应的 Trusted Publisher。

安装验证：
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ taichi-forge==0.6.3rc1
```

### 2.3 生产发行（推 tag）

创建正式 tag 前，tag 所指向的 commit 必须已经把 `version.txt` 更新为 `v0.6.3`，并运行
`python scripts/sync_runtime_dependency.py`，使 `pyproject.toml` 精确依赖
`taichi-forge-runtime==0.6.3`。workflow 会再次同步构建工作区，但不能用这一临时覆盖替代
正式源码 tag 中的版本一致性。

```
git tag forge-v0.6.3
git push origin forge-v0.6.3
```

推 tag 会触发完整生产发布，不能用它代替无发布预演。tag 触发后 workflow 会：

1. 构建两个平台的 runtime artifact，不预先上传。
2. 构建 10 个 shim wheel，按各解释器安装并验证 artifact，再校验完整集合。
3. 确认目标版本在两个 PyPI 项目中尚未发布，再上传整组 wheel。
4. 上传成功后创建非 draft GitHub Release，附带自动生成说明与 wheel。

也可以手动选择 `publish=true`、`target=pypi`、`validation_platform=all`；这会真实上传，
但 GitHub Release 是 draft。两种发布方式择一，不能对同一版本重复上传。正式发布时更新
README、文档索引和发布说明中的待发布状态；本流程说明本身不授权执行发布。

## 3. 常见"无权限"问题速查

| 症状 | 原因 | 解决 |
| ---- | --- | ---- |
| `Error 403: Resource not accessible by integration` 在 `action-gh-release` | Workflow permissions 是只读 | Settings → Actions → General → Workflow permissions 改为 "Read and write" |
| `id-token: write not granted` | 工作流或作业级别缺 `permissions.id-token: write` | 已在顶层声明，检查是否在 job 里被覆盖 |
| PyPI 返回 `invalid-publisher` | Trusted Publisher 没绑定或环境名不匹配 | 按 §1.2 重新绑定，确认 `environment.name` 与 PyPI 侧配置一致 |
| 版本可用性检查拒绝或 PyPI 返回 `File already exists` | 重复上传或此前只完成部分上传 | 检查两个项目的实际文件；流程没有启用 `skip-existing`，不要覆盖或静默跳过，重新发布使用新版本 |
| Release step 成功但 asset 为空 | artifact download 失败 / path 不对 | 看 `Gather wheels` step 输出，确认 `dist/*.whl` 确实存在 |
| shim job 下载 runtime artifact 失败 | runtime job 失败、artifact 过期或复用 run ID 不正确 | 检查 `wheel-windows-runtime` / `wheel-linux-runtime` artifact 与运行权限，不要通过预上传 PyPI 绕过 |
| fork 触发 workflow 没有 id-token | fork 的 `pull_request` 默认没 OIDC 权限 | 改用 `workflow_dispatch` 或从 canonical repo 发起 |
| 标准 runtime 编译意外寻找 CUDA/CCCL 头文件 | production target 错误依赖了 Toolkit-reference source | 确认三个标准 flag 均为 OFF，并把 CUB/CUDART source 只留在独立 reference target |
| driver-only runtime wheel 出现 CUDART 或 manifest | release target 或 auditwheel 意外引入 Toolkit runtime 依赖 | 检查 CMake cache、PE import/ELF `DT_NEEDED` 与 `--dependency-class driver-only` 校验 |
| 旧 0.5.0 runtime wheel 被 validator 拒绝 | 对历史包错误使用了新发行的严格 dependency class | 兼容/repair 工具使用默认 `either`；只有新上传候选强制 `driver-only` |
| Linux shim 导入时报 `llvm::DisableABIBreakingChecks` 未定义 | prebuilt shim 只使用 LLVM headers 且不链接 LLVMSupport，却没有关闭 header 的 link sentinel | 保持 runtime/shim 分包边界；确认 Linux shim 定义 `LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1`，并让 `validate_shim_wheel.py` 拒绝残留 sentinel 的 wheel |
| Linux shim 找不到 `libtaichi_runtime.so`，但 runtime wheel 内存在 `libtaichi_runtime-<hash>.so` | auditwheel 错误地重命名了 wheel-owned 主 runtime；shim 的稳定 `DT_NEEDED` 无法解析该哈希名 | runtime repair 后必须保留 `taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so`；仅允许 grafted dependency 使用 hash 名，并运行 `validate_runtime_wheel.py --strict-binary` |
| Forge 与 Mesa/Intel 或其他原生库同进程时出现错误符号绑定 | Linux shim 缺少直接 runtime 依赖、runtime 被 `RTLD_GLOBAL` 加载，或最终 ELF 未应用私有 ABI version script | 检查 shim 的 `DT_NEEDED`/包相对 `RUNPATH`、runtime export manifest、`forbidden_export_families=[]`，并运行双加载顺序资格脚本；不要用 `LD_PRELOAD` 或调整 import 顺序掩盖问题 |

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
python -m pytest tests/python/test_amdgpu_wheel_contract.py tests/python/test_runtime_wheel_binary_contract.py -q
python -m pytest tests/python/test_runtime_statistics.py tests/python/test_primitive_plan.py -q
python -m py_compile misc/runtime_export_closure.py misc/generate_windows_runtime_export_closure.py misc/generate_elf_runtime_export_closure.py misc/generate_macho_runtime_export_closure.py scripts/repair_runtime_wheel.py scripts/validate_runtime_wheel.py scripts/validate_shim_wheel.py scripts/validate_installed_runtime.py scripts/validate_runtime_load_order.py
```

在各后端可用的 release-equivalent build/安装环境运行至少 30 秒生产尺度 primitive stress；
CUDA/Vulkan 执行前先确认没有其它 Python/GPU compute process：

```text
python tests/python/native_primitive_runtime_stress.py --arch cpu --seconds 30 --threads 4 --items 1048576
python tests/python/native_primitive_runtime_stress.py --arch cuda --seconds 30 --threads 4 --items 1048576
python tests/python/native_primitive_runtime_stress.py --arch vulkan --seconds 30 --threads 4 --items 1048576
```

要求 `result=pass`、空 fallback、正确 dependency class、clear 后 provider bytes 为 0。
stress 输出的 `performance=not_measured` 是预期值；它是正确性/并发/lifetime 门禁，不应被
包装成性能结果。性能结论只能由带 `--performance` idle guard 的 benchmark 单独产生。

workflow 产出后，必须对最终上传候选运行：

```text
python scripts/validate_runtime_wheel.py --wheel-dir <one-native-platform-runtime-dir> --platform <windows-or-manylinux> --dependency-class driver-only --require-vkfft --require-amdgpu --strict-binary
python scripts/validate_shim_wheel.py --wheel-dir <one-shim-wheel-dir> --platform <windows-or-manylinux> --strict-binary
```

随后让 pip 按 shim wheel 的 `Requires-Dist` 安装其 Python 依赖和本地同版本 runtime wheel，
运行 `pip check`；不得在最终安装验证中使用 `--no-deps`。再到仓库目录之外运行
`scripts/validate_installed_runtime.py`；Linux 还要运行
`scripts/validate_runtime_load_order.py`，确认 runtime-first 与 driver-first 都通过。新候选必须
包含通过 binary audit 的 `taichi_runtime.exports.json`，同时不得包含 CUDART 或
`cuda_runtime_major.txt`；历史 0.5.0 包内 CUDART wheel 只属于兼容路径。正式发布
还必须完成 [Linux 复测清单](../forge/linux_revalidation.zh.md) 中适用于发布环境的 GPU、
sanitizer、GGUI/interop 和性能稳定性门槛；仅 `import` 或 smoke test 不足以替代这些检查。

构建包含 AMDGPU、缺 HIP 时仍可使用 CPU/CUDA/Vulkan、AMD 实机 kernel 正确执行，是不同证据。
在兼容 AMD 设备上运行 `tests/python/test_amdgpu_basic.py`；没有 HIP/设备时的 skip 不算实机通过。
Windows-only 记录也不能被写成 Linux 或全矩阵通过。限定本轮验证范围时，保留未测项说明。
