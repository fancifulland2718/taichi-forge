# PyPI 发行流程与权限排查

本项目使用 [`publish_pypi.yml`](../../.github/workflows/publish_pypi.yml) 构建并发布
Windows + Linux 平台的 Python wheel 到 PyPI。本文说明触发方式、前置配置、以及最常见的
"workflow 编译通过但无法创建 Release" 问题的排查步骤。

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

| 字段 | 值 |
| ---- | -- |
| PyPI Project Name | `taichi` |
| Owner | `<仓库 owner>` |
| Repository name | `taichi` |
| Workflow filename | `publish_pypi.yml` |
| Environment name | `pypi` 或 `testpypi` |

绑定完成后，workflow 里的 `pypa/gh-action-pypi-publish@release/v1` 会通过 OIDC 向
PyPI 申请短期 token，**无需手动维护任何 secret**。

### 1.3 （备选）使用传统 API token

如果你的组织策略禁用了 OIDC / Trusted Publishing，需要：
- 在 PyPI 生成项目范围的 API token（`pypi-` 开头）。
- 作为 `PYPI_API_TOKEN`（以及 `TEST_PYPI_API_TOKEN`）保存到 GitHub Secrets。
- 修改 `publish_pypi.yml` 的 publish step，加上 `password: ${{ secrets.PYPI_API_TOKEN }}`。

### 1.4 仓库变量（Repo Variables）

在 Settings → Secrets and variables → **Variables** 标签页配置：

| 变量 | 必需 | 内容 |
| ---- | --- | ---- |
| `LLVM20_WIN_URL`               | ✅ Windows 发行必需 | LLVM 20 Windows zip 的公网 URL（由 `build_llvm20_windows.yml` 产出） |
| `LLVM20_LINUX_URL`             | ✅ Linux 发行必需   | LLVM 20 Linux zip 的公网 URL（manylinux_2_28 构建） |
| `LLVM20_LINUX_MANYLINUX_URL`   | 可选                | manylinux2014 变体（若同时发布两个 wheel 变体） |

这些 URL 可以指向同一个项目的 "LLVM 20" Release 下的 asset，例如：
`https://github.com/<owner>/taichi/releases/download/llvm20/taichi-llvm-20-msvc2026.zip`

### 1.5 （备选）PAT fallback

如果默认 `GITHUB_TOKEN` 即使开启了 "Read and write" 依然无法创建 Release（比如组织
级策略覆盖），把一个 fine-grained PAT（权限：Contents: write）存为 `RELEASE_PAT`：
`publish_pypi.yml` 已经用 `${{ secrets.RELEASE_PAT || secrets.GITHUB_TOKEN }}` 优先
使用它。

## 2. 触发方式

### 2.1 预演（不上传 PyPI）

```
Actions → Publish wheels to PyPI → Run workflow
  version: 1.8.0.dev20260424
  publish: false
  target:  testpypi        (忽略，不会上传)
```

会产出 8 个 wheel artifacts（2 OS × 4 Python）但**不会**创建 Release、不会上传 PyPI。
用来快速验证构建本身。

### 2.2 TestPyPI（真正上传，但到沙箱）

```
Actions → Publish wheels to PyPI → Run workflow
  version: 1.8.0rc1
  publish: true
  target:  testpypi
```

会创建 **draft** GitHub Release（因为不是 tag 触发），并把 wheels 推到 test.pypi.org。
安装验证：
```
pip install -i https://test.pypi.org/simple/ taichi==1.8.0rc1
```

### 2.3 生产发行（推 tag）

```
git tag v1.8.0
git push origin v1.8.0
```

tag 触发后 workflow 会：
1. 8 个 wheel 并行构建（Windows + Linux × Python 3.10–3.13）。
2. 自动合成 GitHub Release（非 draft，包含自动生成的 release notes）。
3. 推到生产 PyPI（需要 `pypi` environment 的 Trusted Publisher 已绑定）。

## 3. 常见"无权限"问题速查

| 症状 | 原因 | 解决 |
| ---- | --- | ---- |
| `Error 403: Resource not accessible by integration` 在 `action-gh-release` | Workflow permissions 是只读 | Settings → Actions → General → Workflow permissions 改为 "Read and write" |
| `id-token: write not granted` | 工作流或作业级别缺 `permissions.id-token: write` | 已在顶层声明，检查是否在 job 里被覆盖 |
| PyPI 返回 `invalid-publisher` | Trusted Publisher 没绑定或环境名不匹配 | 按 §1.2 重新绑定，确认 `environment.name` 与 PyPI 侧配置一致 |
| PyPI 返回 `File already exists` | 重复上传同版本 | 使用 `skip-existing: true`（已启用），或改版本号 |
| Release step 成功但 asset 为空 | artifact download 失败 / path 不对 | 看 `Gather wheels` step 输出，确认 `dist/*.whl` 确实存在 |
| fork 触发 workflow 没有 id-token | fork 的 `pull_request` 默认没 OIDC 权限 | 改用 `workflow_dispatch` 或从 canonical repo 发起 |

## 4. 和 LLVM 20 的关系

- Publish workflow 不会即时编译 LLVM 20（太慢，6 小时超时）。
- 改为先跑一次 [`build_llvm20_windows.yml`](../../.github/workflows/build_llvm20_windows.yml)
  产出 `dist/taichi-llvm-20-msvc2026.zip`（发到 `llvm20` tag 下），然后把该 asset 的
  URL 填到 `LLVM20_WIN_URL` repo variable 里。
- Linux 端同理，需要在一台 manylinux_2_28 容器里 build LLVM 20 并发到 Release，再
  设置 `LLVM20_LINUX_URL`。

## 5. Smoke test 建议

每次 publish 前在本地至少跑一次：

```powershell
# Windows
$env:LLVM_DIR = "D:\taichi\dist\taichi-llvm-20\lib\cmake\llvm"
python build.py --python 3.10
python -m pytest tests/python/test_offline_cache.py -x -q
```

CI 上 wheels 构建完毕后下载一个 artifact 本地 `pip install` 并导入 `import taichi_forge as ti; ti.init()`
以确认运行时加载正常——wheel tag 对不上 Python/OS 是最常见的隐性发行 bug。
