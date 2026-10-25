# GitHub Release 与 PyPI 标准发布流程

[English](pypi_release.en.md) · [构建 wheel](../forge/build_wheels.zh.md)

本文以 **0.7.0** 为例，不表示版本已经上线。沿用已经绑定的两个 PyPI 发布入口：
[publish_runtime_pypi.yml](../../.github/workflows/publish_runtime_pypi.yml) 发布 runtime，
[publish_pypi.yml](../../.github/workflows/publish_pypi.yml) 发布主包。
新增 [release.yml](../../.github/workflows/release.yml) **只负责 GitHub Release**，在主包
成功发布到生产 PyPI 后自动接续，不构建 wheel，也不上传 PyPI。
跨服务没有原子事务；这里的同步是先确认索引文件，再将相同字节的 wheel 与公开附件发布到 GitHub。

## 交付物

| 位置 | 文件及用途 |
| --- | --- |
| PyPI 与 GitHub | 2 个 runtime 平台 wheel，10 个 CPython 3.10–3.14 shim wheel；Windows x86_64 与 manylinux_2_35 x86_64 |
| GitHub | `taichi-forge-0.7.0-source.tar.gz`：固定源码、递归子模块、许可证、构建脚本、公开测试/API 示例、`SOURCE_COMMIT`、`SOURCE_MANIFEST.json` |
| GitHub | `taichi-forge-0.7.0-docs.zip`：公开文档、中英文 README、许可证 |
| GitHub | `RELEASE_NOTES.md`：同一文件先中文后英文，也作为 Release 正文 |
| GitHub | `release-manifest.json`：版本、commit、来源运行、附件大小/SHA256、子模块来源、实际 LLVM 下载 URL 与 SHA256 |
| GitHub | `SHA256SUMS`：全部其他附件（含 manifest）的校验和 |

完整源码从 Git 对象及固定子模块生成，不使用未提交/忽略文件，根目录 `internal/`、
`opt_doc/` 和临时目录不打包。供应商源码内必要的内部头文件保留，许可证随源码保留。
GitHub 自动 Source code ZIP 缺少子模块，不能替代完整源码包。PyPI 仅发布 wheel；
完整源码按构建文档使用，不是可由 pip 自动配置平台 SDK 的 sdist。编译器、SDK、
外部库由用户安装，商业 SDK/runtime 不重新打包。LLVM 公网下载及源码构建脚本公开。
独立 C/C++ AOT SDK 可从同一源码按 [SDK 文档](../forge/aot_sdk.zh.md)构建，
未经独立安装验证的 SDK 二进制不列为预编译发行物。

## 保留现有发布绑定

| PyPI / TestPyPI 项目 | 已有 Trusted Publisher workflow |
| --- | --- |
| `taichi-forge-runtime` | `publish_runtime_pypi.yml` |
| `taichi-forge` | `publish_pypi.yml` |

**已有绑定不需要修改；不要改绑到 `release.yml`。** owner 为 `fancifulland2718`，repository
为 `taichi-forge`，environment 仍为相应的 `pypi` / `testpypi`。两个原上传 job 使用
`id-token: write`，无需长期 PyPI token。GitHub Release job 只需 `contents: write` 与
读取 Actions 产物的权限，不需要 PyPI 环境或 OIDC 上传身份。

继续配置 `LLVM20_WIN_URL`、`LLVM20_WIN_SHA256`、`LLVM20_LINUX_URL`、
`LLVM20_LINUX_SHA256`（Linux URL 支持 `LLVM20_LINUX_MANYLINUX_URL` 别名）。
URL 使用公开 HTTPS，摘要为 SHA256。源码构建脚本：[Windows](../../.github/workflows/build_llvm20_windows.yml)、
[Linux](../../.github/workflows/build_llvm20_linux.yml)。实际下载摘要与编译配置由构建 job 记录。

## 准备与发布顺序

1. 更新 `version.txt`、CMake 版本、native header、runtime 依赖和两份
   `docs/forge/release_notes.*.md`，提交修改。ABI/格式编号独立维护。
   两包可以独立发布；复用兼容 runtime 时显式设置 `runtime_version`，不重命名旧 wheel。
2. 用原 runtime workflow 构建、验证并发布 runtime：`platform=all`、`publish=true`、
   `target=pypi`。`publish=false` 仅构建验证；`platform=windows` 用于单平台诊断，不能发布。
   成功上传后保存 `runtime-publication-<version>-pypi` Actions 元数据产物（90 天），
   包含已发布 wheel 的哈希、实际源码 commit、LLVM 和构建约束；它不进入 PyPI。
3. 原主包 workflow 用已发布 runtime 链接、安装并验证十个 CPython shim。可先手动
   `publish=false` 预演；`wheel_run_id` 可复用该流程的 `validated-shim-wheel-set`，
   其中 `shim-build-info.json` 保留实际构建来源。选择 `target` 本身不会启用上传。
4. 推送匹配的版本 tag，自动触发主包正式发布。runtime 必须已经可从生产 PyPI 获取：

```bash
git tag -a forge-v0.7.0 -m "Taichi Forge 0.7.0"
git push origin forge-v0.7.0
```

也可在已有 tag 上手动运行原主包 workflow，设置 `publish=true`、`target=pypi` 和可选
`wheel_run_id`。复用的 shim 实际构建 commit 必须与 tag 一致，避免和 tag 自动运行重复提交。
`runtime_source=artifact` 与数字 `runtime_run_id` 仍可用于验证；正式发布前它必须已经发布。
`validation_platform=windows` 不发布。TestPyPI 继续由原入口处理，不触发正式 GitHub Release。

主包 PyPI 上传成功后自动调用 GitHub-only workflow：获取已验证 shim 与 PyPI runtime →
核对全套 wheel 的已发布哈希、原发布 job、实际源码身份与 tag → 整理完整源码、公开文档、
双语说明和校验和 → 补齐 GitHub draft → 公开 Release。只将两个项目的 `.whl` 放进 PyPI
上传目录，源码、文档、JSON 和校验和均只作为 GitHub 附件。

源码归档对应 tag 的主包源码。独立发布的兼容 runtime 可以来自另一 commit / 版本，其实际
构建输入、commit 和公开源码链接单独记录在 `release-manifest.json`；不伪称它与主包同源。
两平台 runtime 必须来自同一个 runtime 构建 commit。

最终 wheel 仍须完成 binary audit、`pip check`、源码目录外导入及 Python/NumPy ABI 检查。
按本次变更选择代表性 CPU/CUDA/Vulkan 验证并如实记录未测范围；CI build/smoke 不等同
完整 GPU 验收，Linux 按[复测清单](../forge/linux_revalidation.zh.md)处理。

## 优先复用已验证产物

两个原 workflow 默认 `reuse=true`。完整矩阵运行会先读取小型来源清单，在同仓库查找
**相同源码 commit、包版本，以及 shim 的 runtime 精确依赖**；候选完整、未过期且验证 job
成功时，直接下载其 wheel 集合并复核，跳过 native 或 CPython 编译。tag 触发的主包发布也
会查找相同 commit 的预演产物。上传/Release 失败不会使已经通过的验证失效。

两者均可显式填写 `wheel_run_id`（各自 workflow 的已验证集合编号）。runtime 因此可以先
`publish=false` 完成完整构建，再指定该编号 `publish=true`，无需再次编译。复用记录保留
原构建 commit/run、LLVM 与配置；发布运行不会冒充构建运行。改变版本、损坏或缺失文件
会被拒绝；自动查找无合格候选时正常构建。候选来源和 wheel 集合均保留 90 天。

如需使用变更后的仓库工具链变量或强制重新资格化，设 `reuse=false` 并留空 `wheel_run_id`。
自动匹配复用原产物的已记录工具链，不会因仓库变量改变而重新解释它。单平台诊断仍按原路径
构建，不把部分矩阵冒充可发布集合。GitHub Release 始终只消费已发布的产物，不编译。

## 失败恢复与附件预演

- 原 PyPI 发布失败：在原运行 **Re-run failed jobs**，沿用原 wheel，不重新编译。
  同名同 SHA256 跳过，只补缺失文件；不同字节或 yanked 报错，不盲目 skip-existing。
- PyPI 成功、GitHub 阶段失败：先重跑失败 job。也可手动运行 **GitHub Release / GitHub 发行**，
  提供 `version`、`runtime_version`、成功上传主包的 `shim_run_id`。
  `publish=false` 只生成附件 bundle（两包须已在生产 PyPI），`publish=true` 补齐并公开 GitHub；
  两种情况都不会上传 PyPI 或重编 wheel。
- 恢复依赖原 `validated-shim-wheel-set` 和 runtime publication 元数据；在 Actions 保留期内完成。
  过期或缺失时恢复原始产物，不能重新编译再冒充原来源。完整 provenance 会保存在 GitHub manifest 中。
- draft 上传中断时核对已有附件并补齐缺失项，不使用 `--clobber`。已公开 Release 只接受
  相同元数据/字节的幂等核对；变更内容使用新版本。预发布版本不会替换 latest。
- OIDC invalid-publisher：核对上述两个原 YAML 的项目绑定及 environment，不改绑到新流程。
- 缺子模块：在固定 commit 运行 `git submodule update --init --recursive`，不省略源码或打包 LFS pointer。

用户安装仍为 `python -m pip install taichi-forge==0.7.0`，随后 `python -m pip check`，
通过 `import taichi_forge as ti; print(ti.__version__)` 核对版本。

## 规范依据

[PyPA Actions 发布指南](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
用于构建/发布分离和 Trusted Publishing；[PyPI 证明](https://docs.pypi.org/attestations/)
关联上传身份与文件摘要；[GitHub Release 管理](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
说明先准备 draft 附件再公开。
