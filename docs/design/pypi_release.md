# GitHub Release 与 PyPI 标准发布流程

[English](pypi_release.en.md) · [构建 wheel](../forge/build_wheels.zh.md)

本文以 **0.7.0** 为例，不表示版本已经上线。正式入口是
[release.yml](../../.github/workflows/release.yml)：同一 commit 构建并验证完整矩阵，
然后发布两个 PyPI 项目，最后公开同一批附件的 GitHub Release。
跨服务没有原子事务；“同步”指同源、同版本、同字节及有序确认，不保证瞬间同时可见。

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

## 一次性配置

1. 配置 GitHub Environments `pypi`、`testpypi`，按维护策略设置审核人和 tag 限制。
2. 在 PyPI/TestPyPI 为 **taichi-forge、taichi-forge-runtime 两个项目**分别绑定 Trusted Publisher：
   owner `fancifulland2718`，repository `taichi-forge`，workflow **`release.yml`**，environment
   对应 `pypi`/`testpypi`。旧 builder 的绑定要在索引管理页更改；提交 YAML 不修改账户设置。
   不需要 PyPI 长期 token 或 `RELEASE_PAT`。
3. 配置 `LLVM20_WIN_URL`、`LLVM20_WIN_SHA256`、`LLVM20_LINUX_URL`、`LLVM20_LINUX_SHA256`。
   Linux URL 支持别名 `LLVM20_LINUX_MANYLINUX_URL`。使用公开 HTTPS URL，无登录令牌；
   SHA256 为 64 位十六进制。源码构建脚本：[Windows](../../.github/workflows/build_llvm20_windows.yml)、
   [Linux](../../.github/workflows/build_llvm20_linux.yml)。
4. 上传 job 需要 `id-token: write`；只有 GitHub draft/finalize job 需要 `contents: write`。
   构建只读，Actions 固定 SHA，保留 PyPA 默认发布证明。可启用 GitHub release immutability，
   流程先补齐 draft 附件再公开。

## 准备、预演与发布

1. 更新 `version.txt`，运行 `python scripts/sync_runtime_dependency.py`，同步两包 CMake 版本、
   native header 和 runtime 精确依赖。联合发行要求两包同版本；ABI/格式编号独立维护。
2. 更新两份 `docs/forge/release_notes.*.md` 的版本节、升级边界与支持范围，提交全部修改。
   正文由两份说明生成，链接固定 commit，不以自动提交日志替代发布说明。
3. 运行打包合同和代表后端检查，最终 wheel 经过 binary audit、`pip check`、源码目录外导入、
   Python/NumPy ABI 检查。如实记录 CPU/CUDA/Vulkan/AMDGPU 与 Linux 未测范围；CI build/smoke
   不是完整 GPU 验收。Linux 适用门槛见[复测清单](../forge/linux_revalidation.zh.md)。
4. 手动运行 **Release / 发布**，选候选分支、`publish=false`，生成 2+10 wheel 的完整
   `release-bundle` artifact（90 天），不创建 Release 或上传索引。下载并检查双语说明、
   manifest、checksum，在目标机器安装验证。内容变化需要新 commit/候选，不重命名旧 wheel。
5. 确认后推送版本 tag，自动触发生产发行：

```bash
git tag -a forge-v0.7.0 -m "Taichi Forge 0.7.0"
git push origin forge-v0.7.0
```

手动在已有 tag 上运行可设置 `publish=true`、`target=pypi`、`candidate_run_id=<成功预演 run ID>`
复用候选，要求同仓库 `release.yml`、成功完成、完全相同 commit。避免与 tag 自动运行重复
发布，不移动已发布 tag。第一次 tag 推送通常直接使用自动构建发布。仅选择 target 不启用上传。

顺序：完整验证 → 保存 bundle → GitHub draft → runtime 上传 → 确认 runtime 可获取 →
shim 上传 → 核对 PyPI 全部 wheel SHA256 → 公开 Release。预发布如 `0.7.0rc1` 标记 prerelease，
不替换 latest。手动 `target=testpypi` 只上传 TestPyPI，不创建正式 GitHub Release，仍要求匹配 tag。

仅 Windows 排查可使用 runtime builder 的 `platform=windows` 和 shim 的 `validation_platform=windows`。
shim 用 `runtime_source=artifact` 与数字 `runtime_run_id`，或 `index` 复用兼容 runtime。
两个 builder 均只构建/验证。`wheel_run_id` 是 shim 验证产物，不是完整候选编号。

## 失败恢复

- 构建失败不上传；内容变化需重新提交和验证。
- 部分 PyPI 上传：原运行 **Re-run failed jobs**，保留原 bundle，不重编。
  同名同 SHA256 跳过，缺失文件继续传，不同字节或 yanked 报错；不盲目 skip-existing，不覆盖文件。
- PyPI 成功但 finalize 失败：draft 保留，重跑校验后公开。
- draft 上传中断：校验已有附件，补齐缺失文件，不使用 `--clobber`。
- 已公开 Release 仅允许相同元数据/字节的幂等验证，变更内容使用新版本。
- OIDC invalid-publisher：核对两项目的 workflow、owner/repository、environment；fork PR 无上传权限正常。
- 缺子模块：在固定 commit 运行 `git submodule update --init --recursive`，不跳过源文件或打包 LFS pointer。

用户安装方式仍为 `python -m pip install taichi-forge==0.7.0`，随后 `python -m pip check`，
通过 `import taichi_forge as ti; print(ti.__version__)` 核对版本。

## 规范依据

[PyPA Actions 发布指南](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
用于构建/发布分离和 Trusted Publishing；[PyPI 证明](https://docs.pypi.org/attestations/)
关联上传身份与文件摘要；[GitHub Release 管理](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
说明先准备 draft 附件再公开。
