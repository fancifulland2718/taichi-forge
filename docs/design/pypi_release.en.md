# Standard GitHub Release and PyPI publication

[中文](pypi_release.md) · [Build wheels](../forge/build_wheels.en.md)

This guide uses **0.7.0** as an example without claiming it is published.
Keep the already-bound PyPI publishers: [publish_runtime_pypi.yml](../../.github/workflows/publish_runtime_pypi.yml)
for runtime and [publish_pypi.yml](../../.github/workflows/publish_pypi.yml) for the main package.
The new [release.yml](../../.github/workflows/release.yml) **only handles GitHub Release**,
automatically following successful production PyPI publication of the main package.
It neither builds wheels nor uploads to PyPI. Cross-service publication is not atomic:
verify the index files first, then publish identical wheel bytes and public assets on GitHub.

## Deliverables

| Location | Files and purpose |
| --- | --- |
| PyPI and GitHub | 2 runtime platform wheels, 10 CPython 3.10–3.14 shim wheels; Windows x86_64 and manylinux_2_35 x86_64 |
| GitHub | `taichi-forge-0.7.0-source.tar.gz`: pinned sources, recursive submodules, licenses, build scripts, public tests/API examples, `SOURCE_COMMIT`, `SOURCE_MANIFEST.json` |
| GitHub | `taichi-forge-0.7.0-docs.zip`: public documentation, English/Chinese READMEs, license |
| GitHub | `RELEASE_NOTES.md`: Chinese and English together, also the release body |
| GitHub | `release-manifest.json`: version, commit, originating run, file sizes/SHA256, submodule sources, actual public LLVM URLs and hashes |
| GitHub | `SHA256SUMS`: every other asset, including the manifest |

Sources come from committed Git objects and pinned submodules, excluding untracked/ignored
files, root `internal/`, `opt_doc/` and temporary directories. Required vendor-internal
headers and licenses remain included. GitHub's automatic source ZIP lacks submodules.
PyPI receives wheels only; the full-source archive follows the build guide, not a
pip-installable sdist that configures native SDKs. Users install compilers, SDKs and
external libraries; proprietary SDKs/runtimes are not repackaged. LLVM download and
source build recipes are public. Build the standalone C/C++ AOT SDK from the same sources
using the [SDK guide](../forge/aot_sdk.en.md); binaries without separate installation
qualification are not advertised as prebuilt release assets.

## Keep existing publisher bindings

| PyPI / TestPyPI project | Existing Trusted Publisher workflow |
| --- | --- |
| `taichi-forge-runtime` | `publish_runtime_pypi.yml` |
| `taichi-forge` | `publish_pypi.yml` |

**No rebinding is required; do not bind either project to `release.yml`.** Keep owner
`fancifulland2718`, repository `taichi-forge`, and the corresponding `pypi` / `testpypi`
environment. The original upload jobs retain `id-token: write` without long-lived PyPI
tokens. The GitHub Release job only needs `contents: write` and Actions artifact read
access; it needs neither a PyPI environment nor an OIDC upload identity.

Keep `LLVM20_WIN_URL`, `LLVM20_WIN_SHA256`, `LLVM20_LINUX_URL`, and
`LLVM20_LINUX_SHA256` (`LLVM20_LINUX_MANYLINUX_URL` remains an alias). Use public HTTPS
URLs and SHA256 digests. Source recipes: [Windows](../../.github/workflows/build_llvm20_windows.yml),
[Linux](../../.github/workflows/build_llvm20_linux.yml). Build jobs record actual download hashes and configuration.

## Prepare and publish in order

1. Update `version.txt`, CMake versions, native header, runtime dependency and both
   `docs/forge/release_notes.*.md` files; commit the changes. ABI/format revisions are
   maintained separately. Packages may publish independently; explicitly select
   `runtime_version` when reusing a compatible runtime. Never rename an old wheel.
2. Build, validate and publish runtime with its existing workflow: `platform=all`,
   `publish=true`, `target=pypi`. `publish=false` only builds/validates;
   `platform=windows` is diagnostic and cannot publish. Successful upload saves the
   `runtime-publication-<version>-pypi` Actions metadata artifact (90 days), containing
   published wheel hashes, actual source commit, LLVM and build constraints. It never goes to PyPI.
3. The existing main-package workflow links, installs and validates ten CPython shims
   against the published runtime. Rehearse with `publish=false`; `wheel_run_id` can reuse
   its `validated-shim-wheel-set`, whose `shim-build-info.json` preserves actual build
   provenance. Selecting `target` alone does not enable uploading.
4. Push the matching version tag to trigger main-package publication. Runtime must already
   be available from production PyPI:

```bash
git tag -a forge-v0.7.0 -m "Taichi Forge 0.7.0"
git push origin forge-v0.7.0
```

Alternatively dispatch the original main-package workflow on the existing tag, with
`publish=true`, `target=pypi` and optional `wheel_run_id`. Reused shims' actual build commit
must match the tag; avoid duplicate submissions alongside the automatic tag run.
`runtime_source=artifact` and numeric `runtime_run_id` remain available for validation;
that runtime must be published before releasing its shim. `validation_platform=windows`
cannot publish. TestPyPI still uses the original workflows and triggers no production GitHub Release.

Successful main-package PyPI upload automatically calls the GitHub-only workflow: download
validated shims and PyPI runtime → verify all published wheel hashes, publisher jobs, actual
source identity and tag → assemble complete source, public docs, bilingual notes and hashes →
populate a draft → publish GitHub Release. PyPI staging contains only `.whl` files belonging
to the two projects. Sources, documentation, JSON and checksums are GitHub assets only.

The source archive represents the tagged main-package source. An independently published
compatible runtime may originate at a different commit/version; `release-manifest.json`
records its actual build inputs, commit and public source link separately. Both runtime
platforms must originate at the same runtime source commit.

Final wheels still require binary audit, `pip check`, outside-checkout imports and Python/NumPy
ABI checks. Select representative CPU/CUDA/Vulkan checks for the changes and report untested
scope honestly. CI build/smoke is not full GPU qualification; follow the
[Linux checklist](../forge/linux_revalidation.en.md) where relevant.

## Prefer validated artifact reuse

Both existing workflows default to `reuse=true`. Full-matrix runs first read small source
records, selecting complete, unexpired candidates in this repository with the **same source
commit, package version, and exact runtime dependency for shims**. A successful validation
job is required. Selected wheels are downloaded and checked again, skipping native or
CPython compilation. Tag-triggered shim publication also discovers same-commit dry runs;
a later upload/Release failure does not invalidate successful validation.

Both workflows accept explicit `wheel_run_id` values for their own validated wheel sets.
Runtime can therefore build the full matrix with `publish=false`, then publish those exact
wheels using that ID and `publish=true`. Records retain the original build commit/run,
LLVM and configuration; publication does not become build provenance. Changed versions,
corrupt or missing files fail validation. Automatic lookup with no eligible candidate
builds normally. Candidate metadata and wheel sets are retained for 90 days.

To use changed toolchain repository variables or force qualification, set `reuse=false`
and leave `wheel_run_id` blank. Automatic matching retains the artifact's recorded toolchain;
it does not reinterpret the build using current repository variables. Platform-scoped
validation keeps its original build path and cannot publish a partial matrix. GitHub
Release always consumes published artifacts and never compiles them.

## Recovery and attachment rehearsal

- Original PyPI upload failed: **Re-run failed jobs** in that run, retaining its wheels.
  Matching filename/SHA256 files stay; only missing files upload. Different bytes or
  yanked files fail. No blind skip-existing or recompilation.
- PyPI succeeded but GitHub failed: retry failed jobs first. Alternatively dispatch
  **GitHub Release / GitHub 发行**, specifying `version`, `runtime_version` and the
  main-package upload's `shim_run_id`. `publish=false` only assembles the attachment bundle
  (both packages must already be on production PyPI); `publish=true` completes GitHub
  publication. Neither mode uploads to PyPI or rebuilds wheels.
- Recovery needs the original `validated-shim-wheel-set` and runtime publication metadata;
  finish within Actions retention. Restore original artifacts if missing/expired rather
  than rebuilding and inventing provenance. The GitHub manifest retains full provenance.
- Interrupted draft upload checks existing assets and adds missing ones without `--clobber`.
  Published releases only accept identical metadata/bytes. Changes need a new version;
  prereleases do not replace latest.
- OIDC invalid-publisher: check the two original workflow bindings and environments above;
  do not rebind to the new workflow.
- Missing submodule: `git submodule update --init --recursive` at the pinned commit;
  never omit required sources or archive unresolved LFS pointers.

Install using `python -m pip install taichi-forge==0.7.0`, then `python -m pip check`;
verify with `import taichi_forge as ti; print(ti.__version__)`.

## Standards

[PyPA Actions guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
for build/publication separation and Trusted Publishing; [PyPI attestations](https://docs.pypi.org/attestations/)
for publisher/file identity; [GitHub release management](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
for preparing draft assets before publication.
