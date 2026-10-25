# Standard GitHub Release and PyPI publication

[中文](pypi_release.md) · [Build wheels](../forge/build_wheels.en.md)

This guide uses **0.7.0** as an example without claiming it is published.
[release.yml](../../.github/workflows/release.yml) builds one commit, validates the full
matrix, publishes both PyPI projects, then publishes GitHub with identical artifacts.
Cross-service publication is not atomic: synchronization means identical provenance,
versions and bytes with ordered confirmation, not simultaneous visibility.

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

## One-time configuration

1. Configure GitHub environments `pypi` and `testpypi`, with reviewers/tag policies as appropriate.
2. Configure Trusted Publishers on PyPI/TestPyPI for **both taichi-forge and taichi-forge-runtime**:
   owner `fancifulland2718`, repository `taichi-forge`, workflow **`release.yml`**, environment
   `pypi`/`testpypi`. Replace old builder bindings in index settings; YAML cannot modify them.
   No long-lived PyPI token or `RELEASE_PAT` is required.
3. Set `LLVM20_WIN_URL`, `LLVM20_WIN_SHA256`, `LLVM20_LINUX_URL`, `LLVM20_LINUX_SHA256`.
   `LLVM20_LINUX_MANYLINUX_URL` is an alias. Use public HTTPS without credentials and
   64-digit SHA256 values. Source recipes: [Windows](../../.github/workflows/build_llvm20_windows.yml),
   [Linux](../../.github/workflows/build_llvm20_linux.yml).
4. Upload jobs need `id-token: write`; only draft/finalize jobs need `contents: write`.
   Builds are read-only, Actions SHA-pinned, and PyPA's publish attestations remain enabled.
   GitHub release immutability is supported by populating draft assets before publication.

## Prepare, rehearse and publish

1. Update `version.txt`; run `python scripts/sync_runtime_dependency.py` to align CMake
   versions, native header and runtime dependency. Coordinated releases use matching
   package versions; ABI/format revisions remain independently maintained.
2. Update both `docs/forge/release_notes.*.md` version sections, upgrade boundaries and
   support. Commit everything. The release body uses both notes with commit-pinned links,
   rather than an automatically generated commit log.
3. Run packaging contracts and representative backend checks. Final wheels require binary
   audit, `pip check`, outside-checkout imports and Python/NumPy ABI tests. Record untested
   CPU/CUDA/Vulkan/AMDGPU and Linux scope honestly; build/smoke is not complete GPU qualification.
   Apply the [Linux checklist](../forge/linux_revalidation.en.md) where relevant.
4. Dispatch **Release / 发布** on the candidate branch with `publish=false` to build the
   complete 2+10 wheel `release-bundle` artifact (90 days), without publication. Download,
   inspect both languages/manifest/checksums and install on target machines. Changed contents
   require a new commit/candidate; never rename old wheels to a new version.
5. Push the matching tag to trigger production publication:

```bash
git tag -a forge-v0.7.0 -m "Taichi Forge 0.7.0"
git push origin forge-v0.7.0
```

A manual run on an existing tag can reuse `candidate_run_id=<successful dry-run ID>` with
`publish=true`, `target=pypi`. Reuse requires this repository's successful `release.yml`
run at the identical commit. Avoid duplicating the tag-triggered run or moving a published
tag. The first tag normally uses automatic build/publication. Target alone never enables upload.

Order: full validation → save bundle → GitHub draft → runtime upload → confirm runtime
availability → shim upload → verify all PyPI wheel SHA256 values → publish GitHub Release.
Versions such as `0.7.0rc1` are prereleases, not latest. Manual `target=testpypi` uploads
only to TestPyPI and creates no production GitHub Release; a matching tag is still required.

For Windows diagnosis use runtime `platform=windows`, shim `validation_platform=windows`.
The shim accepts `runtime_source=artifact` with numeric `runtime_run_id`, or `index` with a
compatible published runtime. Both builders only validate. `wheel_run_id` reuses shim
validation artifacts, not a complete release candidate.

## Recovery

- Build failure: no upload. Changed contents require a new commit and validation.
- Partial PyPI upload: **Re-run failed jobs** in the original run to keep its bundle.
  Matching filename/SHA256 files remain; missing files upload. Different bytes or yanked
  files fail; no blind skip-existing or replacement.
- PyPI complete, finalize failed: the draft remains; retry revalidates before publication.
- Interrupted draft upload: check existing assets and add missing files without `--clobber`.
- Published releases only permit identical metadata/byte verification. Changes need a new version.
- OIDC invalid-publisher: check both projects' workflow, owner/repository and environment.
  Fork PRs intentionally lack upload permissions.
- Missing submodule: `git submodule update --init --recursive` at the pinned commit;
  never omit required sources or archive unresolved LFS pointers.

Normal installation remains `python -m pip install taichi-forge==0.7.0`, then `python -m pip check`;
verify with `import taichi_forge as ti; print(ti.__version__)`.

## Standards

[PyPA Actions guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
for build/publication separation and Trusted Publishing; [PyPI attestations](https://docs.pypi.org/attestations/)
for publisher/file identity; [GitHub release management](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
for preparing draft assets before publication.
