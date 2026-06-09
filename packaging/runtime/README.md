# Taichi Forge Runtime

This distribution contains platform-native Taichi Forge runtime libraries and
runtime bitcode resources. It is installed as a dependency of the Python-facing
`taichi-forge` distribution.

It intentionally does not expose the public `taichi_forge` Python API. The
Python package imports `taichi_forge_runtime` only to locate native resources.

The Windows wheel also includes the `taichi_runtime.lib` import library so the
CPython shim wheels can link against the already-published runtime package
during release builds.
