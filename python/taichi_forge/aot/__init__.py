"""Taichi's AOT (ahead of time) module.

Users can use Taichi as a GPU compute shader/kernel compiler by compiling their
Taichi kernels into an AOT module.
"""

import taichi_forge.aot.conventions
from taichi_forge.aot._export import export, export_as
from taichi_forge.aot.conventions.gfxruntime140 import GfxRuntime140
from taichi_forge.aot.module import Module
