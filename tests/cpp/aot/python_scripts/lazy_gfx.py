"""A usable Graph beside an intentionally missing unused shader payload."""
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import taichi_forge as ti

ti.init(arch=ti.vulkan, offline_cache=False)


@ti.kernel
def run(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in out:
        out[i] = i + 7


@ti.kernel
def unused(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in out:
        out[i] = i * 5


module = ti.aot.Module()
module.add_kernel(run)
arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
for name, kernel in [("good", run), ("bad", unused)]:
    builder = ti.graph.GraphBuilder()
    builder.dispatch(kernel, arg)
    module.add_graph(name, builder.compile())
path = Path(os.environ["TAICHI_AOT_FOLDER_PATH"])
module.save(path)
metadata = json.loads((path / "metadata.json").read_text())
for kernel in metadata["kernels"]:
    if kernel["name"].startswith("unused"):
        for task in kernel["tasks_attribs"]:
            (path / (task["name"] + ".spv")).unlink()
with ZipFile(path / "module.tcm", "w", compression=ZIP_DEFLATED) as archive:
    for file in path.iterdir():
        if file.is_file() and file.suffix != ".tcm":
            archive.write(file, file.name)
