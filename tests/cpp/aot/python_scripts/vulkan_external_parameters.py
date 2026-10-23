"""Four-element external ndarray fixture for native Vulkan owner qualification."""
import os

import taichi_forge as ti

ti.init(arch=ti.vulkan, offline_cache=False)


@ti.kernel
def publish_external(value: ti.u64, out: ti.types.ndarray(ti.u32, ndim=1)):
    out[0] = ti.cast(value, ti.u32)
    out[1] = ti.cast(value >> 32, ti.u32)
    out[2] += ti.u32(3)
    out[3] = out[3] ^ ti.u32(0xA5C39E71)


module = ti.aot.Module(caps=["spirv_has_int64"])
module.add_kernel(publish_external)
module.save(os.environ["TAICHI_VULKAN_EXTERNAL_PARAMETERS_AOT_DIR"])
