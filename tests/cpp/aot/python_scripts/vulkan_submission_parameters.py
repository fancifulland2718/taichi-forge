"""Small real compiled ABI fixture for independent Vulkan parameter uploads."""
import os
import taichi_forge as ti

ti.init(arch=ti.vulkan, offline_cache=False)


@ti.kernel
def publish(epoch: ti.u64, step: ti.u64, timestamp: ti.u64,
            header: ti.types.ndarray(ti.u64, ndim=1)):
    header[0] = epoch
    header[1] = step
    header[2] = timestamp


module = ti.aot.Module(caps=["spirv_has_int64"])
module.add_kernel(publish)
module.save(os.environ["TAICHI_VULKAN_PARAMETERS_AOT_DIR"])
