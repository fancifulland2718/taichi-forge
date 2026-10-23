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


@ti.kernel
def initialize(header: ti.types.ndarray(ti.u64, ndim=1)):
    for i in range(3):
        header[i] = ti.u64(7)


module.add_kernel(initialize)


@ti.kernel
def publish_from_seed(epoch: ti.u64, step: ti.u64, timestamp: ti.u64,
                      seed: ti.types.ndarray(ti.u64, ndim=1),
                      header: ti.types.ndarray(ti.u64, ndim=1)):
    header[0] = epoch + seed[0]
    header[1] = step + seed[1]
    header[2] = timestamp + seed[2]
    for i in range(3):
        seed[i] += ti.u64(1)


module.add_kernel(publish_from_seed)
module.add_kernel(publish)
module.save(os.environ["TAICHI_VULKAN_PARAMETERS_AOT_DIR"])
