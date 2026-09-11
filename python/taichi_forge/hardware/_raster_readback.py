"""Explicit RasterPass readback; never imported or executed by device replay."""

import taichi_forge as ti


@ti.kernel
def _read_color(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for x, y in ti.ndrange(output.shape[0], output.shape[1]):
        value = image.fetch(ti.Vector([x, output.shape[1] - 1 - y]), 0)
        for channel in ti.static(range(4)):
            output[x, y, channel] = value[channel]


@ti.kernel
def _read_depth(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for x, y in output:
        output[x, y] = image.fetch(ti.Vector([x, output.shape[1] - 1 - y]), 0).x


def color_numpy(image):
    output = ti.ndarray(ti.f32, (*image.shape, 4))
    _read_color(image, output)
    return output.to_numpy()


def depth_numpy(image):
    output = ti.ndarray(ti.f32, image.shape)
    _read_depth(image, output)
    return output.to_numpy()
