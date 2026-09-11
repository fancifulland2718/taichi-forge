"""Fixed RasterPass draw preparation; leaves ordinary GGUI staging unchanged."""

import inspect

import numpy as np
import taichi_forge as ti
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.ui.scene import SceneV2
from taichi_forge.ui.staging_buffer import (
    copy_all_to_vbo,
    copy_all_to_vbo_particle,
    copy_to_vbo_scalar,
    copy_to_vbo_vector,
)
from taichi_forge.ui.utils import get_field_info


@ti.kernel
def _copy_indices(source: ti.template(), target: ti.types.ndarray(ndim=1)):
    for i in range(target.shape[0]):
        target[i] = source[i]


@ti.kernel
def _copy_transforms(source: ti.template(), target: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for i in range(source.shape[0]):
        for row, col in ti.static(ti.ndrange(4, 4)):
            target[i, row, col] = source[i][row, col]


@ti.kernel
def _pack_generated_normals(
    vbo: ti.types.ndarray(dtype=ti.types.vector(12, ti.f32), ndim=1),
    vertices: ti.template(),
    normals: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    color: ti.template(),
):
    for i in range(vertices.shape[0]):
        vbo[i][0:3] = vertices[i]
        vbo[i][3:6] = normals[i]
        if ti.static(color != 0):
            if ti.static(color.n == 3):
                vbo[i][8:11] = color[i]
                vbo[i][11] = 1.0
            else:
                vbo[i][8:12] = color[i]


@ti.kernel
def _flat_normals(vertices: ti.template(), normals: ti.types.ndarray(dtype=ti.math.vec3, ndim=1)):
    for triangle in range(vertices.shape[0] // 3):
        i = triangle * 3
        n = (vertices[i] - vertices[i + 1]).cross(vertices[i] - vertices[i + 2]).normalized()
        for j in ti.static(range(3)):
            normals[i + j] = n


@ti.kernel
def _indexed_normals(
    vertices: ti.template(),
    indices: ti.types.ndarray(ndim=1),
    normals: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    weights: ti.types.ndarray(dtype=ti.f32, ndim=1),
    triangles: ti.template(),
):
    for i in range(normals.shape[0]):
        normals[i] = ti.Vector([0.0, 0.0, 0.0])
        weights[i] = 0.0
    for triangle in range(triangles):
        a, b, c = indices[triangle * 3], indices[triangle * 3 + 1], indices[triangle * 3 + 2]
        n = (vertices[a] - vertices[b]).cross(vertices[a] - vertices[c]).normalized()
        for i in ti.static(range(3)):
            j = indices[triangle * 3 + i]
            ti.atomic_add(normals[j], n)
            ti.atomic_add(weights[j], 1.0)
    for i in range(normals.shape[0]):
        if weights[i] > 0:
            normals[i] = normals[i] / weights[i]


class PreparedRasterDraw:
    """Program-owned device staging with fixed sources and cold native bindings."""

    def __init__(self, scene, kind, arguments):
        bound = inspect.signature(getattr(SceneV2, kind)).bind(None, **arguments)
        bound.apply_defaults()
        args = dict(bound.arguments)
        args.pop("self")
        self._updates = []
        self._owned_arrays = []
        self._sources = tuple(arguments.values())
        self.has_host_upload = any(isinstance(value, np.ndarray) for value in arguments.values())
        vertices = args.get("vertices", args.get("centers"))
        count = vertices.shape[0]
        source_dims = len(vertices.shape) - (1 if isinstance(vertices, np.ndarray) else 0)
        if count <= 0 or source_dims != 1:
            raise ValueError("RasterPass requires nonempty one-dimensional vertex/center storage")
        vbo = ti.Vector.ndarray(12, ti.f32, count)
        self._owned_arrays.append(vbo)
        color = args.get("per_vertex_color")
        has_color = color is not None

        indices = self._buffer(args.get("indices"), transforms=False)
        normals = args.get("normals")
        generated = kind in ("mesh", "mesh_instance") and normals is None
        if generated:
            if not isinstance(vertices, ti.Field):
                raise TypeError("Automatic RasterPass normals require a vector field")
            normals = ti.Vector.ndarray(3, ti.f32, count)
            self._owned_arrays.append(normals)
            if indices is None:
                if count % 3:
                    raise ValueError("Non-indexed mesh vertex count must be divisible by three")
                self._kernel(_flat_normals, vertices, normals)
            else:
                weights = ti.ndarray(ti.f32, count)
                self._owned_arrays.append(weights)
                self._kernel(_indexed_normals, vertices, indices, normals, weights, indices.shape[0] // 3)

        if isinstance(vertices, ti.Field):
            if generated:
                self._kernel(_pack_generated_normals, vbo, vertices, normals, color if has_color else 0)
            elif kind == "particles":
                radius = args.get("per_vertex_radius")
                self._kernel(
                    copy_all_to_vbo_particle,
                    vbo,
                    vertices,
                    radius if radius is not None else 0,
                    color if has_color else 0,
                )
            else:
                self._kernel(
                    copy_all_to_vbo, vbo, vertices, normals if normals is not None else 0, 0, color if has_color else 0
                )
        else:
            self._kernel(copy_to_vbo_vector, vbo, vertices, 0, 3, ti.Vector([0.0, 0.0, 0.0]))
            if normals is not None:
                self._kernel(copy_to_vbo_vector, vbo, normals, 3, 3, ti.Vector([0.0, 0.0, 0.0]))
            if has_color:
                self._kernel(copy_to_vbo_vector, vbo, color, 8, 4, ti.Vector([1.0, 1.0, 1.0, 1.0]))
            if kind == "particles" and args.get("per_vertex_radius") is not None:
                self._kernel(copy_to_vbo_scalar, vbo, args["per_vertex_radius"], 3)

        vbo_info, index_info = get_field_info(vbo), get_field_info(indices)
        self._native = getattr(scene.scene, kind)
        if kind == "particles":
            n = args["index_count"] if args["index_count"] is not None else count
            self._arguments = (
                vbo_info,
                has_color,
                args["per_vertex_radius"] is not None,
                args["color"],
                args["radius"],
                n,
                args["index_offset"],
            )
        else:
            n = args["vertex_count"] if args["vertex_count"] is not None else count
            ni = (
                args["index_count"] if args["index_count"] is not None else (n if indices is None else indices.shape[0])
            )
            tail = (ni, args["index_offset"], n, args["vertex_offset"])
            if kind == "lines":
                n = min(n, count - args["vertex_offset"])
                n -= n % 2
                self._arguments = (
                    vbo_info,
                    index_info,
                    has_color,
                    args["color"],
                    args["width"],
                    ni,
                    args["index_offset"],
                    n,
                    args["vertex_offset"],
                )
            elif kind == "mesh":
                self._arguments = (
                    vbo_info,
                    has_color,
                    index_info,
                    args["color"],
                    args["two_sided"],
                    *tail,
                    args["show_wireframe"],
                )
            else:
                transforms = self._buffer(args["transforms"], transforms=True)
                instances = (
                    1
                    if transforms is None
                    else (args["instance_count"] if args["instance_count"] is not None else transforms.shape[0])
                )
                self._arguments = (
                    vbo_info,
                    has_color,
                    index_info,
                    args["color"],
                    args["two_sided"],
                    get_field_info(transforms),
                    instances,
                    args["instance_offset"],
                    *tail,
                    args["show_wireframe"],
                )
        self.requested_bytes = sum(array._get_nelement() * array._get_element_size() for array in self._owned_arrays)

    def _kernel(self, kernel, *args):
        # Compilation and allocation must finish before Graph.submit's batch.
        key = kernel._primal.ensure_compiled(*args)
        program = ti.lang.impl.get_runtime().prog
        program.compile_kernel(program.config(), program.get_device_caps(), kernel._primal.compiled_kernels[key])
        self._updates.append((kernel, args))

    def _buffer(self, source, *, transforms):
        if source is None:
            return None
        dtype = source.dtype
        if transforms:
            scalar_dtype = np.dtype("float32") if isinstance(source, np.ndarray) else ti.f32
            shape = source.shape + getattr(source, "element_shape", ())
            field_matrix = isinstance(source, ti.Field) and source.n == 4 and source.m == 4
            if dtype != scalar_dtype or (not field_matrix and tuple(shape[1:]) != (4, 4)):
                raise ValueError("RasterPass transforms require f32 4x4 matrices")
        else:
            allowed = (np.dtype("int32"), np.dtype("uint32")) if isinstance(source, np.ndarray) else (ti.i32, ti.u32)
            if dtype not in allowed or len(source.shape) != 1 or getattr(source, "element_shape", ()):
                raise ValueError("RasterPass indices require scalar i32/u32 one-dimensional storage")
        if isinstance(source, Ndarray):
            return source
        if transforms:
            shape = (source.shape[0], 4, 4)
            if isinstance(source, ti.Field) and (source.n != 4 or source.m != 4):
                raise ValueError("RasterPass transforms must be 4x4 matrices")
            target = ti.ndarray(ti.f32, shape)
        else:
            dtype = ti.u32 if source.dtype in (ti.u32, np.dtype("uint32")) else ti.i32
            target = ti.ndarray(dtype, source.shape)
        self._owned_arrays.append(target)
        if isinstance(source, np.ndarray):
            # Host-owned input is explicitly uploaded; there is no device
            # readback, including for dynamic host index/transform contents.
            self._updates.append((target.from_numpy, (source,)))
        else:
            self._kernel(_copy_transforms if transforms else _copy_indices, source, target)
        return target

    def run(self):
        for update, args in self._updates:
            update(*args)
        self._native(*self._arguments)
