"""Frozen draw parameters, independent of GGUI handles and buffer owners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RasterDrawDefinition:
    kind: str
    color: tuple
    has_color: bool
    index_count: int
    index_offset: int
    vertex_count: int = 0
    vertex_offset: int = 0
    instance_count: int = 1
    instance_offset: int = 0
    has_radius: bool = False
    radius: float = 0
    width: float = 0
    two_sided: bool = False
    wireframe: bool = False

    @classmethod
    def resolve(cls, kind, args, capacity, index_capacity, transform_capacity):
        n = args.get("vertex_count")
        n = capacity if n is None else n
        count = args["index_count"]
        count = (n if index_capacity is None else index_capacity) if count is None else count
        common = dict(
            kind=kind,
            color=tuple(args["color"]),
            has_color=args.get("per_vertex_color") is not None,
            index_count=count,
            index_offset=args["index_offset"],
        )
        if kind == "particles":
            return cls(**common, has_radius=args.get("per_vertex_radius") is not None, radius=args["radius"])
        if kind == "lines":
            n = min(n, capacity - args["vertex_offset"])
            n -= n % 2
            return cls(**common, vertex_count=n, vertex_offset=args["vertex_offset"], width=args["width"])
        instances = 1
        if transform_capacity is not None:
            instances = args["instance_count"] if args["instance_count"] is not None else transform_capacity
        return cls(
            **common,
            vertex_count=n,
            vertex_offset=args["vertex_offset"],
            instance_count=instances,
            instance_offset=args.get("instance_offset", 0),
            two_sided=args["two_sided"],
            wireframe=args["show_wireframe"],
        )

    def native_arguments(self, vertices, indices, transforms):
        if self.kind == "particles":
            return (
                vertices,
                self.has_color,
                self.has_radius,
                self.color,
                self.radius,
                self.index_count,
                self.index_offset,
            )
        tail = (self.index_count, self.index_offset, self.vertex_count, self.vertex_offset)
        if self.kind == "lines":
            return (vertices, indices, self.has_color, self.color, self.width, *tail)
        head = (vertices, self.has_color, indices, self.color, self.two_sided)
        if self.kind == "mesh_instance":
            head += (transforms, self.instance_count, self.instance_offset)
        return (*head, *tail, self.wireframe)
