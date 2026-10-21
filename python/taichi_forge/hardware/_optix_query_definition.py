"""Frozen query semantics shared by direct preparation and native lowering."""

from dataclasses import dataclass, field

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition


@dataclass(frozen=True)
class OptixQueryDefinition:
    ray_count: int
    any_hit: bool
    occlusion: bool
    instance_count: int
    face_rules: tuple | None
    alpha_masks: tuple | None
    native: NativeDefinition = field(init=False)

    def __post_init__(self):
        # Inputs have passed the public recording's type/feature qualification.
        # Keep None distinct from an all-empty mask tuple: legacy direct adapters
        # use the latter to select first-accepted/OMM traversal.
        rules = tuple(
            "table" if not isinstance(v, str) else v for v in self.face_rules or ("two_sided",) * self.instance_count
        )
        masks = tuple(None if v is None else dict(cutoff=v.cutoff, channel=v.channel) for v in self.alpha_masks or ())
        mask_count = sum(v is not None for v in masks)
        parameters = dict(
            ray_count=self.ray_count,
            occlusion=self.occlusion,
            any_hit=self.any_hit,
            face_rules=rules,
            face_table_count=rules.count("table"),
        )
        if mask_count:
            parameters.update(alpha_masks=masks, mask_count=mask_count)
        object.__setattr__(
            self, "native", NativeDefinition("optix", "query", parameters, version=3 if mask_count else 2)
        )

    @property
    def face_tables(self):
        return tuple(v for v in self.face_rules or () if not isinstance(v, str))

    @property
    def masks(self):
        return tuple(v for v in self.alpha_masks or () if v is not None)

    def arrays(self, rays, hits, hit_indices):
        arrays = [
            NativeArrayBinding("rays", rays, GraphAccess.READ),
            NativeArrayBinding("hits", hits, GraphAccess.WRITE),
        ]
        if hit_indices is not None:
            arrays.append(NativeArrayBinding("hit_indices", hit_indices, GraphAccess.WRITE))
        arrays.extend(
            NativeArrayBinding(f"faces_{i}", rule.binding, GraphAccess.READ) for i, rule in enumerate(self.face_tables)
        )
        arrays.extend(NativeArrayBinding(f"uvs_{i}", mask.uvs, GraphAccess.READ) for i, mask in enumerate(self.masks))
        return tuple(arrays)
