"""Frozen SPD semantics and code; live image/plan owners stay with the caller."""

from dataclasses import dataclass, field
import hashlib
import struct

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native_definition import NativeDefinition, NativeOperationBinding, NativeResourceReference
from taichi_forge.hardware._shader_artifact import SpirvShader


@dataclass(frozen=True)
class SpdDefinition:
    reduction: str
    source_layout: tuple
    output_layout: tuple
    shader: SpirvShader
    native: NativeDefinition = field(init=False)

    @classmethod
    def from_images(cls, source, output, reduction, words):
        def layout(image):
            fmt, shape, levels, _ = image._creation_metadata
            return fmt.name, tuple(shape), levels

        shader = SpirvShader(struct.pack(f"<{len(words)}I", *words), "compute")
        return cls(reduction, layout(source), layout(output), shader)

    def __post_init__(self):
        digest = hashlib.sha256()

        def text(value):
            encoded = value.encode()
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)

        # Preserve the native expected-plan certificate byte for byte.
        text("forge.spd.plan.v1")
        text(self.reduction)
        for fmt, shape, levels in (self.source_layout, self.output_layout):
            text(fmt)
            digest.update(struct.pack("<4Q", *shape, 1, levels))
        text(self.shader.code_sha256)
        object.__setattr__(
            self,
            "native",
            NativeDefinition("spd", "plan", dict(reduction=self.reduction, expected_plan=digest.hexdigest())),
        )

    @property
    def shader_words(self):
        return struct.unpack(f"<{len(self.shader.code) // 4}I", self.shader.code)

    def image_references(self, names, symbols=None):
        return tuple(
            (port, NativeResourceReference(symbol, ResourceEffect(name, access, subresource=("image", "whole"))))
            for port, name, symbol, access in zip(
                ("source", "output"),
                names,
                names if symbols is None else symbols,
                (GraphAccess.READ, GraphAccess.WRITE),
            )
        )

    def effects(self, names):
        return tuple(reference.effect for _, reference in self.image_references(names))

    def bind(self, plan, names, symbols):
        return NativeOperationBinding(
            NativeDefinition("spd", "run", dict(expected_plan=self.native.parameters["expected_plan"])),
            (),
            (("plan", plan), *self.image_references(names, symbols)),
        )
