"""AOT structure layout publication over shared source-array bindings."""

from taichi_forge.graph._lowering_array_sources import SourceArrayBindings as _SourceArrayBindings
from taichi_forge.lang.exception import TaichiRuntimeError


class SourceArrayBindings(_SourceArrayBindings):
    def __init__(self, bindings, fixed_bindings, module=None):
        self.module = module
        self.pending_layouts = {}
        super().__init__(bindings, fixed_bindings)

    def structure_layout(self, dtype):
        if self.module is None:
            raise TaichiRuntimeError("Structure source export requires a module layout catalog")
        for name, published in self.module._ndarray_layouts.items():
            if published == dtype.dtype:
                return name
        for name, pending in self.pending_layouts.items():
            if pending.dtype == dtype.dtype:
                return name
        index = len(self.module._ndarray_layouts) + len(self.pending_layouts)
        name = f"__forge_aot_struct_{index}"
        while name in self.module._ndarray_layouts or name in self.pending_layouts:
            index += 1
            name = f"__forge_aot_struct_{index}"
        self.pending_layouts[name] = dtype
        return name

    def publish(self):
        # Publication follows whole-Graph preflight, like retained Field views.
        for name, dtype in self.pending_layouts.items():
            self.module.add_ndarray_layout(name, dtype)
