"""Producer-proven lexical bindings using the ordinary Component capture ABI.

Bindings contain declaration paths or immutable static facts, never instances.
The package definition pass supplies them; normal JIT does not consult them.
"""

from dataclasses import dataclass

from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge_runtime._static_values import read_static_key, static_key


@dataclass(frozen=True)
class LexicalSequence:
    function: object
    name: str
    indices: tuple
    length: int

    def __len__(self):
        return self.length


def binding_at(component, function, name, indices=()):
    binding = component._lexical_bindings.get((function, name))
    for index in indices:
        if binding is None or binding[0] != "sequence" or index >= len(binding[1]):
            return None
        binding = binding[1][index]
    return binding


def fact(binding):
    if type(binding) is not tuple or len(binding) != 2:
        raise TaichiCompilationError("Lexical capture requires a proven declaration binding")
    kind, value = binding
    if kind == "path" and type(value) is tuple and value and all(type(item) is str for item in value):
        return binding
    if kind == "static":
        return kind, static_key(read_static_key(value))
    if kind == "sequence" and type(value) is tuple:
        return kind, len(value)
    raise TaichiCompilationError("Unsupported lexical declaration binding")


def resolve(capture, ctx, function, name, indices=()):
    binding = binding_at(capture.component, function, name, indices)
    identity = (function, name, indices)
    value = fact(binding)
    if capture.lexical.setdefault(identity, value) != value:
        raise TaichiCompilationError("Lexical declaration changed during compilation; retry")
    kind, payload = value
    if kind == "static":
        return read_static_key(payload)
    if kind == "sequence":
        return LexicalSequence(function, name, indices, payload)
    # Record presence and resource accesses use the same lazy member lowering
    # as self.record.field. Unread sequence members never enter the ABI/key.
    for depth in range(1, len(payload)):
        capture.attribute(ctx, payload[depth - 1], prefix=payload[: depth - 1])
    return capture.attribute(ctx, payload[-1], prefix=payload[:-1])


def subscript(capture, ctx, sequence, indices):
    if len(indices) != 1 or type(indices[0]) is not int:
        raise TaichiCompilationError("Lexical resource sequences require a compile-time integer index")
    index = indices[0]
    if not -sequence.length <= index < sequence.length:
        raise TaichiCompilationError("Lexical resource sequence index is out of range")
    return resolve(capture, ctx, sequence.function, sequence.name, (*sequence.indices, index % sequence.length))


def mismatch(code, component):
    for (function, name, indices), expected in code.lexical.items():
        binding = binding_at(component, function, name, indices)
        if binding is None or fact(binding) != expected:
            return f"lexical declaration {name!r} in {function.__qualname__}"
    return None


def contract(capture):
    return sorted(
        (function.__module__, function.__qualname__, name, indices, value)
        for (function, name, indices), value in capture.lexical.items()
    )
