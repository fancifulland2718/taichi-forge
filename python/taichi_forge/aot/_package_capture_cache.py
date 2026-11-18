"""Portable, owner-free dependency facts for the CPU package object cache."""

import ast
import inspect
import json
import sys
from types import ModuleType

from taichi_forge.lang._build_cache import value_digest
from taichi_forge.lang.matrix import MatrixType
from taichi_forge.lang.struct import StructType
from taichi_forge.types._argument_descriptor import describe_annotation
from taichi_forge.types.annotations import Template

from ._component import _ComponentCode, _capture_id, _code_definition, _function, _static_value


class Unavailable(ValueError):
    """This input cannot be reconstructed across producer processes."""


def _tuples(value):
    return tuple(map(_tuples, value)) if isinstance(value, list) else value


def _callable(module, name):
    # The ordinary frontend imports its matrix intrinsics at the first matrix
    # expression. A recovered first entry never executes that expression.
    if module == "taichi_forge.lang.matrix_ops":
        from taichi_forge.lang import matrix_ops as obj
    else:
        obj = sys.modules[module]
    for part in name.split("."):
        obj = inspect.getattr_static(obj, part)
    return obj


def _body(obj):
    return obj._primal.func if getattr(obj, "_is_wrapped_kernel", False) else _function(obj)


def _annotation(value):
    from taichi_forge._lib import core
    from taichi_forge.types.ndarray_type import NdarrayType

    if value is inspect.Parameter.empty:
        return "inferred"
    if value is None or isinstance(value, Template):
        return None if value is None else "template"
    if isinstance(value, core.DataType):
        return str(value)
    if value is int or value is float or value is bool:
        return value.__name__
    if isinstance(value, (StructType, MatrixType, NdarrayType)):
        return (describe_annotation(value).structural_key(), str(getattr(value, "layout", None)),
                str(getattr(value, "boundary", None)))
    raise Unavailable("annotation")


def _binding(value):
    from taichi_forge._lib import core

    if isinstance(value, ModuleType):
        if value is sys.modules["taichi_forge"] or value.__name__.startswith("taichi_forge."):
            return ("module", value.__name__)
        # The legacy shim can share the same frontend object.
        forge = sys.modules["taichi_forge"]
        if (value.__name__ == "taichi" and vars(value).get("__file__") == forge.__file__
                and vars(value).get("lang") is forge.lang):
            return ("module", value.__name__)
        raise Unavailable("external module")
    if isinstance(value, (MatrixType, StructType, core.DataType)):
        return ("type", _annotation(value))
    if inspect.isfunction(value) or inspect.isbuiltin(value) or inspect.isclass(value):
        fn = _body(value)
        if getattr(value, "_is_taichi_function", False):
            if _callable(fn.__module__, fn.__qualname__) is not value:
                raise Unavailable("local helper")
        elif not (fn.__module__ == "builtins" or fn.__module__.startswith("taichi_forge.")):
            raise Unavailable("host callable")
        return ("callable", fn.__module__, fn.__qualname__)
    return ("value", _static_value(value))


def _syntax(definition, memo):
    # Cache only immutable source/code analysis for this export. Live globals,
    # signatures and bound method implementations are checked for each use.
    from taichi_forge.lang.kernel_impl import _SOURCE_TEMPLATE_CACHE

    key = definition, definition.function.__code__
    if key not in memo:
        tree, _ = definition.acquire(cache=_SOURCE_TEMPLATE_CACHE and definition.cached_template is not None)
        node = tree.body[0]
        names = {use.id for stmt in node.body for use in ast.walk(stmt)
                 if isinstance(use, ast.Name) and isinstance(use.ctx, ast.Load)}
        memo[key] = (value_digest((ast.dump(node), _code_definition(key[1]))),
                     sorted(names.difference(key[1].co_varnames)))
    return memo[key]


def _source_facts(kernel, code, memo, component, owner):
    refs = {(kernel.func.__module__, kernel.func.__qualname__)}
    refs.update(definition[:2] for definition in (*code.functions.values(), *code.methods.values()))
    for mapping in (code.functions, code.constants, code.struct_types):
        for fn, _ in mapping:
            if _body(_callable(fn.__module__, fn.__qualname__)) is not fn:
                raise Unavailable("noncanonical lexical owner")
            refs.add((fn.__module__, fn.__qualname__))
    for name, (module, qualified, _) in code.methods.items():
        path = code.method_paths[name]
        receiver = component.read(owner, _capture_id(path[:-1])) if len(path) > 1 else owner
        actual = inspect.getattr_static(receiver, path[-1])
        if actual is not _callable(module, qualified):
            raise Unavailable("noncanonical receiver method")
    result = []
    for module, qualified in sorted(refs):
        obj = _callable(module, qualified)
        fn = _body(obj)
        if (fn.__module__, fn.__qualname__) != (module, qualified):
            raise Unavailable("rebound definition")
        if module.startswith("taichi_forge."):
            continue  # Covered by the producer toolchain fingerprint.
        if fn.__closure__ or fn.__globals__ is not vars(sys.modules[module]):
            raise Unavailable("non-module lexical environment")
        wrapper = getattr(obj, "_primal", None) or getattr(obj, "func", None)
        if wrapper is None or wrapper._definition.compiled is not None:
            raise Unavailable("untracked callable")
        syntax, names = _syntax(wrapper._definition, memo)
        bindings = [(name, _binding(fn.__globals__[name] if name in fn.__globals__ else fn.__builtins__[name]))
                    for name in names if name in fn.__globals__ or name in fn.__builtins__]
        args = [_annotation(arg.annotation) for arg in wrapper.arguments]
        returns = [_annotation(ann) for ann in wrapper.return_type or ()]
        result.append((module, qualified, syntax, bindings, args, returns,
                       getattr(wrapper, "opt_level", None), getattr(wrapper, "is_real_function", None),
                       getattr(wrapper, "pyfunc", None), _static_value(fn.__defaults__),
                       [(k, _static_value(v)) for k, v in (fn.__kwdefaults__ or {}).items()]))
    return value_digest(result)


# These maps contain strings, numbers and tuple-valued capture facts. Preserve
# insertion order: an outer presence guard must precede a conditional member.
_MAPS = ("used", "capture_paths", "capture_source_paths", "static", "objects", "resource_conditions",
         "record_field_keys", "owner_sequence_lengths", "methods", "method_paths", "groups", "member_paths",
         "resource_templates")
_LEXICAL_MAPS = ("functions", "constants", "struct_types")


def freeze(code, memo, component, owner):
    if (code.fixed_groups or code.workspaces or code.lexical or code.model_static_ranges
            or code.contract.get("model_blocks")):
        raise Unavailable("task-local or lexical binding")
    maps = {name: list(getattr(code, name).items()) for name in _MAPS}
    for name in _LEXICAL_MAPS:
        maps[name] = [(fn.__module__, fn.__qualname__, key, val)
                      for (fn, key), val in getattr(code, name).items()]
    return dict(maps=maps, sources=_source_facts(code.kernel, code, memo, component, owner),
                signature=code.call_signature, dynamic=code.dynamic, entry=code.entry_definition,
                equalities=sorted(code.record_equalities), templates=sorted(code.static_templates))


def thaw(facts, kernel, key, entry):
    code = _ComponentCode.__new__(_ComponentCode)
    code.kernel, code.runtime, code.key = kernel, kernel.runtime, key
    for name in _MAPS:
        setattr(code, name, {_tuples(k): _tuples(v) for k, v in facts["maps"][name]})
    for name in _LEXICAL_MAPS:
        setattr(code, name, {(_body(_callable(module, qualified)), k): _tuples(v)
                             for module, qualified, k, v in facts["maps"][name]})
    code.call_signature = _tuples(facts["signature"])
    code.entry_definition = _tuples(facts["entry"])
    code.dynamic = tuple(facts["dynamic"])
    code.record_equalities = frozenset(map(_tuples, facts["equalities"]))
    code.static_templates = frozenset(map(_tuples, facts["templates"]))
    code.model_static_ranges = ()
    for name in ("_function_states", "fixed_groups", "fixed_dependencies", "fixed_members", "lexical", "workspaces"):
        setattr(code, name, {})
    code.semantic_key = entry["semantic_key"]
    code._contract = json.dumps(entry["contract"])
    return code


def matches(code, facts, memo, component, owner, signature, resources, layouts):
    from ._component import _extent_invariant_compact_member, _used_template_signature

    if (code.call_signature != _used_template_signature(signature, code.static_templates)
            or any(resources.get(slot) != name for slot, name in code.resource_templates.items())
            or any(component._paths.get(name) != path for name, path in code.capture_paths.items())
            or not layouts.matches_static(code) or not code.matches(component, owner, layouts=layouts)):
        return False
    if facts["sources"] != _source_facts(code.kernel, code, memo, component, owner):
        return False
    contract = code.contract
    if contract["declaration_sources"] != {
        name: component._sources[name] for name in code.used if name in component._sources
    }:
        return False
    for name, access in contract.get("dense_access", ()):
        field = component.read(owner, name)
        declaration = component.declarations[name]
        schema = json.loads(code.groups[declaration.group])
        compact = schema["version"] == 1 and _extent_invariant_compact_member(field, declaration.validate(field))
        if compact != (access == "compact"):
            return False
    return True
