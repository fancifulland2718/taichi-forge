"""Component capture glue for the existing Graph builder and AOT lowering.

This module collects contracts. Graph owns operation order and executable
lowering; there is no component execution interpreter.
"""

import json

from taichi_forge.aot._component import Array, DenseField, Dimension, RuntimeValue, _digest, _member
from taichi_forge.aot.utils import produce_injected_args_for_graph
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang.matrix import MatrixType
from taichi_forge.types._argument_descriptor import describe_symbolic_arg


class _GraphCapture:
    def __init__(self, component):
        self.component = component
        self.owner = None
        self.codes = []
        self.groups = {}
        self.dimensions = {}
        self.resources = {}
        self.bindings = {}
        self.values = {}
        self.explicit = {}
        self.entries = []
        self.symbols = {}
        self._members = {}

    def dispatch(self, builder, entry, args, template_args, label):
        from taichi_forge.graph._graph import Arg, ArgKind, flatten_args

        owner = getattr(entry, "_kernel_owner", None)
        kernel = getattr(entry, "_primal", None)
        if owner is None or kernel is None:
            raise TaichiCompilationError("A component Graph requires original bound kernels and no extra templates")
        expected_templates = {kernel.arguments[0].name: owner}
        if template_args is not None and (
            set(template_args) != set(expected_templates) or template_args[kernel.arguments[0].name] is not owner
        ):
            raise TaichiCompilationError("Component Graph templates must name the same declared owner")
        if self.owner is None:
            self.owner = owner
        if owner is not self.owner:
            raise TaichiCompilationError("A component Graph root requires one declared resource owner")
        injected = produce_injected_args_for_graph(kernel, args, expected_templates)
        code = self.component.compile(entry, *injected[1:])
        for arg in flatten_args(args):
            if arg.name.startswith("__component_"):
                raise TaichiCompilationError("Explicit Graph arguments cannot use the reserved __component_ prefix")
            descriptor = describe_symbolic_arg(arg).structural_key()
            if arg.name in self.explicit and self.explicit[arg.name] != descriptor:
                raise TaichiCompilationError(f"Component Graph argument {arg.name!r} has incompatible uses")
            self.explicit[arg.name] = descriptor
        contract = code.contract
        for target, additions in ((self.groups, contract["groups"]), (self.dimensions, contract["dimensions"])):
            for name, schema in additions.items():
                if name in target and target[name] != schema:
                    raise TaichiCompilationError(f"Component Graph declaration {name!r} changed while recording")
                target[name] = schema
        hidden = []
        for name in code.dynamic:
            declaration = self.component.declarations[name]
            schema = declaration.schema
            if isinstance(declaration, DenseField):
                group, paths = contract["members"][name]
                member_key = json.dumps((group, paths), sort_keys=True)
                identity = self._members.get(member_key)
                if identity is None:
                    field = _member(owner, name)
                    normal = getattr(field, "_logical_resource_declaration", None)
                    identity = normal["identity"] if normal else name
                    if identity in self.resources:
                        raise TaichiCompilationError(f"Component resource identity collision: {identity!r}")
                    self._members[member_key] = identity
                    self.resources[identity] = dict(
                        group=group,
                        paths=paths,
                        dtype=schema[1][1],
                        element_shape=schema[1][2],
                        shape=schema[2],
                        aliases=[],
                        access="read_write",
                        effects_precision="conservative",
                        initialization="consumer_explicit",
                    )
                for alias, location in contract["members"].items():
                    if (
                        json.dumps(location, sort_keys=True) == member_key
                        and alias not in self.resources[identity]["aliases"]
                    ):
                        self.resources[identity]["aliases"].append(alias)
                symbol_name = "__component_resource_" + _digest((group, paths))
                binding = dict(kind="resource", name=identity)
                symbol = Arg(ArgKind.NDARRAY, symbol_name, declaration.dtype, ndim=len(declaration.shape))
            elif isinstance(declaration, (RuntimeValue, Dimension)):
                kind = "dimension" if isinstance(declaration, Dimension) else "value"
                symbol_name = "__component_" + kind + "_" + name
                binding = dict(kind=kind, name=name)
                dtype = declaration.annotation
                symbol = Arg(ArgKind.MATRIX if isinstance(dtype, MatrixType) else ArgKind.SCALAR, symbol_name, dtype)
                if kind == "value":
                    self.values[name] = schema
            elif isinstance(declaration, Array):
                raise TaichiCompilationError("Component Graph external Ndarray ownership is not qualified yet")
            else:
                raise TaichiCompilationError(f"Unsupported component Graph capture {name!r}")
            if symbol_name in self.bindings and self.bindings[symbol_name] != binding:
                raise TaichiCompilationError("Component Graph binding identity changed while recording")
            self.bindings[symbol_name] = binding
            hidden.append(self.symbols.setdefault(symbol_name, symbol))
        compiled = kernel.compiled_kernels[code.key]
        arguments = flatten_args((*args, *hidden))
        builder._record_dispatch(compiled, arguments, label)
        self.entries.append(
            dict(semantic_key=code.semantic_key, static=contract["static"], arguments=[arg.name for arg in arguments])
        )
        self.codes.append(code)

    def finish(self, builder):
        if not self.codes or len(builder._runtime_graph_dispatches) != len(self.codes):
            raise TaichiCompilationError("Component roots currently require a nonempty ordinary dispatch sequence")
        if builder._nodes or builder._runtime_graph_source_native_count:
            raise TaichiCompilationError("Component root control/native regions require separate qualification")
        # Detect declaration/static mutation over the whole normal recording.
        for code in self.codes:
            if not code.matches(self.component, self.owner):
                raise TaichiCompilationError("Component dependencies changed while recording its Graph")
        facts = dict(
            version=1,
            groups=self.groups,
            dimensions=self.dimensions,
            resources=self.resources,
            values=self.values,
            bindings=self.bindings,
            explicit=self.explicit,
            entries=self.entries,
            required_capabilities=["dense_snode_group_v1", "component_root_v1"],
        )
        facts["semantic_key"] = _digest(facts)
        return facts


class ComponentGraph:
    """Frozen normal Graph plus its compiler-produced resource contract."""

    def __init__(self, graph, contract):
        self._graph = graph
        self._contract = json.dumps(contract, sort_keys=True)

    @property
    def contract(self):
        return json.loads(self._contract)

    def export(self, module, name):
        if name in module._component_roots:
            raise ValueError(f"Duplicate component root {name!r}")
        plan = module.add_execution_plan(name, self._graph, snapshot_state=False)
        contract = self.contract
        contract.update(
            name=name,
            execution=name,
            target=module._arch.name,
            execution_entries=[operation["entry"] for operation in json.loads(plan.to_json())["operations"]],
        )
        module._component_roots[name] = contract
        module._content.append("component:" + name)
        return plan


def compile_component_graph(component, record, *args, **kwargs):
    from taichi_forge.graph import GraphBuilder

    capture = _GraphCapture(component)
    builder = GraphBuilder(_capture_recipe_sources=False, _component_capture=capture)
    record(builder, *args, **kwargs)
    contract = capture.finish(builder)
    graph = builder.compile()
    return ComponentGraph(graph, contract)
