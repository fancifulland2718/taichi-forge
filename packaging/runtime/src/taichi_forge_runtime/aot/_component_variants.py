"""Cold selection of already exported component configurations."""

import json
from urllib.parse import quote

from .._static_values import read_static_key, static_key
from ._component_info import _read_document, inspect_component


def _configuration_key(configuration):
    return tuple(sorted((name, static_key(value)) for name, value in configuration.items()))


def _decode(configuration):
    return {name: read_static_key(value) for name, value in configuration.items()}


def _read_family(path, name):
    from ._runtime import _name

    _name(name)
    return _read_document(path, "component_families/" + quote(name, safe="-_.") + ".json")


class _FamilySelection:
    """One contract/coverage/selection rule for inspection and loaded families."""

    def __init__(self, contract, name, target, get_component):
        if (
            set(contract)
            != {
                "version",
                "name",
                "target",
                "domain",
                "cases",
                "build_report",
                "exported_kernel_entries",
                "coverage",
                "required_capabilities",
            }
            or contract["version"] != 1
            or contract["name"] != name
            or contract["target"] != target
            or contract["coverage"] != "exact_requested_configurations"
            or contract["required_capabilities"] != ["component_variant_selection_v1"]
        ):
            raise ValueError("Unsupported component family contract")
        self._choices, self._components = {}, {}
        for key, choices in contract["domain"].items():
            if not key.isidentifier() or not choices:
                raise ValueError("Invalid component family domain")
            normalized = tuple(static_key(read_static_key(value)) for value in choices)
            if len(set(normalized)) != len(normalized):
                raise ValueError("Duplicate component family choices")
            self._choices[key] = set(normalized)
        if not self._choices or not contract["cases"]:
            raise ValueError("Component family has no declared configurations")
        for case in contract["cases"]:
            if set(case) != {"configuration", "root", "semantic_key"}:
                raise ValueError("Invalid component family case")
            values = _decode(case["configuration"])
            key = self._admit(values)
            if key in self._components:
                raise ValueError("Ambiguous component family configuration")
            component = get_component(case["root"])
            facts = component.contract
            actual = facts.get("configuration", {})
            if (
                facts["semantic_key"] != case["semantic_key"]
                or facts["target"] != target
                or set(actual) != set(self._choices)
                or any(
                    static_key(read_static_key(spec["selected"])) != static_key(values[name])
                    or {static_key(read_static_key(item)) for item in spec["choices"]} != self._choices[name]
                    for name, spec in actual.items()
                )
            ):
                raise ValueError("Component family differs from its actual compiled root")
            self._components[key] = component
        report = contract["build_report"]
        if (
            report.get("version") != 1
            or report.get("missing_configurations") != []
            or report.get("request_admission_complete") is not True
            or {_configuration_key(_decode(value)) for value in report["requested_configurations"]}
            != set(self._components)
            or report["requested_configurations"] != report["completed_configurations"]
        ):
            raise ValueError("Component family coverage does not match its compiled closure")
        self._contract = json.dumps(contract, sort_keys=True)
        self._target = target

    def _check(self):
        pass

    def _admit(self, configuration):
        if set(configuration) != set(self._choices):
            raise ValueError("Component family configuration inputs differ from its declared domain")
        for name, value in configuration.items():
            if static_key(value) not in self._choices[name]:
                raise ValueError(f"Component family configuration {name!r} is outside its declared domain")
        return _configuration_key(configuration)

    @property
    def contract(self):
        self._check()
        return json.loads(self._contract)

    def select(self, configuration):
        self._check()
        key = self._admit(dict(configuration))
        if key not in self._components:
            raise ValueError(
                "No exported component variant for this legal configuration; rebuild its declared coverage"
            )
        return self._components[key]

    @property
    def target(self):
        self._check()
        return self._target

    def configuration_sources(self):
        """Query provenance only if every case uses the same normal inputs.

        No representative case is chosen on the caller's behalf. Existing
        families without common provenance remain explicitly selectable.
        """
        self._check()
        sources = None
        for component in self._components.values():
            current = component.configuration_sources()
            if sources is None:
                sources = current
            elif sources != current:
                raise ValueError("Component family cases have different configuration source paths")
        return sources


class ComponentVariantsInfo(_FamilySelection):
    """Cold inspection without native load or device admission.

    Selection checks actual root contracts and exact declared build coverage.
    Loading still checks Plan/kernel closure, capabilities and resources.
    """

    def __init__(self, path, name):
        contract = _read_family(path, name)
        super().__init__(contract, name, contract.get("target"), lambda root: inspect_component(path, root))


def inspect_component_variants(path, name):
    """Inspect an exported family; select() returns an ordinary ComponentInfo."""
    return ComponentVariantsInfo(path, name)


class ComponentVariants(_FamilySelection):
    """A selector over existing component/ExecutionPlan contracts, not a VM."""

    def __init__(self, module, name):
        with module._runtime._lock:
            module._check()
            self._module = module
            contract = _read_family(module._source_path, name)
            super().__init__(contract, name, module._runtime.arch, module.component)
            kernels = set()
            for component in self._components.values():
                kernels.update(component._plan.kernel_entries)
                kernels.update(entry for operations in component.contract.get("field_io_bindings", {}).values()
                               for entry in operations.values())
            if set(contract["exported_kernel_entries"]) != kernels:
                raise ValueError("Component family coverage does not match its compiled closure")

    def _check(self):
        self._module._check()

    def instantiate(self, dimensions, *, configuration, inputs=None):
        return self.select(configuration).instantiate(dimensions, configuration=configuration, inputs=inputs)
