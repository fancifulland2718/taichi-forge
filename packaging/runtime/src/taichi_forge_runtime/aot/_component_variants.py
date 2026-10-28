"""Cold selection of already exported component configurations."""

import json
from urllib.parse import quote
from zipfile import ZipFile

from .._static_values import read_static_key, static_key


def _configuration_key(configuration):
    return tuple(sorted((name, static_key(value)) for name, value in configuration.items()))


def _decode(configuration):
    return {name: read_static_key(value) for name, value in configuration.items()}


class ComponentVariants:
    """A selector over existing component/ExecutionPlan contracts, not a VM."""

    def __init__(self, module, name):
        from ._runtime import _name

        _name(name)
        with module._runtime._lock:
            module._check()
            relative = "component_families/" + quote(name, safe="-_.") + ".json"
            if module._source_path.is_dir():
                payload = (module._source_path / relative).read_bytes()
            else:
                with ZipFile(module._source_path) as archive:
                    payload = archive.read(relative)
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError("Component family contract is too large")
            contract = json.loads(payload)
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
                or contract["target"] != module._runtime.arch
                or contract["coverage"] != "exact_requested_configurations"
                or contract["required_capabilities"] != ["component_variant_selection_v1"]
            ):
                raise ValueError("Unsupported component family contract")
            self._module, self._choices, self._components = module, {}, {}
            for key, choices in contract["domain"].items():
                if not key.isidentifier() or not choices:
                    raise ValueError("Invalid component family domain")
                normalized = tuple(static_key(read_static_key(value)) for value in choices)
                if len(set(normalized)) != len(normalized):
                    raise ValueError("Duplicate component family choices")
                self._choices[key] = set(normalized)
            if not self._choices or not contract["cases"]:
                raise ValueError("Component family has no declared configurations")
            kernels = set()
            for case in contract["cases"]:
                if set(case) != {"configuration", "root", "semantic_key"}:
                    raise ValueError("Invalid component family case")
                values = _decode(case["configuration"])
                key = self._admit(values)
                if key in self._components:
                    raise ValueError("Ambiguous component family configuration")
                component = module.component(case["root"])
                facts = component.contract
                actual = facts.get("configuration", {})
                if (
                    facts["semantic_key"] != case["semantic_key"]
                    or set(actual) != set(self._choices)
                    or any(
                        static_key(read_static_key(spec["selected"])) != static_key(values[name])
                        or {static_key(read_static_key(item)) for item in spec["choices"]} != self._choices[name]
                        for name, spec in actual.items()
                    )
                ):
                    raise ValueError("Component family differs from its actual compiled root")
                self._components[key] = component
                kernels.update(component._plan.kernel_entries)
            report = contract["build_report"]
            if (
                report.get("version") != 1
                or report.get("missing_configurations") != []
                or report.get("request_admission_complete") is not True
                or {_configuration_key(_decode(value)) for value in report["requested_configurations"]}
                != set(self._components)
                or report["requested_configurations"] != report["completed_configurations"]
                or set(contract["exported_kernel_entries"]) != kernels
            ):
                raise ValueError("Component family coverage does not match its compiled closure")
            self._contract = json.dumps(contract, sort_keys=True)

    def _admit(self, configuration):
        if set(configuration) != set(self._choices):
            raise ValueError("Component family configuration inputs differ from its declared domain")
        for name, value in configuration.items():
            if static_key(value) not in self._choices[name]:
                raise ValueError(f"Component family configuration {name!r} is outside its declared domain")
        return _configuration_key(configuration)

    @property
    def contract(self):
        self._module._check()
        return json.loads(self._contract)

    def select(self, configuration):
        self._module._check()
        key = self._admit(dict(configuration))
        if key not in self._components:
            raise ValueError(
                "No exported component variant for this legal configuration; rebuild its declared coverage"
            )
        return self._components[key]

    def instantiate(self, dimensions, *, configuration):
        return self.select(configuration).instantiate(dimensions, configuration=configuration)
