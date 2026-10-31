"""Read compiler-produced component facts without loading native code.

Inspection shares header/static validation with Component. It does not admit
device execution, allocate resources, or interpret layouts or execution plans.
"""
import json
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

from .._static_values import read_static_key, static_key


def _read_document(path, relative):
    path = Path(path).resolve(strict=True)
    if path.is_dir():
        payload = (path / relative).read_bytes()
    else:
        with ZipFile(path) as archive:
            if archive.namelist().count(relative) != 1:
                raise ValueError(f"Artifact document must occur exactly once: {relative!r}")
            payload = archive.read(relative)
    return json.loads(payload)


def _read_contract(path, name):
    from ._runtime import _name

    _name(name)
    contract = _read_document(path, "components/" + quote(name, safe="-_.") + ".json")
    _validate_header(contract, name)
    return contract


def _read_device_requirements(path):
    # GFX's serialized map is the native loader's authority, including entries
    # other than the inspected component. Do not infer requirements from SPIR-V
    # or copy them into a second component schema.
    metadata = _read_document(path, "metadata.json")
    entries = metadata.get("required_caps") if isinstance(metadata, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Artifact has no queryable device capability requirements; rebuild it")
    result = {}
    for entry in entries:
        if (
            not isinstance(entry, dict) or set(entry) != {"key", "value"}
            or not isinstance(entry["key"], str) or not entry["key"]
            or type(entry["value"]) is not int or not 0 <= entry["value"] <= 0xFFFFFFFF
            or entry["key"] in result
        ):
            raise ValueError("Invalid module device capability requirements")
        result[entry["key"]] = entry["value"]
    return result


def _validate_header(contract, name):
    required = {
        "version",
        "groups",
        "dimensions",
        "resources",
        "values",
        "bindings",
        "explicit",
        "entries",
        "required_capabilities",
        "semantic_key",
        "name",
        "execution",
        "execution_entries",
        "target",
    }
    if contract.get("version") in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        required.add("compilation")
    if contract.get("version") in (3, 4, 5, 6, 7, 8, 9, 10):
        required.add("configuration")
    if contract.get("version") in (4, 5, 6, 7, 8, 9, 10):
        required.update(("control", "internal_arguments", "execution_fingerprint"))
    if contract.get("version") in (5, 6, 7, 8, 9, 10):
        required.add("constraints")
    if contract.get("version") in (6, 7, 8, 9, 10):
        required.update(("fixed_groups", "fixed_bindings"))
    if contract.get("version") in (7, 8, 9, 10):
        required.update(("field_io", "field_io_bindings"))
    if contract.get("version") in (8, 9, 10):
        required.add("lifecycle")
    if contract.get("version") in (9, 10):
        required.add("setup_bindings")
    if contract.get("version") == 10:
        required.add("operation_bindings")
    host_resource_capability = "component_host_resources_v1" in contract.get("required_capabilities", ())
    if host_resource_capability:
        required.add("host_resources")
    private_shape_capability = "component_private_array_extents_v1" in contract.get("required_capabilities", ())
    if private_shape_capability:
        required.add("private_shapes")
    if set(contract) != required or contract["version"] not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) or contract["name"] != name:
        raise ValueError("Unsupported component contract")


def _configuration(contract):
    result = {}
    if contract["version"] == 3 or (contract["version"] >= 4 and contract["configuration"]):
        if not isinstance(contract["configuration"], dict) or not contract["configuration"]:
            raise ValueError("Invalid component configuration domain")
        for key, spec in contract["configuration"].items():
            if not key.isidentifier() or set(spec) != {"selected", "choices"} or not spec["choices"]:
                raise ValueError("Invalid component static choice declaration")
            selected = static_key(read_static_key(spec["selected"]))
            choices = tuple(static_key(read_static_key(value)) for value in spec["choices"])
            if len(set(choices)) != len(choices) or selected not in choices:
                raise ValueError("Component selected configuration is outside its domain")
            for entry in contract["entries"]:
                for dependency, value in entry["static"]:
                    if dependency == key and static_key(read_static_key(value)) != selected:
                        raise ValueError("Component configuration differs from compiled static dependencies")
            result[key] = (selected, choices)
    configuration_sources = contract.get("compilation", {}).get("configuration_sources")
    if "configuration_sources" in contract.get("compilation", {}):
        if not isinstance(configuration_sources, dict) or set(configuration_sources) != set(result):
            raise ValueError("Component configuration sources must cover the admitted inputs")
        for source in configuration_sources.values():
            if (
                not isinstance(source, dict) or set(source) != {"kind", "path"}
                or source["kind"] not in ("attribute", "recording_argument")
                or not isinstance(source["path"], list) or not source["path"]
                or any(not isinstance(part, str) or not part.isidentifier() for part in source["path"])
                or source["kind"] == "recording_argument" and len(source["path"]) != 1
            ):
                raise ValueError("Invalid component configuration source path")
    return result


def _check_configuration(expected, configuration):
    supplied = dict(configuration or {})
    if set(supplied) != set(expected):
        raise ValueError(
            "Component configuration inputs differ: "
            f"missing={sorted(set(expected)-set(supplied))}, "
            f"extra={sorted(set(supplied)-set(expected))}"
        )
    for name, value in supplied.items():
        selected, choices = expected[name]
        actual = static_key(value)
        if actual not in choices:
            raise ValueError(f"Component configuration {name!r} is outside its declared domain")
        if actual != selected:
            raise ValueError(f"Component configuration {name!r} requires a different compiled variant")


class ComponentInfo:
    """Immutable inspection snapshot; matching static inputs is not admission.

    Loading through Runtime/Module and instantiate/prepare must still validate
    target capabilities, numerical entries, layouts, resources and lifetimes.
    The returned dictionaries are copies of the actual artifact document.
    """

    def __init__(self, path, name):
        contract = _read_contract(path, name)
        self._configuration = _configuration(contract)
        self._contract_json = json.dumps(contract, sort_keys=True)
        self._sources_json = json.dumps(contract.get("compilation", {}).get("configuration_sources"))
        self._target = contract["target"]
        self._semantic_key = contract["semantic_key"]
        self._device_requirements = _read_device_requirements(path) if self._target == "vulkan" else None

    @property
    def target(self):
        return self._target

    @property
    def semantic_key(self):
        return self._semantic_key

    @property
    def contract(self):
        return json.loads(self._contract_json)

    def configuration_sources(self):
        sources = json.loads(self._sources_json)
        if sources is None:
            raise ValueError("Artifact has no configuration source paths; rebuild it to query model provenance")
        return sources

    def required_device_capabilities(self):
        """Return module-wide Vulkan requirements from native GFX metadata.

        This is separate from component protocol capabilities. Other targets
        currently have no equivalent query here; absence of a query must not
        be interpreted as requiring no device capabilities.
        """
        if self._device_requirements is None:
            raise NotImplementedError("Device requirement inspection currently supports Vulkan artifacts")
        return dict(self._device_requirements)

    def check_device_capabilities(self, capabilities):
        """Filter cold candidates against Runtime.device_capabilities().

        Missing capabilities have level zero. Native load/prepare still perform
        admission; satisfying this query does not validate layouts or limits.
        """
        required = self.required_device_capabilities()
        available = dict(capabilities)
        if any(type(level) is not int or not 0 <= level <= 0xFFFFFFFF for level in available.values()):
            raise ValueError("Device capability levels must be uint32 integers")
        missing = [
            f"{name}: requires {level}, available {available.get(name, 0)}"
            for name, level in sorted(required.items()) if available.get(name, 0) < level
        ]
        if missing:
            raise ValueError("Module device capabilities are not satisfied: " + "; ".join(missing))

    def check_configuration(self, configuration=None):
        """Check exact declared static inputs using instantiate's existing rule.

        This is a cold selection aid. Dynamic dimension/resource compatibility
        and native capability admission remain the loaded component's task.
        """
        _check_configuration(self._configuration, configuration)


def inspect_component(path, name):
    """Inspect one exported root in a TCM/directory without creating a Runtime."""
    return ComponentInfo(path, name)
