"""Select records in a shared LLVM definition library without touching device code."""

import copy
import importlib.util


def differences(base, view):
    """Return replacements and deletions. A missing value is distinct from JSON null."""
    changes = []

    def compare(path, left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in left:
                if key not in right:
                    changes.append([path + [key]])
            for key, item in right.items():
                if key in left:
                    compare(path + [key], left[key], item)
                else:
                    changes.append([path + [key], item])
        elif type(left) is not type(right) or left != right:
            changes.append([path, right])

    compare([], base, view)
    return changes


def select_records(base, changes):
    # Copy only containers on replacement paths. Shared immutable records
    # are already private to this consumer's JSON decoder.
    result = copy.copy(base)
    copied = {()}
    for change in changes:
        path = change[0]
        if not path:
            result = change[1]
            continue
        current = result
        for index, part in enumerate(path[:-1]):
            prefix = tuple(path[:index + 1])
            if prefix not in copied:
                current[part] = current[part].copy()
                copied.add(prefix)
            current = current[part]
        if len(change) == 1:
            del current[path[-1]]
        else:
            current[path[-1]] = change[1]
    return result


def select_manifest(document, arch):
    from ._package_manifest import _EntryRecords, decode_manifest

    if document.get("kind") != "forge_unified_definition_library" or document.get("version") not in (1, 2):
        raise ValueError("Unsupported unified LLVM library")
    target = "x64" if arch == "cpu" else arch
    if target not in document["targets"]:
        raise ValueError(f"Unified LLVM library has no implementation for {target}")
    base = decode_manifest(document["base"].copy())
    if document["version"] == 2:
        if base["target"] != "unified-llvm" or not isinstance(document["targets"], list):
            raise ValueError("Invalid portable LLVM manifest")
        base["frontend_target"] = target
        base["portable_llvm"] = True
        base["target_capabilities"] = document["target_capabilities"].get(target, [])
        resources = base.pop("device_resources", {}).get(target)
        if resources:
            base["owners"] = select_records(base["owners"], resources)
        if "field_groups" in base:
            base["field_groups"] = {
                name: {**contract, "device_members": contract["device_field_declarations"].get(target, {})}
                if "device_field_declarations" in contract else contract
                for name, contract in base["field_groups"].items()
            }
        base["files"] = {**base["files"], "code.tcm": document["tcm_sha256"]}
        return base
    changes = document["targets"][target]
    entries = base["entries"]
    if isinstance(entries, _EntryRecords):
        # Keep shared contract fragments lazy. Materialize only entries whose
        # logical ABI actually differs in this projection.
        general, per_entry = [], {}
        for change in changes:
            path, *value = change
            if len(path) >= 2 and path[0] == "entries":
                per_entry.setdefault(path[1], []).append([path[2:], *value])
            else:
                general.append(change)
        if per_entry:
            selected = copy.copy(entries)
            selected.headers = entries.headers.copy()
            selected._cache = {}
            for symbol, patches in per_entry.items():
                if patches[0][0]:
                    selected.headers[symbol] = select_records(entries[symbol], patches)
                elif len(patches[0]) == 1:
                    del selected.headers[symbol]
                else:
                    selected.headers[symbol] = patches[0][1]
            base["entries"] = selected
        changes = general
    manifest = select_records(base, changes)
    if manifest["frontend_target"] != target or manifest["target"] != "unified-llvm":
        raise ValueError("Invalid unified LLVM target selection")
    manifest["files"] = {**manifest["files"], "code.tcm": document["tcm_sha256"]}
    return manifest


class HostProjection:
    """Resolve imports in a loaded package's private namespace to its selected files."""

    def __init__(self, namespace, directory, manifest):
        self.namespace = namespace
        self.modules = {}
        overrides = manifest.get("host_paths", {})
        for relative in manifest["files"]:
            if not relative.startswith("host/") or not relative.endswith(".py"):
                continue
            parts = relative[5:-3].split("/")
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            name = ".".join([namespace, *parts])
            path = directory / overrides.get(relative, relative)
            locations = [str((directory / relative).parent)] if is_package else None
            self.modules[name] = (path, locations)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.namespace and not fullname.startswith(self.namespace + "."):
            return None
        try:
            source, locations = self.modules[fullname]
        except KeyError:
            raise ModuleNotFoundError(f"Module outside the selected Host projection: {fullname}", name=fullname) from None
        return importlib.util.spec_from_file_location(fullname, source, submodule_search_locations=locations)
