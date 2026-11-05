"""Lossless storage of repeated compiler and execution call records.

This is a serialization layer, not a code identity or execution mechanism.
Records retain every field. Readers restore the original contract before any
existing semantic, resource, or device admission checks run.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import re
import zlib


_FORMAT = "component_contract_records_v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf8")


def _record_file(digest, directory="component_contracts"):
    return directory + "/" + digest + ".json"


def _records(contract):
    for index, value in enumerate(contract.get("entries", ())):
        yield ("entries", index), value
    for index, value in enumerate(contract.get("compilation", {}).get("entries", ())):
        yield ("compilation", "entries", index), value

    def control(value, path):
        if isinstance(value, dict):
            if value.get("kind") == "kernel":
                yield path, value
            else:
                for key, child in value.items():
                    yield from control(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from control(child, path + (index,))

    yield from control(contract.get("control"), ("control",))


def _zip_cost(name, payload):
    compressor = zlib.compressobj(1, zlib.DEFLATED, -15)
    compressed = len(compressor.compress(payload) + compressor.flush())
    # Local and central-directory headers; ASCII generated names have no extras.
    return compressed + 76 + 2 * len(name.encode("utf8"))


def _unique_record_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate stored record field")
        result[key] = value
    return result


def encode_contract_documents(contracts):
    """Return filename -> bytes, using shared records only if the ZIP shrinks.

    Input keys are component document paths. Compatibility is byte-exact JSON
    equality, including target, ABI and static facts: no names are normalized.
    The comparison uses Module.archive's deflate level and includes extra file
    headers. Small contracts may intentionally retain the original representation.
    """
    return _encode_documents(contracts, _records, _FORMAT, "contract", "component_contracts")


def _encode_documents(contracts, records, storage_format, content_key, directory):
    original = {name: _json_bytes(contract) for name, contract in contracts.items()}
    counts, payloads, locations = Counter(), {}, {}
    for name, contract in contracts.items():
        for path, record in records(contract):
            if not isinstance(record, dict):
                continue
            payload = _json_bytes(record)
            digest = sha256(payload).hexdigest()
            if digest in payloads and payloads[digest] != payload:
                raise ValueError("Component record digest collision")
            payloads[digest] = payload
            counts[digest] += 1
            locations[name, path] = digest
    shared = {digest for digest, count in counts.items() if count > 1}
    if not shared:
        return original

    documents, used = {}, set()
    for name, contract in contracts.items():
        references = []

        def replace(value, path=()):
            digest = locations.get((name, path))
            if digest in shared:
                references.append([list(path), digest])
                used.add(digest)
                return None
            if isinstance(value, dict):
                return {key: replace(child, path + (key,)) for key, child in value.items()}
            if isinstance(value, list):
                return [replace(child, path + (index,)) for index, child in enumerate(value)]
            return value

        remainder = replace(contract)
        documents[name] = _json_bytes({
            "storage_format": storage_format,
            "sha256": sha256(original[name]).hexdigest(),
            content_key: remainder,
            "references": references,
        }) if references else original[name]
    documents.update((_record_file(digest, directory), payloads[digest]) for digest in sorted(used))
    if sum(_zip_cost(name, payload) for name, payload in documents.items()) >= sum(
        _zip_cost(name, payload) for name, payload in original.items()
    ):
        return original
    return documents


def decode_contract_document(document, read_bytes):
    """Expand one component; referenced records are direct JSON, never programs."""
    return _decode_document(document, read_bytes, _FORMAT, "contract", "component_contracts", _contract_path)


def _contract_path(path):
    return (len(path) == 2 and path[0] == "entries" and type(path[1]) is int
            or len(path) == 3 and path[:2] == ["compilation", "entries"] and type(path[2]) is int
            or len(path) > 1 and path[0] == "control")


def _decode_document(document, read_bytes, storage_format, content_key, directory, allowed_path):
    if not isinstance(document, dict):
        raise ValueError("Unsupported component contract")
    if "storage_format" not in document:
        return document
    if (set(document) != {"storage_format", "sha256", content_key, "references"}
            or document["storage_format"] != storage_format
            or not isinstance(document["sha256"], str)
            or not _DIGEST.fullmatch(document["sha256"])
            or not isinstance(document[content_key], dict)
            or not isinstance(document["references"], list)):
        raise ValueError("Unsupported component contract storage")
    contract, records = deepcopy(document[content_key]), {}
    for reference in document["references"]:
        if (not isinstance(reference, list) or len(reference) != 2
                or not isinstance(reference[0], list) or not reference[0]
                or any(not isinstance(part, str) and type(part) is not int for part in reference[0])
                or not isinstance(reference[1], str) or not _DIGEST.fullmatch(reference[1])):
            raise ValueError("Invalid component record reference")
        path, digest = reference
        # Only compiler record positions may be shared; identity/header fields
        # remain in the component document and are validated after expansion.
        if not allowed_path(path):
            raise ValueError("Invalid component record path")
        parent = contract
        try:
            for part in path[:-1]:
                if not (isinstance(parent, dict) and isinstance(part, str)
                        or isinstance(parent, list) and type(part) is int and part >= 0):
                    raise ValueError("Invalid component record path")
                parent = parent[part]
            part = path[-1]
            if not (isinstance(parent, dict) and isinstance(part, str)
                    or isinstance(parent, list) and type(part) is int and part >= 0):
                raise ValueError("Invalid component record path")
            if parent[part] is not None:
                raise ValueError("Component record reference must replace an empty slot exactly once")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Invalid component record path") from exc
        if digest not in records:
            payload = read_bytes(_record_file(digest, directory))
            if sha256(payload).hexdigest() != digest:
                raise ValueError("Component record digest mismatch")
            record = json.loads(payload, object_pairs_hook=_unique_record_fields)
            if not isinstance(record, dict):
                raise ValueError("Invalid component compiler record")
            records[digest] = record
        # Separate logical calls and component instances must never share mutable
        # Python contract dictionaries as a consequence of disk interning.
        parent[part] = deepcopy(records[digest])
    if sha256(_json_bytes(contract)).hexdigest() != document["sha256"]:
        raise ValueError(f"Expanded {content_key} digest mismatch")
    return contract


def _plan_records(plan):
    def visit(value, path):
        if isinstance(value, dict):
            if value.get("kind") == "kernel":
                yield path, value
            else:
                for key, child in value.items():
                    yield from visit(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from visit(child, path + (index,))

    for key in ("operations", "initialization"):
        yield from visit(plan.get(key), (key,))


def encode_plan_documents(plans):
    """Share exact kernel call records; keep control, resources and order intact.

    This is an on-disk representation only. Runtime expands the document before
    the ordinary ExecutionPlan parser and native preparation see it.
    """
    return _encode_documents(plans, _plan_records, "execution_plan_records_v1", "plan", "execution_records")


def decode_plan_document(document, read_bytes):
    """Restore a plan before its existing schema and execution-domain checks."""
    return _decode_document(document, read_bytes, "execution_plan_records_v1", "plan", "execution_records",
                            lambda path: len(path) > 1 and path[0] in ("operations", "initialization"))
