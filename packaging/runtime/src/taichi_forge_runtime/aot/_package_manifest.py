"""Wire format for sharing repeated definition-package rules across kernels.

Version 99 stores admission/dependency objects as indices into a recipe table.
A recipe joins JSON fragments (one named rule or source per fragment). Keeping
indices local to the two storage fields leaves application keys unrestricted.
The public manifest and kernel binding layer see the original logical records.
"""

import json
import zlib
from collections.abc import Mapping


_SECTIONS = ("admission", "dependencies")
_POOL = "entry_contract_records"
_STORAGE_VERSION = 99


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _fragments(section):
    # Keep each named input rule or source dependency intact. Punctuation is
    # explicit so original ordering and independent mutable records survive.
    yield "{"
    for i, (name, group) in enumerate(section.items()):
        if i:
            yield ","
        yield _json(name) + ":"
        if isinstance(group, dict):
            yield "{"
            for j, (key, record) in enumerate(group.items()):
                if j:
                    yield ","
                yield _json(key) + ":" + _json(record)
            yield "}"
        elif isinstance(group, list):
            yield "["
            for j, record in enumerate(group):
                if j:
                    yield ","
                yield _json(record)
            yield "]"
        else:
            yield _json(group)
    yield "}"


def encode_manifest(manifest):
    """Select compact storage when its table and references pay for themselves."""
    fragments = {}
    recipes = {}
    headers = {}
    original = zlib.compressobj(1)
    original_size = 0
    references = []
    for symbol, entry in manifest["entries"].items():
        header = entry.copy()
        for field in _SECTIONS:
            section = entry.get(field)
            if not isinstance(section, dict) or not section:
                continue
            pieces = tuple(_fragments(section))
            recipe = tuple(fragments.setdefault(piece, len(fragments)) for piece in pieces)
            index = recipes.setdefault(recipe, len(recipes))
            header[field] = index
            references.append(index)
            original_size += len(original.compress("".join(pieces).encode("ascii")))
        headers[symbol] = header
    if not references:
        return manifest
    pool = dict(version=manifest["version"], fragments=list(fragments), recipes=[list(recipe) for recipe in recipes])
    storage_cost = len(zlib.compress(_json({_POOL: pool, "references": references}).encode("ascii"), 1))
    if storage_cost >= original_size + len(original.flush()):
        return manifest
    return {**manifest, "version": _STORAGE_VERSION, "entries": headers, _POOL: pool}


class _EntryRecords(Mapping):
    def __init__(self, headers, pool):
        self.headers = headers
        self._cache = {}
        fragments = pool["fragments"]
        if not isinstance(fragments, list) or not all(type(x) is str for x in fragments):
            raise ValueError("Invalid package contract fragments")
        for recipe in pool["recipes"]:
            if not isinstance(recipe, list) or not all(type(i) is int and 0 <= i < len(fragments) for i in recipe):
                raise ValueError("Invalid package contract fragment reference")
        self._fragments = fragments
        self._recipes = pool["recipes"]

    def __contains__(self, symbol):
        return symbol in self.headers

    def __len__(self):
        return len(self.headers)

    def __iter__(self):
        return iter(self.headers)

    def __getitem__(self, symbol):
        if symbol in self._cache:
            return self._cache[symbol]
        entry = self.headers[symbol].copy()
        for field in _SECTIONS:
            index = entry.get(field)
            if index is None or isinstance(index, dict):
                continue
            if type(index) is not int or not 0 <= index < len(self._recipes):
                raise ValueError("Invalid package entry contract reference")
            value = json.loads("".join(self._fragments[i] for i in self._recipes[index]))
            if not isinstance(value, dict):
                raise ValueError("Expected a definition package contract object")
            entry[field] = value
        return self._cache.setdefault(symbol, entry)


def decode_manifest(document):
    if _POOL not in document and document.get("version") != _STORAGE_VERSION:
        return document
    pool = document.get(_POOL)
    if (
        document.get("version") != _STORAGE_VERSION
        or not isinstance(pool, dict)
        or set(pool) != {"version", "fragments", "recipes"}
        or type(pool["version"]) is not int
        or pool["version"] <= 0
        or pool["version"] == _STORAGE_VERSION
        or not isinstance(pool["recipes"], list)
    ):
        raise ValueError("Invalid definition package contract pool")
    document.pop(_POOL)
    document["version"] = pool["version"]
    document["entries"] = _EntryRecords(document["entries"], pool)
    return document


def entry_headers(manifest):
    entries = manifest["entries"]
    return entries.headers if isinstance(entries, _EntryRecords) else entries


def materialize_manifest(manifest):
    if isinstance(manifest["entries"], _EntryRecords):
        manifest["entries"] = dict(manifest["entries"])
    return manifest
