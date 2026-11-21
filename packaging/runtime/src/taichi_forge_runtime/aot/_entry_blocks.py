"""Bounded random access to definition-package entry documents in a TCM.

Neighboring definitions share a Deflate window. The logical JSON bytes and
their existing record validation remain unchanged. A lookup inflates one
hash bucket and at most one block, irrespective of other definitions.
"""
from collections import defaultdict
from hashlib import sha256
import json

from ._component_storage import _unique_record_fields, _zip_cost, _zip_header_size

ENTRY_PREFIX = "component_entries/"
BLOCK_PREFIX = "component_entry_blocks/v1/"
INDEX_PREFIX = BLOCK_PREFIX + "index/"
BLOCK_LIMIT = 256 * 1024
RUNTIME_REQUIREMENT = "component_entry_blocks_v1"


def _index_name(name):
    return INDEX_PREFIX + sha256(name.encode("utf8")).hexdigest()[:2] + ".json"


def _block_name(number):
    return BLOCK_PREFIX + str(number) + ".bin"


def _groups(names, read):
    group, size = [], 0
    for name in names:
        data = read(name)
        if group and size + len(data) > BLOCK_LIMIT:
            yield group
            group, size = [], 0
        group.append((name, data))
        size += len(data)
    if group:
        yield group


def pack_entry_blocks(names, read, *, level):
    """Yield physical members from logical names ordered by definition."""
    indexes = defaultdict(dict)
    number = 0
    for group in _groups(names, read):
        if len(group) == 1:
            yield from group
            continue
        block_name = _block_name(number)
        block = b"".join(data for _, data in group)
        original_cost = sum(_zip_cost(name, data, level) for name, data in group)
        packed_cost = _zip_cost(block_name, block, level)
        updates = defaultdict(dict)
        offset = 0
        for name, data in group:
            index = _index_name(name)
            updates[index][name] = [number, offset, len(data)]
            offset += len(data)
        # Charge each new index its ZIP headers. Uncompressed index growth is
        # a conservative estimate for the ordinary ASCII entry names.
        for index, additions in updates.items():
            packed_cost += len(json.dumps(additions, separators=(",", ":")).encode("utf8")) + 6
            if index not in indexes:
                packed_cost += _zip_header_size(index)
        if packed_cost >= original_cost:
            yield from group
            continue
        for index, additions in updates.items():
            indexes[index].update(additions)
        yield block_name, block
        number += 1
    for index, entries in sorted(indexes.items()):
        yield index, json.dumps(entries, separators=(",", ":")).encode("utf8")


def read_entry(name, read, exists):
    """Resolve exactly one logical document, including older direct members."""
    index_name = _index_name(name)
    if not exists(index_name):
        return read(name)
    index = json.loads(read(index_name), object_pairs_hook=_unique_record_fields)
    if not isinstance(index, dict):
        raise ValueError("Invalid packed component entry index")
    if exists(name):
        if name in index:
            raise ValueError(f"Duplicate logical component entry: {name!r}")
        return read(name)
    if name not in index:
        raise ValueError(f"Missing packed component entry: {name!r}")
    record = index[name]
    if (not isinstance(record, list) or len(record) != 3
            or any(type(item) is not int or item < 0 for item in record)):
        raise ValueError("Invalid packed component entry range")
    number, start, size = record
    if not size or start + size > BLOCK_LIMIT:
        raise ValueError("Invalid packed component entry range")
    block = read(_block_name(number))
    if len(block) > BLOCK_LIMIT or start + size > len(block):
        raise ValueError("Invalid packed component entry block")
    return block[start:start + size]


def indexed_entry_names(archive):
    """Enumerate logical entries for offline metadata rewrites only."""
    for name in archive.namelist():
        if name.startswith(INDEX_PREFIX):
            entries = json.loads(archive.read(name), object_pairs_hook=_unique_record_fields)
            if not isinstance(entries, dict):
                raise ValueError("Invalid packed component entry index")
            for entry in entries:
                if not entry.startswith(ENTRY_PREFIX) or not entry.endswith(".json") or _index_name(entry) != name:
                    raise ValueError("Invalid packed component entry name")
                yield entry
