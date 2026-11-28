"""Bounded storage blocks for similar unified LLVM roots (index version 3)."""

from collections import defaultdict
import hashlib
import zlib


STRIPE = 1024
BLOCK_LIMIT = 512 * 1024


def block_name(group):
    return "unit_blocks/" + group[0][0] + ".bin"


def block_candidates(entries, roots, sizes):
    # A shared physical root has one deterministic owner even across aliases.
    owners = {}
    for symbol, header in entries.items():
        digest = roots[symbol]
        name = header["definition"]
        if digest not in owners or name < owners[digest]:
            owners[digest] = name
    families = defaultdict(list)
    for digest, name in owners.items():
        if sizes[digest] <= BLOCK_LIMIT:
            families[name].append((digest, sizes[digest]))
    for name in sorted(families):
        group, total = [], 0
        for digest, size in sorted(families[name], key=lambda member: (member[1], member[0])):
            if len(group) == 4 or total + size > BLOCK_LIMIT:
                if len(group) > 1:
                    yield tuple(group)
                group, total = [], 0
            group.append((digest, size))
            total += size
        if len(group) > 1:
            yield tuple(group)


def stripe_units(units):
    return b"".join(unit[offset:offset + STRIPE]
                    for offset in range(0, max(map(len, units)), STRIPE) for unit in units)


def unpack_units(group, data):
    if len(data) != sum(size for _, size in group):
        raise ValueError("Invalid unified LLVM unit block size")
    result = {digest: bytearray() for digest, _ in group}
    cursor = 0
    for offset in range(0, max(size for _, size in group), STRIPE):
        for digest, size in group:
            if offset < size:
                span = min(STRIPE, size - offset)
                result[digest].extend(data[cursor:cursor + span])
                cursor += span
    return result


def index_blocks(archive, index):
    # Only called for a hash-pinned incremental build input.
    locations = {}
    for rows in index["unit_blocks"]:
        group = tuple((digest, size) for digest, size in rows)
        if (not 2 <= len(group) <= 4 or
                any(type(size) is not int or size <= 0 for _, size in group)):
            raise ValueError("Invalid unified LLVM unit block layout")
        total = sum(size for _, size in group)
        if total > BLOCK_LIMIT or archive.getinfo(block_name(group)).file_size != total:
            raise ValueError("Invalid unified LLVM unit block size")
        for digest, size in group:
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in "0123456789abcdef" for c in digest) or digest in locations):
                raise ValueError("Invalid unified LLVM unit block digest")
            locations[digest] = (group, size)
    return locations


def deflate_size(data, level):
    return len(zlib.compress(data, level)) - 6  # exclude the zlib header and Adler32


def verify_unit(digest, data):
    if not data.startswith(b"BC\xc0\xde") or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Invalid unified LLVM unit: {digest}")
    return data
