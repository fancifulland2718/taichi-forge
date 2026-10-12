"""Portable native object envelopes shared by operations and resource recipes."""

NATIVE_FIELDS = frozenset({"kind", "provider", "operation", "version", "parameters", "bindings"})


def native_references(description):
    from ._runtime import _name

    _name(description["provider"])
    _name(description["operation"])
    version = description["version"]
    if type(version) is not int or not 0 < version <= 0xFFFFFFFF:
        raise ValueError("Native operation version must be a positive uint32")
    if not isinstance(description["parameters"], dict) or not isinstance(description["bindings"], dict):
        raise ValueError("Native parameters and bindings must be objects")
    for port, slot in description["bindings"].items():
        _name(port)
        _name(slot)
    return tuple(description["bindings"].values())
