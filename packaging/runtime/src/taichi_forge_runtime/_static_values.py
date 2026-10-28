"""Typed immutable configuration values shared by compilation and consumption."""


def static_key(value):
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is tuple:
        return ("tuple", tuple(static_key(item) for item in value))
    raise ValueError("Component static captures must be immutable scalar values or tuples")


def read_static_key(key):
    """Validate a serialized key without coercing booleans, numbers or tuples."""
    if not isinstance(key, (list, tuple)) or len(key) != 2:
        raise ValueError("Invalid component static value key")
    kind, value = key
    if kind == "tuple" and isinstance(value, (list, tuple)):
        result = tuple(read_static_key(item) for item in value)
    elif kind == "float" and type(value) is str:
        result = float.fromhex(value)
        if result.hex() != value:
            raise ValueError("Noncanonical component static float")
    elif kind == "NoneType" and value is None:
        result = None
    elif kind in ("bool", "int", "str") and type(value).__name__ == kind:
        result = value
    else:
        raise ValueError("Invalid component static value type")
    return result
