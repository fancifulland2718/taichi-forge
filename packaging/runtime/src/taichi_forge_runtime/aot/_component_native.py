"""Validate resource references in compiler-produced native component facts.

Execution and storage validation remain in the ordinary ExecutionPlan/native
provider. This projection never prepares or interprets a numerical operation.
"""


def operation_bindings(operation):
    if not isinstance(operation, dict):
        raise ValueError("Invalid component native operation")
    kind = operation.get("kind")
    if kind == "sort":
        if not {"kind", "keys"} <= set(operation) or set(operation) - {"kind", "keys", "values"}:
            raise ValueError("Invalid component native sort operation")
        ports = [(key, value) for key, value in operation.items() if key != "kind"]
    elif kind == "compact":
        if set(operation) != {"kind", "values", "flags", "output", "count"}:
            raise ValueError("Invalid component native compact operation")
        ports = [(key, value) for key, value in operation.items() if key != "kind"]
    elif kind == "native":
        if (set(operation) != {"kind", "provider", "operation", "version", "parameters", "bindings"}
                or operation["provider"] not in ("builtin", "vulkan_ray", "optix") or type(operation["version"]) is not int
                or not isinstance(operation["bindings"], dict)):
            raise ValueError("Invalid component native definition")
        parameters, bindings = operation["parameters"], operation["bindings"]
        version = operation["version"]
        if version != 1 and not (operation["provider"] == "optix" and operation["operation"] == "query" and version in (4, 5)):
            raise ValueError("Unqualified component native operation version")
        if operation["provider"] == "optix" and operation["operation"] == "query":
            if (version not in (4, 5) or not isinstance(parameters, dict)
                    or set(parameters) != {"occlusion", "any_hit", "graph_fixed_query"} | ({"front_only"} if version == 4 else set())
                    or any(type(value) is not bool for value in parameters.values())
                    or parameters["occlusion"] and not parameters["any_hit"]
                    or set(bindings) != {"scene", "rays", "hits"} | (set() if parameters["occlusion"] else {"hit_indices"}) | ({"faces"} if version == 5 else set())):
                raise ValueError("Invalid component OptiX query definition")
            if len(set(bindings.values())) != len(bindings):
                raise ValueError("Component OptiX ports require distinct resources")
            ports = list(bindings.items())
        elif operation["provider"] in ("vulkan_ray", "optix"):
            if (operation["operation"] != "instance_refit" or parameters != {}
                    or set(bindings) != {"scene", "transforms"} or bindings["scene"] == bindings["transforms"]):
                raise ValueError("Invalid component native ray instance refit definition")
            ports = list(bindings.items())
        elif operation["operation"] == "sort":
            if (parameters != {"nan_policy": "last"} or not {"keys"} <= set(bindings)
                    or set(bindings) - {"keys", "values"}):
                raise ValueError("Invalid component prepared sort definition")
            ports = list(bindings.items())
        elif operation["operation"] == "compact":
            if (not isinstance(parameters, dict) or set(parameters) != {"column_count"}
                    or type(parameters["column_count"]) is not int or parameters["column_count"] <= 0):
                raise ValueError("Invalid component compact columns")
            count = parameters["column_count"]
            # Check length before enumerating ports from untrusted metadata.
            if len(bindings) != 2 + 2 * count:
                raise ValueError("Invalid component compact column bindings")
            expected = {"flags", "count"} | {f"{role}_{i}" for i in range(count) for role in ("values", "output")}
            if set(bindings) != expected:
                raise ValueError("Invalid component compact column bindings")
            ports = [(name.split("_")[0], value) for name, value in bindings.items()]
        else:
            raise ValueError("Unqualified component native operation")
    else:
        raise ValueError("Unqualified component native operation")
    if any(not isinstance(value, str) or not value or "\0" in value for _, value in ports):
        raise ValueError("Component native ports require resource names")
    return ports


def map_operation_bindings(operation, resolve):
    operation_bindings(operation)
    if operation["kind"] == "native":
        return {**operation, "bindings": {port: resolve(value) for port, value in operation["bindings"].items()}}
    return {key: value if key == "kind" else resolve(value) for key, value in operation.items()}
