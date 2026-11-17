"""Project optional input records and complete allocation groups.

Whole optional Dense groups retain their native layout and allocation lifetime.
Borrowed query ports may project conditional members of a shared allocation;
the original owner still retains its complete layout contract.
"""

def project_input_records(owner, schema, declarations):
    from taichi_forge.types import primitive_types
    from ._component import _InputPresence, _InputRecord
    from ._package_records import write_member

    for name, record in list(schema.items()):
        if record["kind"] != "record" or not record.get("optional") or record.get("struct_field"):
            continue
        path = tuple(record.get("path", (name,)))
        members = {
            key: item
            for key, item in schema.items()
            if len(item.get("path", (key,))) > len(path) and tuple(item.get("path", (key,)))[: len(path)] == path
        }
        if not members or any(
            not (
                item["kind"] == "dimension"
                or item["kind"] == "value"
                and item.get("element", {}).get("kind", "scalar") == "scalar"
                or item["kind"] == "ndarray"
                and item.get("external")
                and not item.get("optional_input")
            )
            for item in members.values()
        ):
            continue
        # Whole present-only groups can share code without sharing storage.
        # Partial group layouts and dimensions escaping the record need their
        # existing layout contract, not an inferred nullable descriptor.
        axes = {key for key, item in members.items() if item["kind"] == "dimension"}

        def uses_axis(value):
            if isinstance(value, str):
                return value in axes
            if isinstance(value, dict):
                return any(uses_axis(child) for child in value.values())
            return isinstance(value, (tuple, list)) and any(uses_axis(child) for child in value)

        conditioned = {key: item for key, item in schema.items() if name in item.get("conditions", {})}
        storage = {key: item for key, item in conditioned.items() if item["kind"] == "dense_field"}
        groups = {item.get("group", "fields") for item in storage.values()}
        if (
            any(
                item.get("conditions") != {name: True}
                or item["kind"] not in ("dense_field", "dimension")
                or item["kind"] == "dense_field"
                and (item.get("struct_field") or item["element"].get("model_tensor") or not item.get("group_builder"))
                for item in conditioned.values()
            )
            or any(
                item["kind"] == "dense_field" and item.get("group", "fields") in groups and key not in storage
                for key, item in schema.items()
            )
            or any(
                key not in members
                and any(
                    uses_axis(item.get(part, ())) for part in ("expression", "shape", "shape_expressions", "element")
                )
                for key, item in schema.items()
            )
        ):
            # A first edge leaving the nullable namespace also covers derived
            # dimensions that subsequently feed an owned layout or another
            # input. Those consumers need a joint presence/binding contract.
            continue
        presence = "_forge_package_record_present_" + name
        record.update(dynamic_input_record=True, input_presence_attribute=presence)
        schema[presence] = dict(kind="value", dtype="i32", input_presence=True)
        declarations[presence] = _InputPresence(primitive_types.i32)
        declarations[name] = _InputRecord(presence)
        write_member(owner, presence, 1, schema)
        for item in members.values():
            item["input_record"] = name
        for item in storage.values():
            item["input_record"] = name
            item["dynamic_input_storage"] = True


def project_boolean_groups(schema, declarations):
    """Reuse nullable Dense arguments for a whole present-only allocation group.

    Keep the original Host boolean declaration and allocation. Only its device
    use becomes an immutable invocation value. Mixed groups, layout selection,
    child construction and other conditional declarations remain specialized.
    """
    from taichi_forge.types import primitive_types
    from ._component import _InputPresence

    candidates = {
        name
        for name, predicate in schema.items()
        if predicate["kind"] == "static"
        and not predicate.get("conditions")
        and not predicate.get("kernel_only")
        and not predicate.get("static_dtype")
        and {tuple(value) for value in predicate.get("choices", ())} == {("bool", False), ("bool", True)}
    }

    def eligible(name):
        conditioned = {key: item for key, item in schema.items() if name in item.get("conditions", {})}
        storage = {key: item for key, item in conditioned.items() if item["kind"] == "dense_field"}
        groups = {item.get("group", "fields") for item in storage.values()}
        return not (
            not storage
            or any(
                not item.get("conditions", {}).keys() <= candidates
                or any(value is not True for value in item.get("conditions", {}).values())
                # A conditionally declared dimension can be read without its
                # Field. It needs a presence contract of its own before it can
                # use a nullable argument, unlike an unconditional shape axis.
                or item["kind"] != "dense_field"
                or item["element"].get("model_tensor")
                or any(member.get("model_tensor") for member in item["element"].get("members", {}).values())
                or not item.get("group_builder")
                for item in conditioned.values()
            )
            or any(
                item["kind"] == "dense_field" and item.get("group", "fields") in groups and key not in storage
                for key, item in schema.items()
            )
            or any(
                name in item.get(guards, {})
                for item in schema.values()
                for guards in ("choice_guards", "shape_guards", "dtype_guards", "source_guards")
            )
        )

    # A nested group's predicates form one eligibility closure. If one guard
    # also selects a type/layout or a partial group, retain specialization for
    # every group relying on that guard. No application policy is inferred.
    while True:
        accepted = {name for name in candidates if eligible(name)}
        if accepted == candidates:
            break
        candidates = accepted
    for name in sorted(candidates):
        predicate = schema[name]
        predicate.update(dynamic_storage_condition=True, input_presence_attribute=name)
        declarations[name] = _InputPresence(primitive_types.i32)
    for item in schema.values():
        conditions = item.get("conditions", {})
        if item["kind"] == "dense_field" and conditions and conditions.keys() <= candidates:
            names = sorted(conditions)
            item.update(input_record=names[0], dynamic_input_storage=True)
            if len(names) > 1:
                item["input_conditions"] = names


def project_owned_records(environment, schema, *, borrowed=False, uses=()):
    """Project a complete optional child's numeric state through existing ports.

    A typed child with no implementation/layout alternatives is a nullable
    namespace, not another numerical algorithm. Keep its Host construction and
    native layouts; use the existing record-presence and Dense argument ABI.
    Borrowed queries run after entry-specific static classification, so unused
    producer configuration and prepared numeric predicates need no presence
    variants. Their complete Field views keep the source owner's native layout.
    """
    from taichi_forge.types import primitive_types
    from ._component import Array, DenseField, _ConditionalDenseField, _InputPresence, _InputRecord
    from taichi_forge_runtime._dimensions import extent_dependencies

    declarations = dict(environment.declarations)
    query_paths = [tuple(item.get("path", (key,))) for key, item in schema.items()
                   if item["kind"] == "owner_input"] if borrowed else []
    for name, record in schema.items():
        conditions = record.get("conditions", {})
        path = tuple(record.get("path", (name,)))
        # A query can contain an optional numeric view of externally owned
        # buffers. Its namespace is owned, but none of its storage is allocated
        # here. Keep outer construction guards; project only local presence.
        nested_view = borrowed and record["kind"] == "owner" and any(
            len(path) > len(prefix) and path[:len(prefix)] == prefix for prefix in query_paths)
        if (record["kind"] != ("owner_input" if borrowed else "owner") and not nested_view
                or record.get("owner_alternatives") or record.get("dynamic_input_record") or not conditions):
            continue
        local = [(key, value) for key, value in conditions.items()
                 if not nested_view or environment._paths[key][:-1] == path[:-1]]
        if len(local) != 1:
            continue
        flag, expected = local[0]
        parents = {key: value for key, value in conditions.items() if key != flag}
        predicate = schema[flag]
        if (parents and not nested_view or expected is not True or predicate["kind"] != "static"
                or predicate.get("conditions", {}) != parents
                or any(schema[key]["kind"] != "static" or schema[key].get("dynamic_storage_condition")
                       or schema[key].get("prepare_static") for key in parents)
                or {tuple(v) for v in predicate["choices"]} != {("bool", False), ("bool", True)}):
            continue
        members = {key: item for key, item in schema.items()
                   if len(item.get("path", (key,))) > len(path)
                   and tuple(item.get("path", (key,)))[:len(path)] == path}

        def numeric_member(key, item):
            if borrowed and any(item.get(part) for part in (
                    "owner_alternatives", "choice_guards", "shape_guards", "dtype_guards", "source_guards",
                    "shape_policy", "dtype_policy", "source_policy")):
                return False
            if item["kind"] in ("dimension", "value", "dense_field"):
                return True
            if not borrowed:
                return False
            return (item["kind"] == "record" and not item.get("optional")
                    or item["kind"] == "ndarray" and item.get("external") and not item.get("optional_input")
                    or item["kind"] == "static" and (
                        item.get("prepare_static") or environment._paths[key][-1] not in uses))

        if not members or any(not numeric_member(key, item)
                              or item.get("conditions") != conditions for key, item in members.items()):
            continue
        storage = {key: item for key, item in members.items() if item["kind"] == "dense_field"}
        external_view = nested_view and any(item["kind"] == "ndarray" for item in members.values()) and all(
            item["kind"] == "dimension" or item["kind"] == "ndarray" and item.get("external")
            and not item.get("optional_input") for item in members.values())
        groups = {declarations[key].group for key in storage if isinstance(declarations[key], DenseField)}
        axes = {key for key, item in members.items() if item["kind"] == "dimension"}
        if (not storage and not external_view or nested_view and not external_view
                or any(item.get("struct_field") or item["element"].get("model_tensor")
                               or not borrowed and not item.get("group_builder") for item in storage.values())
                or any(isinstance(value, DenseField) and value.group in groups and key not in storage
                       for key, value in declarations.items())
                or any(flag in item.get("conditions", {}) and key not in members and key != name
                       for key, item in schema.items())
                or any(flag in item.get(guards, {}) for item in schema.values()
                       for guards in ("choice_guards", "shape_guards", "dtype_guards", "source_guards"))
                or any(key not in members and (
                    axes.intersection(item.get("shape", ()))
                    or axes.intersection(extent_dependencies(item["expression"]) if item.get("expression") else ())
                    or borrowed and any(axes.intersection(extent_dependencies(axis))
                                        for axis in item.get("shape_expressions", ()))
                    or tuple(item.get("source_path", ()))[:len(path)] == path)
                       for key, item in schema.items())):
            continue
        # Entry-specific borrowed views share the original schema dictionaries.
        # Publish new metadata only in this projection, never in another entry.
        if borrowed:
            record = schema[name] = dict(record)
            predicate = schema[flag] = dict(predicate)
            # Presence is fixed by the normal preparation binding. Reuse the
            # existing finite scalar contract so consumption folds the absent
            # query before device generation, and rebinding invalidates it.
            predicate["prepare_static"] = True
        record.update(dynamic_input_record=True, input_presence_attribute=flag)
        predicate.update(dynamic_storage_condition=True, input_presence_attribute=flag)
        declarations[name] = _InputRecord(flag)
        declarations[flag] = _InputPresence(primitive_types.i32)
        for key, item in members.items():
            if borrowed:
                item = schema[key] = dict(item)
            item["input_record"] = name
            if key in storage:
                value = declarations[key]
                declarations[key] = _ConditionalDenseField(value.dtype, value.shape, value.group,
                                                           ((flag, True),))
                item["dynamic_input_storage"] = True
                if borrowed:
                    item["shape"] = list(value.shape)
            elif borrowed and item["kind"] == "ndarray":
                # Presence is recorded by the enclosing input namespace. Use
                # the same nullable external descriptor and lease as records;
                # it is not an allocation owned by the query's caller.
                value = declarations[key]
                declarations[key] = Array(value.dtype, value.ndim, value.shape, value.resource_id)
    environment._set_declarations(declarations)


def project_query_groups(environment, schema, uses):
    """Bind optional query fields independently of their producer's tree.

    The source owner still constructs and owns the original complete layout.
    Query captures carry actual Dense views, so fields sharing that tree need
    not share presence. Conditional extents retain the same admission guard.
    """
    from taichi_forge.types import primitive_types
    from taichi_forge_runtime._dimensions import extent_dependencies
    from ._component import DenseField, _ConditionalDenseField, _ConditionalDimension, _InputPresence

    declarations = dict(environment.declarations)
    inputs = [tuple(item.get("path", (key,))) for key, item in schema.items() if item["kind"] == "owner_input"]
    for flag, predicate in list(schema.items()):
        if predicate["kind"] != "static":
            continue  # Host sequences have no numerical capture path.
        path = environment._paths[flag]
        if (predicate.get("dynamic_storage_condition")
                or not any(path[:len(prefix)] == prefix for prefix in inputs)
                or {tuple(v) for v in predicate["choices"]} != {("bool", False), ("bool", True)}
                or uses.get(path[-1], {"prepare_condition"}) != {"prepare_condition"}):
            continue
        parents = predicate.get("conditions", {})
        if any(schema[key]["kind"] != "static" or schema[key].get("dynamic_storage_condition") for key in parents):
            continue
        conditions = {**parents, flag: True}
        members = {key: item for key, item in schema.items() if flag in item.get("conditions", {})}
        storage = {key: item for key, item in members.items() if item["kind"] == "dense_field"}
        axes = {key for key, item in members.items() if item["kind"] == "dimension"}
        if (not storage or any(item["kind"] not in ("dimension", "dense_field")
                               or item.get("conditions") != conditions for item in members.values())
                or any(item.get("struct_field") or item["element"].get("model_tensor")
                       or not isinstance(declarations[key], DenseField) for key, item in storage.items())
                or any(flag in item.get(guards, {}) for item in schema.values()
                       for guards in ("choice_guards", "shape_guards", "dtype_guards", "source_guards"))
                or any(key not in members and (
                    axes.intersection(item.get("shape", ()))
                    or axes.intersection(extent_dependencies(item["expression"]) if item.get("expression") else ())
                    or any(axes.intersection(extent_dependencies(axis)) for axis in item.get("shape_expressions", ())))
                       for key, item in schema.items())):
            continue
        schema[flag] = {**predicate, "dynamic_storage_condition": True,
                        "input_presence_attribute": flag, "prepare_static": True}
        declarations[flag] = _InputPresence(primitive_types.i32)
        for key, original in members.items():
            item = schema[key] = {**original, "input_record": flag}
            value = declarations[key]
            if key in storage:
                declarations[key] = _ConditionalDenseField(value.dtype, value.shape, value.group, ((flag, True),))
                item.update(dynamic_input_storage=True, shape=list(value.shape))
            else:
                declarations[key] = _ConditionalDimension(value.kind, value.capacity, value.expression, ((flag, True),))
    environment._set_declarations(declarations)


def ordered_presence_conditions(schema, paths, conditions):
    """Order declared guard dependencies before any conditional member reads."""
    pending = sorted(conditions, key=lambda name: (len(paths[name]), paths[name]))
    ordered, done = [], set()
    while pending:
        ready = [name for name in pending if not (set(schema[name].get('conditions', {})) & set(pending))]
        if not ready:
            raise ValueError('Nested query presence conditions contain a cycle')
        ordered.extend(ready)
        done.update(ready)
        pending = [name for name in pending if name not in done]
    return ordered


def project_nested_query_records(environment, schema, uses, *, query_paths=None):
    """Project a joint presence closure, without changing the source layout.

    Dynamic guards qualify together inside their finite outer layout. The
    invocation keeps their ordered predicates so absent parents never expose
    child data. A producer-selected ancestor need not itself become dynamic.
    """
    from dataclasses import replace
    from taichi_forge.types import primitive_types
    from ._component import (Array, DenseField, Dimension, _ConditionalDenseField, _ConditionalRecord,
                             _ConditionalDimension, _InputPresence, _InputRecord, _OptionalArray, _OwnerReference)

    paths = environment._paths
    inputs = [paths[key] for key, item in schema.items() if item['kind'] == 'owner_input']
    # This is a query projection, not the owner's allocation schema. Unread
    # construction members must not pin a query's presence to production
    # variants. Include extent dependencies of every possibly read member;
    # computed reflection keeps the conservative complete-schema behavior.
    from taichi_forge_runtime._dimensions import extent_dependencies
    # Host sequences own independent entry instances, not a device path in
    # this query. Keep their Host contract while projecting only device keys.
    device_keys = {key for key, item in schema.items() if item['kind'] != 'host_owner_sequence'}
    required = device_keys.copy() if '*' in uses else {
        key for key in device_keys if paths[key][-1] in uses}
    if query_paths is not None:
        reads, unknown = query_paths
        # A resolved method can distinguish identically named members on
        # different receivers. For unresolved subtrees retain the existing
        # conservative closure (including computed-name patterns/reflection),
        # rather than treating the missing method proof as an empty readset.
        required = {key for key in device_keys
                    if any(path[:len(paths[key])] == tuple(paths[key]) for path in reads)
                    or key in required and any(tuple(paths[key][:len(prefix)]) == prefix for prefix in unknown)}
    pending = list(required)
    while pending:
        item = schema[pending.pop()]
        dependencies = set(item.get('conditions', {}))
        for part in ('choice_guards', 'shape_guards', 'dtype_guards', 'source_guards'):
            dependencies.update(item.get(part, {}))
        for part in ('input_presence_attribute', 'input_record'):
            if item.get(part):
                dependencies.add(item[part])
        dependencies.update(item.get('input_conditions', ()))
        for part in ('expression', 'shape_expressions'):
            expressions = [item[part]] if part == 'expression' and part in item else item.get(part, ())
            for expression in expressions:
                dependencies.update(extent_dependencies(expression))
        dependencies.update(axis for axis in item.get('shape', ()) if isinstance(axis, str))
        for key in (dependencies & device_keys) - required:
            required.add(key)
            pending.append(key)
    candidates = {key for key, item in schema.items()
                  if key in required and item['kind'] == 'static'
                  and {tuple(value) for value in item.get('choices', ())} == {('bool', False), ('bool', True)}
                  # Existing nullable query schedules retain their guarded
                  # parallel lowering. Type/alias/early-return uses do not.
                  and uses.get(paths[key][-1], {'prepare_condition'}) <= {'prepare_condition', 'parallel_structure'}
                  and (any(len(paths[key]) > len(prefix) and paths[key][:len(prefix)] == prefix for prefix in inputs)
                       or any(item['kind'] == 'owner_input' and key in item.get('conditions', {})
                              for item in schema.values()))
                  and any(key in member.get('conditions', {}) for member in schema.values())}

    def numeric(key, item):
        if any(item.get(part) for part in ('owner_alternatives', 'choice_guards', 'shape_guards',
                                          'dtype_guards', 'source_guards', 'shape_policy', 'dtype_policy',
                                          'source_policy')):
            return False
        kind = item['kind']
        if kind in ('owner', 'owner_input', 'value', 'dimension', 'constant'):
            return True
        if kind == 'record':
            return not item.get('struct_field') and (not item.get('optional') or item.get('dynamic_input_record'))
        if kind == 'dense_field':
            return (not item.get('struct_field') and not item['element'].get('model_tensor')
                    and isinstance(environment.declarations[key], DenseField))
        if kind == 'ndarray':
            return item.get('external', False)
        return kind == 'static' and (key in candidates or item.get('prepare_static')
                                     or len(item['choices']) == 1 or paths[key][-1] not in uses)

    while True:
        accepted = set()
        for flag in candidates:
            # An outer owner may have a real dtype/shape alternative. Its
            # already-selected branch must not pin independent inner resource
            # presence. Only dominating static ancestors can remain outside
            # this joint dynamic closure; sibling/inner guards still qualify
            # together, and no negative dynamic predicate is admitted.
            parents = schema[flag].get('conditions', {})
            fixed_parents = {key for key in parents if key not in candidates
                             and schema[key]['kind'] == 'static'
                             and not schema[key].get('prepare_static')
                             and not schema[key].get('dynamic_storage_condition')}
            members = [(key, item) for key, item in schema.items()
                       if key in required and flag in item.get('conditions', {})]
            if (any(not numeric(key, item)
                    or not item.get('conditions', {}).keys() <= candidates | fixed_parents
                    or any(item.get('conditions', {}).get(key) is not expected
                           for key, expected in parents.items() if key in fixed_parents)
                    or any(value is not True for key, value in item.get('conditions', {}).items()
                           if key in candidates)
                    for key, item in members)
                    or any(flag in item.get(part, {}) for key, item in schema.items() if key in required
                           for part in ('choice_guards', 'shape_guards', 'dtype_guards', 'source_guards'))):
                continue
            accepted.add(flag)
        if accepted == candidates:
            break
        candidates = accepted
    # A single inner guard can remain after a finite outer layout was retained.
    # It uses the same admission contract as the jointly dynamic nested case.
    if not candidates or all(schema[key].get('dynamic_storage_condition') for key in candidates):
        return
    records = {key: item for key, item in schema.items()
               if key in required and item['kind'] in ('owner', 'owner_input', 'record') and item.get('conditions')
               and any(flag in candidates and paths[flag][:-1] == paths[key][:-1]
                       for flag in item['conditions'])}
    declarations = dict(environment.declarations)
    for flag in candidates:
        schema[flag] = {**schema[flag], 'dynamic_storage_condition': True,
                        'input_presence_attribute': flag, 'prepare_static': True, 'nested_query_input': True}
        declarations[flag] = _InputPresence(primitive_types.i32)
    for key, original in list(schema.items()):
        if key not in required:
            continue
        conditions = {flag: expected for flag, expected in original.get('conditions', {}).items()
                      if flag in candidates}
        if not conditions:
            continue
        guards = ordered_presence_conditions(schema, paths, conditions)
        item = schema[key] = {**original, 'input_conditions': guards, 'nested_query_input': True}
        ancestors = [name for name in records if len(paths[name]) < len(paths[key])
                     and paths[key][:len(paths[name])] == paths[name]]
        item['input_record'] = max(ancestors, key=lambda name: len(paths[name])) if ancestors else guards[0]
        if key in records:
            flag = next(flag for flag in reversed(guards) if paths[flag][:-1] == paths[key][:-1])
            item.update(dynamic_input_record=True, input_presence_attribute=flag)
            declarations[key] = _InputRecord(flag)
        elif item['kind'] in ('owner', 'owner_input', 'record'):
            # An unconditional child inherits its parent's guard; it must not
            # replace that guard's canonical optional namespace or become optional
            # itself. Preserve aliases and check its actual type when reachable.
            item.pop('dynamic_input_record', None)
            item.pop('input_presence_attribute', None)
            if not isinstance(declarations[key], _OwnerReference):
                declarations[key] = _ConditionalRecord(tuple((flag, True) for flag in guards))
        elif isinstance(declarations[key], DenseField):
            value = declarations[key]
            declarations[key] = _ConditionalDenseField(value.dtype, value.shape, value.group,
                                                       tuple((flag, True) for flag in guards))
            item.update(dynamic_input_storage=True, shape=list(value.shape))
        elif isinstance(declarations[key], Dimension):
            value = declarations[key]
            declarations[key] = _ConditionalDimension(value.kind, value.capacity, value.expression,
                                                       tuple((flag, True) for flag in guards))
        elif isinstance(declarations[key], Array):
            value = declarations[key]
            declarations[key] = (replace(value, conditions=()) if isinstance(value, _OptionalArray)
                                 else Array(value.dtype, value.ndim, value.shape, value.resource_id))
    environment._set_declarations(declarations)


def project_struct_member_inputs(environment, schema):
    """Use existing Dense member descriptors for a whole optional StructField.

    This is compilation metadata only: the Host still owns one original group
    and record. No new allocation, aggregate pointer ABI or layout is invented.
    """
    from ._component import _capture_id

    for name, item in list(schema.items()):
        if not item.get("dynamic_input_storage"):
            continue
        if not item.get("struct_field"):
            item["shape"] = list(environment.declarations[name].shape)
            continue
        path = tuple(item.get("path", (name,)))
        for member, element in item["element"]["members"].items():
            key = _capture_id((*path, member))
            schema[key] = {
                **{k: v for k, v in item.items() if k != "struct_field"},
                "path": [*path, member],
                "element": element,
                "shape": list(environment.declarations[key].shape),
            }
