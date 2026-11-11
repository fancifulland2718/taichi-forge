"""Re-encode component metadata in a frozen AOT archive without compiling code.

This producer utility changes metadata storage only. It does not combine
modules, rename code, change execution semantics or discover dependencies.
"""
import argparse
from copy import copy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from taichi_forge_runtime.aot._component_storage import (
    decode_contract_document, encode_contract_documents, decode_plan_document, encode_plan_documents,
    decode_entry_document, encode_entry_documents,
)


def share_component_contracts(source, destination, *, shared_execution_plans=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Use a new destination; the frozen archive must be retained")
    original = source.read_bytes()
    with ZipFile(BytesIO(original)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Component archive has duplicate member names")
        roots = {name: decode_contract_document(json.loads(archive.read(name)), archive.read)
                 for name in names if name.startswith("components/") and name.endswith(".json")}
        documents = encode_contract_documents(roots)
        # Verify exact expanded values before producing a release input. Neither
        # per-component identity nor its interpretation is changed by this tool.
        for name, contract in roots.items():
            if decode_contract_document(json.loads(documents[name]), documents.__getitem__) != contract:
                raise ValueError("Component storage roundtrip changed the original contract")
        entries = {name: decode_entry_document(json.loads(archive.read(name)), archive.read)
                   for name in names if name.startswith("component_entries/") and name.endswith(".json")}
        entry_documents = encode_entry_documents(entries)
        for name, contract in entries.items():
            if decode_entry_document(json.loads(entry_documents[name]), entry_documents.__getitem__) != contract:
                raise ValueError("Entry storage roundtrip changed the original contract")
        documents.update(entry_documents)
        plans = {}
        if shared_execution_plans:
            from taichi_forge_runtime.aot._plan import ExecutionPlan, _unique_fields

            plans = {name: decode_plan_document(json.loads(archive.read(name), object_pairs_hook=_unique_fields),
                                               archive.read)
                     for name in names if name.startswith("executions/") and name.endswith(".json")}
            plan_documents = encode_plan_documents(plans)
            for name, plan in plans.items():
                restored = decode_plan_document(json.loads(plan_documents[name]), plan_documents.__getitem__)
                if restored != plan:
                    raise ValueError("Plan storage roundtrip changed the original plan")
                # Preserve admission: storage never bypasses the ordinary parser.
                ExecutionPlan.from_json(json.dumps(restored))
            documents.update(plan_documents)
        output = BytesIO()
        unchanged, compressed_delta = {}, 0
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=1) as target:
            target.comment = archive.comment
            for item in archive.infolist():
                if (item.filename in roots or item.filename in entries
                        or item.filename.startswith("component_contracts/")
                        or shared_execution_plans and (item.filename in plans
                                                       or item.filename.startswith("execution_records/"))):
                    continue
                payload = archive.read(item)
                target.writestr(copy(item), payload, compresslevel=1)
                unchanged[item.filename] = sha256(payload).hexdigest()
                compressed_delta += target.getinfo(item.filename).compress_size - item.compress_size
            for name, payload in sorted(documents.items()):
                info = ZipInfo(name, date_time=(2000, 12, 1, 0, 0, 0))
                target.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=1)
    candidate = output.getvalue()
    # Account for the whole container as well as the record-level cost estimate.
    chosen = candidate if len(candidate) < len(original) else original
    if chosen is original:
        compressed_delta = 0
    with destination.open("xb") as stream:
        stream.write(chosen)
    return dict(source=str(source.resolve()), source_sha256=sha256(original).hexdigest(),
                destination=str(destination.resolve()), sha256=sha256(chosen).hexdigest(),
                before_bytes=len(original), after_bytes=len(chosen), roots=len(roots), entries=len(entries),
                changed=chosen is not original, noncontract_payload_sha256=unchanged,
                noncontract_compressed_byte_delta=compressed_delta,
                shared_execution_plans=shared_execution_plans, execution_plan_count=len(plans))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--report")
    parser.add_argument("--shared-execution-plans", action="store_true")
    args = parser.parse_args()
    result = share_component_contracts(args.source, args.destination, shared_execution_plans=args.shared_execution_plans)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "noncontract_payload_sha256"}))
