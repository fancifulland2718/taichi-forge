"""Re-encode component metadata in a frozen AOT archive without compiling code.

This producer utility changes only component contract storage. It does not
combine modules, rename code, change execution plans or discover dependencies.
"""
import argparse
from copy import copy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from taichi_forge_runtime.aot._component_storage import decode_contract_document, encode_contract_documents


def share_component_contracts(source, destination):
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
        output = BytesIO()
        unchanged, compressed_delta = {}, 0
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=1) as target:
            target.comment = archive.comment
            for item in archive.infolist():
                if item.filename in roots or item.filename.startswith("component_contracts/"):
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
                before_bytes=len(original), after_bytes=len(chosen), roots=len(roots),
                changed=chosen is not original, noncontract_payload_sha256=unchanged,
                noncontract_compressed_byte_delta=compressed_delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = share_component_contracts(args.source, args.destination)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "noncontract_payload_sha256"}))
