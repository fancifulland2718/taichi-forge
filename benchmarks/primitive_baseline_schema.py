import argparse
import csv
import json
import sys
from pathlib import Path


BASELINE_SCHEMA_VERSION = 1

BASELINE_COLUMNS = [
    "schema_version",
    "summary_path",
    "case",
    "primitive",
    "arch",
    "dtype",
    "n",
    "storage",
    "method",
    "correct",
    "median_us",
    "p95_us",
    "api_return_median_us",
    "repeats",
    "warmup",
    "workspace_bytes_peak",
    "compile_elapsed_us",
    "compile_kernel_calls",
    "compile_kernel_total_s",
    "cache_files",
    "cache_bytes",
    "cache_path",
    "compile_csv",
    "compile_trace",
]

_KNOWN_PRIMITIVES = (
    "bucket",
    "compact",
    "grouped_reduce",
    "histogram",
    "indexed_copy",
    "reduce",
    "scan",
    "scatter_add",
    "sort",
    "transform",
)


def _read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _infer_primitive(summary, case_name):
    if case_name in _KNOWN_PRIMITIVES:
        return case_name
    for primitive in _KNOWN_PRIMITIVES:
        if case_name.startswith(primitive):
            return primitive
    for primitive in _KNOWN_PRIMITIVES:
        if f"{primitive}_method" in summary or f"{primitive}_storage" in summary:
            return primitive
    return case_name


def _method_for(summary, primitive):
    return _first_present(summary, [f"{primitive}_method", "method"])


def _storage_for(summary, primitive):
    return _first_present(summary, [f"{primitive}_storage", "storage"])


def _compile_kernel_stats(summary):
    calls = 0
    total_s = 0.0
    for item in summary.get("compile", {}).get("top", []):
        path = item.get("path", "")
        if path.endswith("taichi::lang::Program::compile_kernel"):
            calls += int(item.get("calls", 0))
            total_s += float(item.get("total_s", 0.0))
    return calls, total_s


def _case_rows(summary_path, summary):
    compile_info = summary.get("compile", {})
    cache_info = summary.get("cache", {})
    compile_kernel_calls, compile_kernel_total_s = _compile_kernel_stats(summary)
    case_results = summary.get("case_results", {})
    case_names = summary.get("cases") or sorted(case_results)

    rows = []
    for case_name in case_names:
        result = case_results.get(case_name, {})
        timing = result.get("timing", {})
        memory = result.get("memory", {})
        primitive = _infer_primitive(summary, case_name)
        rows.append(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "summary_path": str(summary_path),
                "case": case_name,
                "primitive": primitive,
                "arch": summary.get("arch"),
                "dtype": summary.get("dtype"),
                "n": summary.get("n"),
                "storage": _storage_for(summary, primitive),
                "method": _method_for(summary, primitive),
                "correct": result.get("correct"),
                "median_us": timing.get("median_us"),
                "p95_us": timing.get("p95_us"),
                "api_return_median_us": timing.get("api_return_median_us"),
                "repeats": timing.get("repeats"),
                "warmup": timing.get("warmup"),
                "workspace_bytes_peak": memory.get(
                    "workspace_bytes_peak", summary.get("workspace_bytes_peak_sum")
                ),
                "compile_elapsed_us": compile_info.get("elapsed_us"),
                "compile_kernel_calls": compile_kernel_calls,
                "compile_kernel_total_s": compile_kernel_total_s,
                "cache_files": cache_info.get("files"),
                "cache_bytes": cache_info.get("bytes"),
                "cache_path": cache_info.get("path"),
                "compile_csv": compile_info.get("csv"),
                "compile_trace": compile_info.get("chrome_trace"),
            }
        )
    return rows


def normalize_summary(summary_path):
    path = Path(summary_path)
    return _case_rows(path, _read_json(path))


def iter_summary_paths(inputs):
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            yield from sorted(path.rglob("summary*.json"))
        else:
            yield path


def collect_rows(inputs):
    rows = []
    for path in iter_summary_paths(inputs):
        rows.extend(normalize_summary(path))
    return rows


def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASELINE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Normalize primitive benchmark summaries into the S1 "
            "compile/cache/workspace baseline schema."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Summary JSON files or directories containing summary*.json files.",
    )
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    rows = collect_rows(args.inputs)
    if args.csv_path:
        write_csv(rows, args.csv_path)
    json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
