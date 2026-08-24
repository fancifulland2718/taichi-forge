import json

from tests.python import hardware_process_memory as memory


def test_process_gpu_memory_prefers_package_free_nvml(monkeypatch):
    monkeypatch.setattr(
        memory,
        "_nvml_process_gpu_bytes",
        lambda: (1234, "nvml_compute_graphics_process_v3", None),
    )

    def unexpected_smi():
        raise AssertionError("nvidia-smi fallback should not be called")

    monkeypatch.setattr(memory, "_nvidia_smi_process_gpu_bytes", unexpected_smi)

    assert memory._nvidia_process_gpu_bytes() == (
        1234,
        "nvml_compute_graphics_process_v3",
        None,
    )


def test_process_gpu_memory_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        memory,
        "_nvml_process_gpu_bytes",
        lambda: (None, None, "nvml_process_not_listed"),
    )
    monkeypatch.setattr(
        memory,
        "_nvidia_smi_process_gpu_bytes",
        lambda: (64 * 1024 * 1024, "nvidia-smi_compute_process", None),
    )

    assert memory._nvidia_process_gpu_bytes() == (
        64 * 1024 * 1024,
        "nvidia-smi_compute_process",
        None,
    )


def test_process_gpu_memory_reports_both_observer_failures(monkeypatch):
    monkeypatch.setattr(
        memory,
        "_nvml_process_gpu_bytes",
        lambda: (None, None, "nvml_process_memory_unavailable"),
    )
    monkeypatch.setattr(
        memory,
        "_nvidia_smi_process_gpu_bytes",
        lambda: (None, None, "nvidia-smi_process_memory_unavailable"),
    )

    value, source, reason = memory._nvidia_process_gpu_bytes()

    assert value is None
    assert source is None
    assert reason == (
        "nvml:nvml_process_memory_unavailable;"
        "nvidia-smi:nvidia-smi_process_memory_unavailable"
    )


def test_process_memory_plateau_requires_long_run_and_exact_gpu_process_scope(
    tmp_path, monkeypatch
):
    output = tmp_path / "memory.json"
    monkeypatch.setenv(memory.OUTPUT_ENV, str(output))
    rss = iter((100, 120, 124, 200, 220, 224))
    gpu = iter((10, 12, 12, 20, 22, 22))
    monkeypatch.setattr(memory, "_rss_bytes", lambda: (next(rss), "test_rss", None))
    monkeypatch.setattr(
        memory,
        "_nvidia_process_gpu_bytes",
        lambda: (next(gpu), "test_gpu_process", None),
    )

    short = memory.ProcessMemoryPlateau("short", ("cuda-cudss",))
    for phase in ("before", "midpoint", "after"):
        short.capture(phase)
    short_record = short.finish(16)

    qualified = memory.ProcessMemoryPlateau("qualified", ("cuda-cudss",))
    for phase in ("before", "midpoint", "after"):
        qualified.capture(phase)
    qualified_record = qualified.finish(10_000)

    assert not short_record["qualification"]["process_level_memory_qualified"]
    assert qualified_record["qualification"]["process_level_memory_qualified"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == memory.SCHEMA
    assert [record["case_id"] for record in payload["records"]] == [
        "short",
        "qualified",
    ]


def test_process_memory_plateau_fails_closed_without_gpu_process_measurement(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(memory.OUTPUT_ENV, str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "_rss_bytes", lambda: (100, "test_rss", None))
    monkeypatch.setattr(
        memory,
        "_nvidia_process_gpu_bytes",
        lambda: (None, None, "nvidia-smi_process_memory_unavailable"),
    )
    observer = memory.ProcessMemoryPlateau("wddm", ("cuda-cufft",))
    for phase in ("before", "midpoint", "after"):
        observer.capture(phase)

    record = observer.finish(10_000)

    assert record["qualification"]["rss_plateau"]
    assert not record["qualification"]["gpu_process_available"]
    assert not record["qualification"]["process_level_memory_qualified"]
