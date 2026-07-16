from taichi_forge import _version_check


def test_version_check_thread_starts_at_most_once(monkeypatch):
    starts = []

    class StubThread:
        def __init__(self, *, target, daemon):
            assert target is _version_check.try_check_version
            assert daemon

        def start(self):
            starts.append(True)

    monkeypatch.setattr(_version_check, "_version_check_started", False)
    monkeypatch.setattr(_version_check.threading, "Thread", StubThread)
    monkeypatch.setenv("TI_SKIP_VERSION_CHECK", "ON")
    _version_check.start_version_check_thread()
    assert not starts

    monkeypatch.delenv("TI_SKIP_VERSION_CHECK")
    _version_check.start_version_check_thread()
    _version_check.start_version_check_thread()
    assert len(starts) == 1
