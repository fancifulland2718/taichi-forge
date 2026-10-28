"""Completed artifacts survive a process killed without finalizing its index."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils


def child(cache_dir, backend, mode, ready):
    ti.init(arch=getattr(ti, backend), offline_cache=True,
            offline_cache_file_path=cache_dir, enable_fallback=False)
    program = ti.lang.impl.get_runtime().prog
    program._set_kernel_executable_lifecycle_telemetry_enabled(True)

    @ti.kernel
    def calculate(x: ti.i32) -> ti.i32:
        return x * 3 + 7

    assert calculate(11) == 40
    stats = program._debug_kernel_executable_lifecycle_stats(False)
    temporary = Path(ready).with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(stats)))
    temporary.replace(ready)
    if mode == 'wait':
        while True:
            time.sleep(1)
    if mode == 'reset':
        ti.reset()
    # Also bypass cleanup on the reader so recovery never relies on reset.
    os._exit(0)


@pytest.mark.parametrize('arch', [a for a in (ti.cpu, ti.cuda, ti.vulkan) if a in test_utils.expected_archs()])
@pytest.mark.parametrize('corrupt', [False, True])
def test_cache_survives_forced_termination(tmp_path, arch, corrupt):
    backend = 'cpu' if arch == ti.cpu else str(arch).split('.')[-1]
    cache = tmp_path / 'cache'
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    env = os.environ.copy()
    env['TI_OFFLINE_CACHE'] = '1'
    env['TI_INPROC_DISK_MIRROR_MB'] = '0'
    command = [sys.executable, __file__, str(cache), backend]
    with (tmp_path / 'child.log').open('w') as log:
        process = subprocess.Popen(command + ['wait', str(first)], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while not first.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert first.exists(), (tmp_path / 'child.log').read_text()
            assert json.loads(first.read_text())['compiler_invocations'] == 1
            artifacts = list(cache.glob('*.tic'))
            assert len(artifacts) == 1
            assert not list(cache.glob('*.tcb'))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    if corrupt:
        artifacts[0].write_bytes(b'truncated')
    subprocess.run(command + ['exit', str(second)], env=env, check=True, timeout=90,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stats = json.loads(second.read_text())
    assert stats['compiler_invocations'] == int(corrupt)
    assert stats['disk_loads'] == int(not corrupt)
    assert not list(cache.glob('*.tmp'))


def test_concurrent_cache_writers_and_index_recovery(tmp_path):
    cache = tmp_path / 'cache'
    command = [sys.executable, __file__, str(cache), 'cpu']
    env = os.environ.copy()
    env['TI_INPROC_DISK_MIRROR_MB'] = '0'
    workers = []
    try:
        for index in range(3):
            workers.append(subprocess.Popen(
                command + ['exit', str(tmp_path / f'writer{index}.json')], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for worker in workers:
            assert worker.wait(timeout=90) == 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
    result = tmp_path / 'reader.json'
    subprocess.run(command + ['reset', str(result)], env=env, check=True, timeout=90,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stats = json.loads(result.read_text())
    assert stats['compiler_invocations'] == 0
    assert stats['disk_loads'] == 1
    assert len(list(cache.glob('*.tic'))) == 1
    assert len(list(cache.glob('*.tcb'))) == 1
    assert not list(cache.glob('*.tmp'))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_effective_cache_path_and_disabled_cache(tmp_path, monkeypatch):
    env_path = str(tmp_path / 'environment')
    monkeypatch.setenv('TI_OFFLINE_CACHE_FILE_PATH', env_path)
    ti.init(arch=ti.cpu, offline_cache=False)
    assert ti.lang.impl.current_cfg().offline_cache_file_path == env_path

    explicit_path = str(tmp_path / 'explicit')
    ti.init(arch=ti.cpu, offline_cache=False, offline_cache_file_path=explicit_path)
    assert ti.lang.impl.current_cfg().offline_cache_file_path == explicit_path

    @ti.kernel
    def calculate() -> ti.i32:
        return 42

    assert calculate() == 42
    ti.reset()
    assert not Path(env_path).exists()
    assert not Path(explicit_path).exists()


if __name__ == '__main__':
    child(*sys.argv[1:])
