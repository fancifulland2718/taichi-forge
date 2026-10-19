"""Runtime-only consumer of selected segmented reduction helpers."""

import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_dot_probe import raises
from tests.python.aot_segmented_scan_probe import inputs, normalize, read, write


def expected(offsets, values, dtype):
    return [normalize(sum(values[begin:end]), dtype) for begin, end in zip(offsets, offsets[1:])]


def main(mode, artifact):
    info = json.loads(Path(artifact).with_suffix('.json').read_text(encoding='utf8'))
    if mode == 'borrowed':
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime('cuda')
    calls = []
    with runtime:
        module = runtime.load_module(artifact)
        dtype, capacity, offsets = info['dtype'], info['capacity'], info['offsets']
        segments = len(offsets) - 1
        for strategy in info['strategies']:
            frames = []
            for _ in range(2):
                args = dict(input=runtime.ndarray(dtype, capacity), result=runtime.ndarray(dtype, segments))
                write(args['result'], [73] * segments, dtype)
                call = module.prepare_execution(strategy, args)
                assert read(args['result'], dtype) == [73] * segments
                frames.append((args, call))
                calls.append(call)
            for iteration in (1, 0, -1):
                for index, (args, call) in enumerate(frames):
                    host = inputs(capacity, dtype, iteration + index)
                    write(args['input'], host, dtype)
                    call()
                    assert read(args['result'], dtype) == expected(offsets, host, dtype)
                    assert read(args['input'], dtype) == host
            for invalid in (
                dict(input=frames[0][0]['input'], result=frames[0][0]['input']),
                dict(input=runtime.ndarray(dtype, capacity - 1), result=frames[0][0]['result']),
                dict(input=frames[0][0]['input'], result=runtime.ndarray('f32', segments)),
            ):
                raises(lambda: module.prepare_execution(strategy, invalid))
            frames[0][1].close()
            frames[1][1]()
        calls[-1]()
        if mode == 'borrowed':
            ti.reset()
        else:
            module.close()
        raises(calls[-1])
        for call in calls:
            call.close()
    if mode == 'standalone':
        assert 'taichi_forge' not in sys.modules
        assert 'numpy' not in sys.modules


if __name__ == '__main__':
    main(*sys.argv[1:])
