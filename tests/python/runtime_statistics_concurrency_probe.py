import faulthandler
import sys
import threading
import time

import numpy as np
import taichi as ti

faulthandler.dump_traceback_later(4, repeat=False)
ti.init(arch=getattr(ti, sys.argv[1]), offline_cache=False)
keys = ti.field(ti.i32, shape=4096)
values = ti.field(ti.i32, shape=4096)
keys.from_numpy(np.arange(4096, dtype=np.int32)[::-1].copy())
values.from_numpy(np.arange(4096, dtype=np.int32))
ti.algorithms.sort(keys, values, stable=True)
ti.sync()
ready = threading.Event()
errors = []

def worker():
    try:
        transaction = ti.lang.impl.get_runtime().prog._begin_runtime_submission_transaction()
        ready.set()
        time.sleep(0.15)
        ti.algorithms.sort(keys, values, stable=True)
        transaction._finish().wait()
    except BaseException as exc:
        errors.append(exc)

thread = threading.Thread(target=worker)
thread.start()
assert ready.wait(2)
print('STATS_START', flush=True)
snapshot = ti.runtime.stats()
print('STATS_DONE', snapshot.submission.native_submissions, flush=True)
thread.join(3)
assert not thread.is_alive()
assert not errors, errors
np.testing.assert_array_equal(keys.to_numpy(), np.arange(4096))
faulthandler.cancel_dump_traceback_later()
print('PASS', flush=True)
