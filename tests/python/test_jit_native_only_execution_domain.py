"""Native-only selected recipes need an owner, not a fabricated kernel."""

import numpy as np
import taichi_forge as ti

from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_native_only_sort_uses_live_storage_and_private_execution():
    keys, values = ti.ndarray(ti.i32, 4), ti.ndarray(ti.i32, 4)
    with ti.algorithms.prepare_sort(keys, values) as sort:
        builder = ti.graph.GraphBuilder()
        builder.append_native(sort.record(persistent=True))
        graph = builder.compile()
        with ti.graph.execution_domain(capacity=1, instance_capacity=1) as domain:
            with graph.prepare({}, execution_domain=domain) as call:
                call.initialize()
                graph.close()
                sort.close()
                for source in ([5, 1, 9, 3], [2, 8, 4, 0]):
                    data = np.array(source, np.int32)
                    keys.from_numpy(data)
                    values.from_numpy(data * 10)
                    with call.submit() as ticket:
                        ticket.wait()
                    np.testing.assert_array_equal(keys.to_numpy(), np.sort(data))
                    np.testing.assert_array_equal(values.to_numpy(), np.sort(data) * 10)
