"""Public CUDA preparation owns one runtime view across all graph phases."""

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_sparse_retirement import _sparse_module
from tests.python.test_vulkan_sparse_execution_domain import export_domain_cases


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_cuda_sparse_domain_control_and_retirement(tmp_path):
    from taichi_forge_runtime import runtime_contract_manifest
    from taichi_forge._lib import core
    feature = "cuda_execution_domain_sparse_fields"
    assert runtime_contract_manifest()["features"][feature]
    assert core.get_runtime_contract_manifest()["features"][feature]
    _sparse_module(tmp_path, "ptx", export_extra=export_domain_cases,
                   consumer="aot_sparse_domain_probe.py")


@test_utils.test(arch=ti.cuda, debug=True, offline_cache=False)
def test_cuda_domain_result_error_reuse_and_rootless_owner():
    @ti.kernel
    def check(value: ti.u64, output: ti.types.ndarray(ti.u64, 1)):
        assert value > 0, "domain value must be positive"
        output[0] = value

    arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.u64)
    out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u64, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(check, arg, out)
    graph = builder.compile()
    array = ti.ndarray(ti.u64, shape=1)
    with ti.aot.borrow_runtime() as runtime:
        domain = runtime.execution_domain(capacity=1, instance_capacity=1)
        call = graph.prepare(dict(value=0, out=array), execution_domain=domain,
                             mutable_parameters=("value",))
        call.initialize()
        bad = call.submit(values=dict(value=0))
        with pytest.raises(RuntimeError, match="positive"):
            bad.wait()
        good = call.submit(values=dict(value=(1 << 40) + 7))
        good.wait()
        assert array.to_numpy().tolist() == [(1 << 40) + 7]
        with pytest.raises(RuntimeError, match="positive"):
            bad.wait()
        last = call.submit(values=dict(value=0))
        with pytest.raises(RuntimeError, match="positive"):
            call.close()
        assert call._prepared._handle is None
        with pytest.raises(RuntimeError, match="positive"):
            last.wait()
        replacement = graph.prepare(dict(value=11, out=array), execution_domain=domain)
        replacement.initialize()
        replacement.submit().wait()
        assert array.to_numpy().tolist() == [11]
        replacement.close()
        check(9, array)
        ti.sync()
        assert array.to_numpy().tolist() == [9]
        domain.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_sparse_instances_share_dense_directory_without_sharing_managers():
    weight = ti.field(ti.i32, shape=())
    weight[None] = 3
    fields = [ti.field(ti.i32) for _ in range(2)]
    for field in fields:
        builder = ti.FieldsBuilder()
        builder.pointer(ti.i, 4).place(field)
        builder.finalize()

    def make_advance(field):
        @ti.kernel
        def advance(out: ti.types.ndarray(ti.i32, 1)):
            field[1] += weight[None]
            out[0] = field[1]
        return advance

    kernels = [make_advance(field) for field in fields]
    graphs = []
    outputs = [ti.ndarray(ti.i32, shape=1) for _ in fields]
    for advance, output in zip(kernels, outputs):
        graph = ti.graph.GraphBuilder()
        graph.dispatch(advance,
                       ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
        graphs.append(graph.compile())
        advance(output)
    ti.sync()
    with ti.aot.borrow_runtime() as runtime:
        domains = [runtime.execution_domain(capacity=2, instance_capacity=1) for _ in fields]
        calls = [graph.prepare(dict(out=output), execution_domain=domain)
                 for graph, output, domain in zip(graphs, outputs, domains)]
        for call in calls:
            call.initialize()
        tickets = [call.submit() for call in calls]
        for ticket in reversed(tickets):
            ticket.wait()
        assert [out.to_numpy().tolist() for out in outputs] == [[6], [6]]
        # Dense value access retains the existing caller-ordered Field contract;
        # only the directory lifetime is shared by the two sparse owners.
        assert weight[None] == 3
        calls[0].close()
        kernels[0](outputs[0])
        calls[1].submit().wait()
        ti.sync()
        assert [out.to_numpy().tolist() for out in outputs] == [[9], [9]]
        calls[1].close()
        for domain in domains:
            domain.close()
