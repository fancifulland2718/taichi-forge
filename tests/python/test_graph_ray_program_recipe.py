"""Real native programs compose through opaque recipes, without replay metadata work."""

from contextlib import ExitStack
import json
import multiprocessing
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.map_fusion import GraphMapFusionRecipeProvider
from tests import test_utils


def _window(owners, *, scalar=17, drift=None):
    ray = ti.hardware.ray
    output, middle, result = [ti.ndarray(ti.u32, 8) for _ in range(3)]
    if ti.cfg.arch == ti.vulkan:
        from tests.python.test_hardware_vulkan_ray_program import _shader

        if not ray.is_program_available():
            pytest.skip("Vulkan RT pipeline unavailable")
        program = owners.enter_context(
            ray.VulkanRayTracingPipeline(raygen={"run": _shader("sbt_record.rgen", "raygen")})
        )
        recording = program.record(
            8,
            raygen=ray.VulkanSbtRecord("run", struct.pack("<I", scalar)),
            bindings={"output": ray.VulkanRayBinding(0, 0, access="read_write")},
        )
        launch = owners.enter_context(recording.prepare(dict(output=output))).initialize()
        expected = 6 * np.arange(8, dtype=np.uint32) + 2 * scalar + 2
        kind = "vulkan_ray_program"
    else:
        from tests.python.test_hardware_optix_program_public import (
            PARAMS,
            E,
            _module,
            _provider,
            _program,
            _record,
            _scene,
        )

        provider = _provider()
        owners.callback(provider.close)
        gas, scene = _scene(provider)
        owners.callback(gas.close)
        owners.callback(scene.close)
        if drift in ("code", "layout"):
            module = _module()
            parameters = PARAMS
            if drift == "code":
                code = module.code.replace(b"add.s32 \t%r3, %r1, %r4;", b"sub.s32 \t%r3, %r1, %r4;")
                assert code != module.code
                module = ray.PtxModule(code)
            else:
                parameters = ray.OptixParameterLayout(32, PARAMS.fields)
            program = provider.program(
                (module,),
                raygen={"render": E(0, "__raygen__render")},
                miss={"miss": E(0, "__miss__value")},
                hit_groups={"hit": ray.OptixHitGroup(E(0, "__closesthit__value"), E(0, "__anyhit__mask"))},
                parameters=parameters,
                payload_count=1,
            )
        else:
            program = _program(provider)
        owners.enter_context(program)
        recording = _record(program, scene)
        recording = program.record(
            8,
            raygen=recording.raygen,
            miss=recording.miss,
            hit=tuple(reversed(recording.hit)) if drift == "sbt_mapping" else recording.hit,
            parameters={"scene": "world", "output": "output", "bias": 5 if scalar == 17 else scalar, "count": 8},
            scenes={"world": scene},
        )
        launch = owners.enter_context(recording.prepare(dict(output=output))).initialize()
        expected = np.array([246, 26, 26, 26] * 2, dtype=np.uint32)
        kind = "optix_program"

    @ti.kernel
    def first(source: ti.types.ndarray(ti.u32, ndim=1), target: ti.types.ndarray(ti.u32, ndim=1)):
        for i in range(8):
            target[i] = source[i] + 1

    @ti.kernel
    def second(source: ti.types.ndarray(ti.u32, ndim=1), target: ti.types.ndarray(ti.u32, ndim=1)):
        for i in range(8):
            target[i] = source[i] * 2

    builder = ti.graph.GraphBuilder()
    builder.append_native(launch.graph_recording(), admission="auto")
    args = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.u32, ndim=1) for name in ("output", "middle", "result")]
    builder.dispatch(first, args[0], args[1])
    builder.dispatch(second, args[1], args[2])
    return builder.freeze(), dict(output=output, middle=middle, result=result), expected, launch, kind


def _contracts(backend):
    return dict(
        workload_context=ti.graph.GraphWorkloadContext({"fixture": "native-program-consumer", "count": 8}),
        evaluation_contract=ti.graph.GraphEvaluationContract(
            {"metric": "structural-candidate-not-performance", "oracle": "integer-v1"}
        ),
        backend_environment=ti.graph.GraphBackendEnvironment(
            {"backend": backend, "scope": "test-contract-not-device-qualification"}
        ),
        target=ti.graph.GraphOptimizationTarget(objectives=(("contract_candidate", "max"),)),
    )


def _providers():
    return (GraphRuntimeAssemblyProvider(), GraphMapFusionRecipeProvider(), GraphBindingFrameRecipeProvider())


def _resolve_program_window(backend, artifact, result_queue):
    try:
        ti.init(arch=ti.vulkan if backend == "vulkan" else ti.cuda, offline_cache=False)
        with ExitStack() as owners:
            definition, arguments, expected, _, _ = _window(owners)
            providers = _providers()
            artifact = ti.graph.GraphRecipeSelectionArtifact.from_dict(artifact)
            applicability = definition.check_recipe_applicability(artifact, providers=providers, **_contracts(backend))
            assert applicability.evidence_applicable, applicability
            selection = definition.resolve_recipe(artifact, providers=providers)
            with definition.materialize(selection) as materialized:
                arguments["output"].from_numpy(np.arange(8, dtype=np.uint32))
                materialized.executor.run(materialized.executor.bind(arguments))
                np.testing.assert_array_equal(arguments["result"].to_numpy(), expected)
                result_queue.put(
                    {"physical_id": materialized.materialized_physical_id, "recipe_id": selection.recipe_id}
                )
    except Exception as error:
        result_queue.put({"error": repr(error)})
        raise
    finally:
        ti.reset()


@test_utils.test(arch=[ti.vulkan, ti.cuda], offline_cache=False)
def test_native_program_recipe_report_reallocation_and_contract_drift():
    with ExitStack() as owners:
        definition, arguments, expected, launch, kind = _window(owners)
        providers = _providers()
        baseline = definition.baseline_recipe.recipe_id
        observed = set()

        def evaluate(graph, request):
            arguments["output"].from_numpy(np.arange(8, dtype=np.uint32))
            graph.run(graph.bind(arguments))
            np.testing.assert_array_equal(arguments["result"].to_numpy(), expected)
            observed.add(request.recipe_id)
            return {"contract_candidate": float(request.recipe_id != baseline)}

        contracts = _contracts(definition.backend)
        decision = definition.search_recipes(
            providers=providers,
            **contracts,
            budget=ti.graph.GraphSearchBudget(evaluation_limit=8, repeat_count=1),
            strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
        ).run(evaluate)
        assert decision.status == "selected", decision.report.to_json()
        assert decision.report.search_complete
        assert baseline in observed
        assert len(observed) == (4 if definition.backend == "vulkan" else 2)
        report = json.loads(decision.report.to_json())
        native = report["reuse"]["context"]["native_source_contracts"]
        assert len(native) == 1
        source = native[0]["source_contract"]
        assert source["kind"] == kind
        assert source["source"] == "provider_declared_not_measured"
        assert source["launch"]["sbt"]["raygen"]["record_count"] == 1
        assert "Frozen native source contracts" in decision.report.to_markdown()
        if kind == "optix_program":
            assert source["execution"]["capture"] == "unavailable"
        # Report reads and memory observations must not change the identity.
        launch.memory_report()
        assert report == json.loads(decision.report.to_json())
        fresh, rebound, expected2, fresh_launch, _ = _window(owners)
        assert fresh.semantic_graph_id == definition.semantic_graph_id
        artifact = ti.graph.GraphRecipeSelectionArtifact.from_dict(decision.selection_artifact.to_dict())
        applicability = fresh.check_recipe_applicability(artifact, providers=providers, **contracts)
        assert applicability.evidence_applicable, applicability
        handle = fresh.resolve_recipe(artifact, providers=providers)
        with fresh.materialize(handle) as materialized:
            rebound["output"].from_numpy(np.arange(8, dtype=np.uint32))
            materialized.executor.run(materialized.executor.bind(rebound))
            np.testing.assert_array_equal(rebound["result"].to_numpy(), expected2)
            assert materialized.materialized_physical_id == artifact.structure["materialized_physical_id"], (
                materialized.manifest.to_dict(),
                report,
            )
        # Retiring a recipe must not retire the caller-owned program packet.
        fresh_launch.run()
        ti.sync()
        drifted = dict(contracts, workload_context=ti.graph.GraphWorkloadContext({"count": 9}))
        assert not fresh.check_recipe_applicability(artifact, providers=providers, **drifted).evidence_applicable
        different, *_ = _window(owners, scalar=23)
        with pytest.raises(ti.graph.GraphRecipeReuseError):
            different.resolve_recipe(artifact, providers=providers)
        if definition.backend == "cuda":
            for change in ("code", "layout", "sbt_mapping"):
                # Valid programs with a changed contract must not inherit a
                # previous program's selection. No stale executable is loaded.
                with ExitStack() as changed_owners:
                    changed, *_ = _window(changed_owners, drift=change)
                    with pytest.raises(ti.graph.GraphRecipeReuseError):
                        changed.resolve_recipe(artifact, providers=providers)
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(target=_resolve_program_window, args=(definition.backend, artifact.to_dict(), queue))
        child.start()
        child.join(timeout=30)
        try:
            assert not child.is_alive(), "program recipe child did not finish"
            assert child.exitcode == 0
            assert queue.get(timeout=5) == {
                "physical_id": artifact.structure["materialized_physical_id"],
                "recipe_id": handle.recipe_id,
            }
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
            queue.close()
