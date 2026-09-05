"""Whole-Graph immutable argument-image and executable-reuse recipes."""

from dataclasses import replace

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask


def _eligible(spec, backend):
    from taichi_forge._lib import core
    from taichi_forge.graph._graph import _CompiledCGraphNode
    from taichi_forge.lang import impl

    if backend != "cuda" or impl.current_cfg().arch != core.Arch.cuda:
        return False
    native = getattr(core, "_CudaGraphBindingExecutor", None)
    if native is None or not native.available():
        return False
    config = impl.current_cfg()
    if config.debug or config.kernel_profiler or len(spec.nodes) != 1:
        return False
    node = spec.nodes[0]
    return (
        isinstance(node, _CompiledCGraphNode)
        and spec.needs_runtime_args
        and not spec.snode_tree_dependency_info
        and not node.source_native_count
        and not node.native_action_manifests
        and not node.temporary_actions
        and not node.parallel_dispatch_groups
        and all(operation[0] == "dispatch" for operation in node.recipe_operations)
    )


class _BindingFrameExecutor:
    """Installed once; public binding versions own the immutable native frames."""

    execution_kind = "cuda_prepared_binding_graph"
    physical_submission_mode = "cuda_immutable_argument_frames_exec_reuse"

    def __init__(self, instance):
        from taichi_forge._lib import core
        from taichi_forge.graph._graph import _GraphRunContext
        from taichi_forge.lang import impl

        spec = instance.spec
        if not _eligible(spec, "cuda"):
            raise ValueError("immutable argument frames require one ordinary CUDA ndarray Graph")
        node = spec.nodes[0]
        self._native = core._CudaGraphBindingExecutor(node.compiled_graph, impl.current_cfg(), impl.get_runtime().prog)
        self._dispatch_count = node.physical_dispatch_count
        self._task_count = sum(len(stage["tasks"]) for stage in spec.pipeline_definition)
        self._raw_context = _GraphRunContext()

    def prewarm(self):
        return self

    def prepare_binding_version(self, version):
        if not version.fast_path_qualified:
            raise ValueError("immutable argument frames require a publication-qualified Graph binding")
        frame = self._native.prepare(version.flattened_args)
        return replace(version, execution_frame=frame)

    def run_prepared(self, invocation):
        version = invocation.binding_version
        if version is None:
            # A mapping has no immutable publication identity: this is an
            # explicit prepare+run call, not an allegedly upload-free replay.
            context = self._raw_context
            context.begin(invocation.arguments, flattened_args=invocation.flattened_args)
            try:
                frame = self._native.prepare(context.flattened_args())
            finally:
                context.end()
        else:
            frame = version.execution_frame
        self._native.run(frame)

    def invalidate_runtime(self, preserve_executables=False):
        self._native.close()
        self._raw_context = None

    @property
    def snapshot_graph_stats(self):
        from taichi_forge.graph._graph import _empty_backend_stats

        native = self._native.snapshot()
        result = _empty_backend_stats()
        result.update(
            backend="cuda",
            # This names the installed plan, not a counted last launch. No
            # replay counters or timing instrumentation are enabled by reading.
            last_path="cuda_prepared_binding_plan",
            diagnostics_counters_complete=False,
            known_compiled_dispatches=self._dispatch_count,
            known_compiled_tasks=self._task_count,
            known_persistent_argument_bytes=native["argument_bytes"],
            binding_frame_state=native,
        )
        return result

    @property
    def debug_graph_stats(self):
        return self.snapshot_graph_stats


class GraphBindingFrameRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "binding_frames",
        capabilities=("immutable-argument-images", "whole-graph-executable-reuse", "typed-runtime-fragment"),
        domain_version="immutable-binding-frame-domain-v1",
        semantic_fingerprint="ordinary-cuda-graph-binding-lifetime-v1",
    )

    def fragments(self, definition):
        if not _eligible(definition._runtime_spec, definition.backend):
            return ()
        spec = definition._runtime_spec
        return (
            _fragment(
                definition,
                family="binding_frames",
                source_key="whole-graph-bindings",
                choice_id="immutable-argument-images",
                coverage=tuple(source.region_id for source in definition.sources),
                tasks=(
                    GraphFragmentTask.create(
                        "whole-graph-bindings:execute",
                        "cuda_complete_graph_binding_reuse",
                        effects=spec.pre_optimization_ir_root.effects,
                        bindings=spec.pre_optimization_ir_root.bindings,
                        physical={
                            "queue": "default",
                            "argument_images": "immutable_per_published_binding",
                            "argument_upload": "preparation_only",
                            "executable_count": 1,
                            "binding_transition": "whole_executable_update",
                            "argument_lifetime": "published_binding_and_inflight_work",
                            "workspace_lanes": 1,
                        },
                    ),
                ),
                provider_descriptor=self.descriptor,
            ),
        )

    def contribute_runtime(self, assembly, selection):
        if selection.source_key != "whole-graph-bindings" or selection.choice_id != "immutable-argument-images":
            raise ValueError("unknown whole-Graph immutable binding selection")
        assembly.select_binding_executor(_BindingFrameExecutor)

    def describe(self, definition, fragment_key):
        return {
            **super().describe(definition, fragment_key),
            "display_name": "Whole-Graph immutable argument frames",
            "changes": (
                "prepare argument images when bindings are published",
                "reuse one executable across prepared bindings without reuploading arguments",
                "retain argument images and allocation leases until last device use",
            ),
            "limitations": (
                "one ordinary CUDA Graph and one workspace lane",
                "no SNode, external synchronization domain, vendor command or device-controlled topology",
                "raw mapping calls include argument preparation; use Graph.bind to amortize it",
                "prepared frames trade retained argument memory and setup for binding-switch cost",
                "whole-Graph coverage is exclusive in the current exact-cover composer",
                "benefit and driver-owned memory require workload measurements",
            ),
        }


__all__ = ["GraphBindingFrameRecipeProvider"]
