"""Cold immutable binding frames for mixed Vulkan kernel/provider Graphs."""

from dataclasses import replace
import weakref


def prepared_boundaries(spec):
    """Existing root actions that keep their own ordered graphics submission.

    This is a cold recipe predicate, not an inline-recording claim. Reuse the
    existing prepared-action contract without importing a hardware family.
    """
    return tuple(
        node
        for node, recording in spec._native_preparers
        if recording.backend == "vulkan"
        and recording.queue == "graphics"
        and recording.stream_binding == "runtime_ordered"
        and recording.no_host_readback
        and not node.temporary_actions
        and getattr(node.executable, "graph_publish_time_binding_validation_stable", False) is True
    )


def eligible(spec, backend):
    from taichi_forge._lib import core
    from taichi_forge.graph._graph import _CompiledCGraphNode, _CompiledNativeGraphNode
    from taichi_forge.lang import impl

    config = impl.current_cfg()
    if (
        backend != "vulkan"
        or config.arch != core.Arch.vulkan
        or config.debug
        or config.kernel_profiler
        or spec.runtime_lifetime_leases
        or not hasattr(core, "_prepare_vulkan_graph_recording")
    ):
        return False
    if not spec.nodes:
        return False
    if spec._texture_binding_requirements:
        native = getattr(core, "_VulkanFixedGraphRecording", None)
        supports_images = getattr(native, "supports_texture_bindings", None)
        if supports_images is None or not supports_images():
            return False
    if spec._acceleration_structure_binding_requirements:
        native = getattr(core, "_VulkanFixedGraphRecording", None)
        supports_as = getattr(native, "supports_acceleration_structure_bindings", None)
        if supports_as is None or not supports_as() or not impl.get_runtime().prog.vulkan_ray_query_available():
            return False
    boundaries = prepared_boundaries(spec)
    reusable = False
    for node in spec.nodes:
        if isinstance(node, _CompiledCGraphNode):
            if (
                node.temporary_actions
                or node.parallel_dispatch_groups
                or node.source_native_count
                or any(op[0] != "dispatch" for op in node.recipe_operations)
            ):
                return False
            if node.snode_tree_dependency_info:
                native = getattr(core, "_VulkanFixedGraphRecording", None)
                supports_trees = getattr(native, "supports_snode_tree_dependencies", None)
                if supports_trees is None or not supports_trees(impl.get_runtime().prog, node.compiled_graph):
                    return False
            reusable = True
        elif isinstance(node, _CompiledNativeGraphNode):
            if node in boundaries:
                continue
            recording = getattr(node.executable, "_recording", None)
            if not callable(getattr(recording, "_vulkan_graph_command", None)) or not getattr(
                recording.source, "_statistics", {}
            ).get("inline_recording_available"):
                return False
            reusable = True
        else:
            return False
    return reusable


class _SegmentedBindingFrame:
    """One published frame; graphics actions retain their original owners.

    Only the compute frames own recorded secondary commands. Closing releases
    those registrations through their existing retirement path, including work
    already queued in a primary command buffer.
    """

    def __init__(self, frames, actions):
        self._frames = tuple(frames)
        self._actions = tuple(actions)
        self._closed = False

    def run(self):
        if self._closed:
            raise RuntimeError("Prepared Vulkan Graph is closed")
        for execute in self._actions:
            execute()

    def close(self):
        self._closed = True
        self._actions = ()
        for frame in self._frames:
            frame.close()
        self._frames = ()

    def argument_bytes(self):
        return sum(frame.argument_bytes() for frame in self._frames)


class VulkanBindingFrameExecutor:
    execution_kind = "vulkan_prepared_binding_graph"
    physical_submission_mode = "vulkan_secondary_immutable_argument_frames"

    def __init__(self, instance):
        from taichi_forge._lib import core
        from taichi_forge.graph._graph import _CompiledCGraphNode, _GraphRunContext
        from taichi_forge.lang import impl

        spec = instance.spec
        if not eligible(spec, "vulkan"):
            raise ValueError(
                "Vulkan binding frames require fixed dispatches, inline-recordable native plans "
                "or prepared runtime-ordered graphics actions"
            )
        self._program = impl.get_runtime().prog
        self._prepare = core._prepare_vulkan_graph_recording
        boundaries = prepared_boundaries(spec)
        self._segments = []
        sources = []
        for node in spec.nodes:
            if node in boundaries:
                if sources:
                    self._segments.append(tuple(sources))
                    sources = []
                self._segments.append(node)
            elif isinstance(node, _CompiledCGraphNode):
                sources.append(node.compiled_graph)
            else:
                recording = node.executable._recording
                sources.append(recording._vulkan_graph_command())
        if sources:
            self._segments.append(tuple(sources))
        self._segmented = bool(boundaries)
        if self._segmented:
            self.physical_submission_mode = "vulkan_secondary_frames_with_ordered_graphics"
        self._frames = weakref.WeakSet()
        self._context = _GraphRunContext()
        self._dispatch_count = spec.dispatch_count
        self._task_count = sum(len(stage["tasks"]) for stage in spec.pipeline_definition)
        self._compute_stage_counts = tuple(
            (stage["physical_dispatch_count"], len(stage["tasks"]))
            for stage in spec.pipeline_definition
            if stage["kind"] == "cgraph"
        )

    def prewarm(self):
        return self

    def _frame(self, arguments, flattened=None, native_actions=None):
        self._context.begin(arguments, flattened_args=flattened)
        frames = []
        actions = []
        try:
            args = self._context.flattened_args()
            for segment in self._segments:
                if isinstance(segment, tuple):
                    frame = self._prepare(self._program, list(segment), args)
                    frames.append(frame)
                    # A missing secondary path is not an ordinary fallback.
                    if not frame.uses_secondary_commands():
                        raise ValueError("Vulkan recipe requires embeddable secondary commands")
                    actions.append(frame.run)
                else:
                    # Graph.bind/raw preparation already created this packet.
                    # Do not rebuild descriptors or change its lifetime policy.
                    actions.append(native_actions[segment])
        except BaseException:
            for frame in reversed(frames):
                frame.close()
            raise
        finally:
            self._context.end()
        frame = _SegmentedBindingFrame(frames, actions) if self._segmented else frames[0]
        self._frames.add(frame)
        return frame

    def prepare_binding_version(self, version):
        if not version.fast_path_qualified:
            raise ValueError("Vulkan immutable frames require publication-qualified owned bindings")
        frame = self._frame(version.execution_arguments, version.flattened_args, version.native_actions)
        return replace(
            version,
            execution_frame=frame,
            fast_path_qualified=True,
            volatile_reasons=(),
        )

    def run_prepared(self, invocation):
        if invocation.binding_version is not None:
            invocation.binding_version.execution_frame.run()
        else:
            # A raw mapping explicitly includes cold preparation. Published
            # Graph.bind versions own the amortized, upload-free replay path.
            frame = self._frame(invocation.arguments, invocation.flattened_args, invocation.native_actions)
            try:
                frame.run()
            finally:
                frame.close()

    def invalidate_runtime(self, preserve_executables=False):
        for frame in tuple(self._frames):
            frame.close()
        self._frames.clear()
        self._segments.clear()
        self._context = None

    @property
    def snapshot_graph_stats(self):
        from taichi_forge.graph._graph import _empty_backend_stats

        result = _empty_backend_stats()
        result.update(
            backend="vulkan",
            last_path=(
                "vulkan_prepared_compute_with_ordered_graphics" if self._segmented else "vulkan_prepared_binding_plan"
            ),
            diagnostics_counters_complete=False,
            known_compiled_dispatches=self._dispatch_count,
            known_compiled_tasks=self._task_count,
            known_persistent_argument_bytes=sum(frame.argument_bytes() for frame in self._frames),
        )
        if self._segmented:
            # Reports consume one row per logical compute segment. Do not let
            # a whole-plan row leave all later segments falsely "not_run".
            # Argument storage belongs to the complete binding, counted once.
            return tuple(
                {
                    **result,
                    "known_compiled_dispatches": dispatches,
                    "known_compiled_tasks": tasks,
                    "known_persistent_argument_bytes": (result["known_persistent_argument_bytes"] if index == 0 else 0),
                }
                for index, (dispatches, tasks) in enumerate(self._compute_stage_counts)
            ) or (result,)
        return result

    @property
    def debug_graph_stats(self):
        return self.snapshot_graph_stats
