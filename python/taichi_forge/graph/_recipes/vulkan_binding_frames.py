"""Cold immutable binding frames for mixed Vulkan kernel/provider Graphs."""

from dataclasses import replace
import weakref


def prepared_boundaries(spec):
    """Existing root actions that keep their own runtime-ordered submission.

    This is a cold recipe predicate, not an inline-recording claim. Reuse the
    existing prepared-action contract without importing a hardware family.
    Prefer an existing inline command over introducing a new ordered boundary.
    """
    return tuple(
        node
        for node, recording in spec._native_preparers
        if recording.backend == "vulkan"
        and recording.queue in ("graphics", "compute")
        and recording.stream_binding == "runtime_ordered"
        and recording.no_host_readback
        and not callable(getattr(recording, "_vulkan_graph_command", None))
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
        or not hasattr(core, "_publish_vulkan_graph_commands")
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
            if not callable(getattr(recording, "_vulkan_graph_command", None)):
                return False
            # Legacy command-sequence factories also exist before their source
            # has been qualified. Prepared owner-backed factories are already
            # fixed and validate resources when publishing the native command;
            # they do not own a command-sequence source/statistics object.
            source = getattr(recording, "source", None)
            if source is not None and not getattr(source, "_statistics", {}).get("inline_recording_available"):
                return False
            reusable = True
        else:
            return False
    return reusable


def graphics_queue_eligible(spec):
    """Cold capability check for whole-Graph compute/graphics co-recording."""
    from taichi_forge._lib import core
    from taichi_forge.lang import impl

    native = getattr(core, "_VulkanFixedGraphRecording", None)
    supports = getattr(native, "supports_graphics_queue", None)
    boundaries = prepared_boundaries(spec)
    return (
        bool(boundaries)
        and all(
            recording.queue == "graphics" and getattr(recording, "_supports_prepared_graphics_graph_command", False)
            for node, recording in spec._native_preparers
            if node in boundaries
        )
        and supports is not None
        and supports(impl.get_runtime().prog)
    )


class _SegmentedBindingFrame:
    """One published frame; ordered actions retain their original owners.

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


def independent_graphics_eligible(spec):
    from taichi_forge._lib import core
    from taichi_forge.graph._graph import _CompiledCGraphNode
    from taichi_forge.graph._ir import GraphAccess

    if not hasattr(getattr(core, "_VulkanFixedGraphRecording", None), "uses_independent_graphics"):
        return False
    boundaries = prepared_boundaries(spec)
    if len(boundaries) != 1:
        return False
    middle = spec.nodes.index(boundaries[0])
    if middle == 0 or middle + 1 == len(spec.nodes):
        return False
    prefix = spec.nodes[:middle]
    if not all(isinstance(node, _CompiledCGraphNode) for node in prefix + spec.nodes[middle + 1 :]):
        return False
    graphics_resources = {effect.resource for effect in boundaries[0].ir_node.effects}
    pending = [node.ir_node for node in prefix]
    while pending:
        node = pending.pop()
        pending.extend(node.children)
        if any(
            "texture" in str(binding.kind).lower() or "acceleration" in str(binding.kind).lower()
            for binding in node.bindings
        ):
            return False
        if any(effect.access == GraphAccess.OPAQUE or effect.resource in graphics_resources for effect in node.effects):
            return False
    return True


class VulkanBindingFrameExecutor:
    execution_kind = "vulkan_prepared_binding_graph"
    physical_submission_mode = "vulkan_secondary_immutable_argument_frames_published"

    def __init__(self, instance):
        from taichi_forge._lib import core
        from taichi_forge.graph._graph import _CompiledCGraphNode, _GraphRunContext
        from taichi_forge.lang import impl

        spec = instance.spec
        if not eligible(spec, "vulkan"):
            raise ValueError(
                "Vulkan binding frames require fixed dispatches, inline-recordable native plans "
                "or prepared runtime-ordered compute/graphics actions"
            )
        self._program = impl.get_runtime().prog
        self._prepare = core._prepare_vulkan_graph_recording
        self._publish = core._publish_vulkan_graph_commands
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
            self.physical_submission_mode = (
                "vulkan_secondary_frames_with_ordered_graphics_published"
                if all(node.recordable_action.backend_command_recording.queue == "graphics" for node in boundaries)
                else "vulkan_secondary_frames_with_ordered_native_published"
            )
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
        # Publish the complete compute/native sequence, not each secondary
        # segment. Graph.run must make progress without a later unrelated
        # dispatch or ti.sync; Graph.submit retains its enclosing batch.
        self._publish(self._program)

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
                (
                    "vulkan_prepared_compute_with_ordered_graphics"
                    if self.physical_submission_mode == "vulkan_secondary_frames_with_ordered_graphics_published"
                    else "vulkan_prepared_compute_with_ordered_native"
                )
                if self._segmented
                else "vulkan_prepared_binding_plan"
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


class VulkanGraphicsQueueBindingExecutor(VulkanBindingFrameExecutor):
    """An explicitly selected complete recipe, not a global queue override."""

    physical_execution_queue = "graphics"
    independent_graphics = False

    def __init__(self, instance):
        super().__init__(instance)
        if not graphics_queue_eligible(instance.spec):
            raise ValueError("mixed recording requires prepared graphics packets and a compute-capable graphics queue")
        self.execution_kind = "vulkan_prepared_graphics_queue_graph"
        self.physical_submission_mode = "vulkan_complete_graphics_queue_immutable_frame"

    def _frame(self, arguments, flattened=None, native_actions=None):
        self._context.begin(arguments, flattened_args=flattened)
        frame = None
        try:
            sources = []
            for segment in self._segments:
                if isinstance(segment, tuple):
                    sources.extend(segment)
                else:
                    sources.append(native_actions[segment]._vulkan_graph_command())
            frame = self._prepare(self._program, sources, self._context.flattened_args(), self.independent_graphics)
            if not frame.uses_graphics_queue():
                raise ValueError("mixed Graph recipe did not materialize on the graphics queue")
            if self.independent_graphics and not frame.uses_independent_graphics():
                raise ValueError("independent graphics recipe did not materialize its fork/join")
        except BaseException:
            if frame is not None:
                frame.close()
            raise
        finally:
            self._context.end()
        self._frames.add(frame)
        return frame

    @property
    def snapshot_graph_stats(self):
        result = super().snapshot_graph_stats
        rows = result if isinstance(result, tuple) else (result,)
        path = (
            "vulkan_prepared_graphics_fork_join_plan"
            if self.independent_graphics
            else "vulkan_prepared_graphics_queue_plan"
        )
        return tuple({**row, "last_path": path} for row in rows)


class VulkanIndependentGraphicsBindingExecutor(VulkanGraphicsQueueBindingExecutor):
    independent_graphics = True
    physical_execution_queue = None

    def __init__(self, instance):
        super().__init__(instance)
        if not independent_graphics_eligible(instance.spec):
            raise ValueError("independent graphics recipe requires a disjoint buffer-only compute prefix")
        self.execution_kind = "vulkan_prepared_graphics_fork_join_graph"
        self.physical_submission_mode = "vulkan_complete_graphics_fork_join_immutable_frame"

    def refine_physical_topology(self, tasks, commands):
        # The outer fork/join is known even though vendor-internal draw commands
        # remain opaque. Preserve task identities and the manifest's exactness.
        middle = [i for i, task in enumerate(tasks) if task.kind == "native_action"]
        if len(middle) != 1 or len(tasks) != len(commands):
            raise ValueError("independent graphics physical topology does not match its complete recipe")
        index = middle[0]
        if index == 0 or index + 1 == len(tasks):
            raise ValueError("independent graphics physical topology is missing a fork branch or consumer")
        tasks, commands = list(tasks), list(commands)
        for records in (tasks, commands):
            records[index] = replace(records[index], depends_on=())
            records[index + 1] = replace(records[index + 1], depends_on=(index - 1, index))
        return tasks, commands
