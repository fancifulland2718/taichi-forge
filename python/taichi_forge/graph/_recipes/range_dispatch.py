"""Whole-dispatch controller recipes over the final compiled range contract."""

from dataclasses import asdict, dataclass
import hashlib
import json
import threading

from taichi_forge.lang._range_dispatch import device_range_dispatch_rejection


@dataclass(frozen=True)
class _RangeDispatchManifest:
    recipe_id: str
    payload_json: str

    def to_dict(self):
        return json.loads(self.payload_json)


class GraphRangeDispatchSource:
    """Retain a qualified source, constructing alternatives only on discovery."""

    def __init__(self, kernel_fn, args, label, kernel_cpp, manifests, template_args):
        self.kernel_fn = kernel_fn
        self.args = tuple(args)
        self.label = label
        self.baseline_kernel_cpp = kernel_cpp
        self.baseline_manifests = manifests
        self.template_args = None if template_args is None else dict(template_args)
        self._recipe_source_key = ""
        self.baseline_plan = None
        self.direct_manifest = None
        self.selected_recipe_id = ""
        self._candidate = None
        self._candidate_prepared = False
        self._candidate_failure = ""
        self._lock = threading.Lock()

    @classmethod
    def try_create(cls, kernel_fn, args, label, kernel_cpp, template_args, task_facts):
        from taichi_forge.graph._graph import _kernel_task_manifests
        from taichi_forge.lang import kernel_impl, impl
        from taichi_forge._lib import core

        if impl.current_cfg().arch not in (core.Arch.cuda, core.Arch.vulkan) or isinstance(
            kernel_fn, kernel_impl._OffloadExecutionPlanBinding
        ):
            return None
        # Builder-local facts are shared by repeated occurrences and disappear
        # with the builder. No shape-keyed global cache or live binding here.
        manifests = task_facts.get(kernel_cpp)
        if manifests is None:
            manifests = _kernel_task_manifests(kernel_cpp)
            task_facts[kernel_cpp] = manifests
        if not any(device_range_dispatch_rejection(task) is None for task in manifests):
            return None
        return cls(kernel_fn, args, label, kernel_cpp, manifests, template_args)

    def _manifest(self, plan, manifests):
        payload = {
            "schema": "compiled-range-dispatch-v1",
            "semantic_kernel_identity": plan.semantic_kernel_identity,
            "offload_plan": plan.stable_payload,
            "offload_compilation_identity": plan.compilation_identity,
            "physical_stages": tuple(
                {"name": f"offload_{task.task_index}", "kind": task.task_type, "manifest": asdict(task)}
                for task in manifests
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return _RangeDispatchManifest("range-dispatch:" + hashlib.sha256(encoded.encode()).hexdigest(), encoded)

    def manifests(self):
        from taichi_forge.graph._graph import gen_cpp_kernel, _kernel_task_manifests
        from taichi_forge.lang._offload_execution_plan import _OffloadExecutionPlan, _bind_offload_execution_plan
        from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError

        with self._lock:
            if not self._candidate_prepared:
                self.baseline_plan = _OffloadExecutionPlan.from_task_manifests(self.baseline_manifests)
                self.direct_manifest = self._manifest(self.baseline_plan, self.baseline_manifests)
                self.selected_recipe_id = self.direct_manifest.recipe_id
                plan = self.baseline_plan
                for task in self.baseline_manifests:
                    if device_range_dispatch_rejection(task) is None:
                        plan = plan.replace_task(task.task_index, range_dispatch="device")
                try:
                    binding = _bind_offload_execution_plan(self.kernel_fn, plan)
                    kernel_cpp = gen_cpp_kernel(binding, self.args, template_args=self.template_args)
                    tasks = _kernel_task_manifests(kernel_cpp)
                    plan.validate_materialization(tasks)
                    self._candidate = (self._manifest(plan, tasks), kernel_cpp)
                except (TypeError, ValueError, RuntimeError, TaichiCompilationError, TaichiRuntimeError) as error:
                    self._candidate_failure = str(error) or type(error).__name__
                self._candidate_prepared = True
        return (self.direct_manifest,) if self._candidate is None else (self.direct_manifest, self._candidate[0])

    def materialize(self, recipe_id):
        self.manifests()
        if recipe_id == self.direct_manifest.recipe_id:
            return self.baseline_kernel_cpp
        if self._candidate is not None and recipe_id == self._candidate[0].recipe_id:
            return self._candidate[1]
        raise ValueError("range dispatch recipe unavailable: " + self._candidate_failure)
