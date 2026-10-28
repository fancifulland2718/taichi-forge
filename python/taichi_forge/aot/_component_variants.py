"""Bounded compilation of requested normal component configurations.

This collects existing Graph roots; it never enumerates a Boolean product or
interprets numerical execution. Configuration selection remains a cold query.
"""

import json
import math
import time
from dataclasses import dataclass

from taichi_forge.aot._component import _digest, _static_value
from taichi_forge.aot._component_graph import _GraphCapture, compile_component_graph
from taichi_forge.lang import impl


@dataclass(frozen=True)
class BuildBudget:
    max_configurations: int = 64
    max_kernel_variants: int = 128
    max_seconds: float = None

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.max_configurations, self.max_kernel_variants)):
            raise ValueError("Component count budgets must be positive integers")
        if self.max_seconds is not None and (
            type(self.max_seconds) not in (int, float) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("Component time budget must be finite and positive")


class Variant:
    """One normal component recording request, with its ordinary arguments."""

    def __init__(self, component, record, *args, **kwargs):
        self.component, self.record, self.args, self.kwargs = component, record, args, dict(kwargs)


class ComponentBuildError(RuntimeError):
    def __init__(self, reason, report):
        super().__init__(reason)
        self.report = json.loads(json.dumps(report))


class _Build:
    def __init__(self, budget):
        self.budget, self.started = budget, time.perf_counter()
        self.runtime, self.target = impl.get_runtime(), impl.current_cfg().arch.name
        self.requested, self.completed, self.kernels = [], [], {}
        self.admission_complete = False
        self.blocked_variant = None

    def check_time(self, phase):
        if impl.get_runtime() is not self.runtime or impl.current_cfg().arch.name != self.target:
            raise RuntimeError("Component build Runtime changed")
        if self.budget.max_seconds is not None and time.perf_counter() - self.started > self.budget.max_seconds:
            raise RuntimeError(f"Component time budget exceeded {phase}; no complete family was published")

    def admit_kernel(self, code):
        self.check_time("after frontend dependency discovery")
        if code.semantic_key not in self.kernels:
            facts = dict(
                entry=code.entry_definition,
                static=code.contract["static"],
                captures=[name for name, _ in code.contract["captures"]],
            )
            if len(self.kernels) >= self.budget.max_kernel_variants:
                self.blocked_variant = dict(semantic_key=code.semantic_key, **facts)
                raise RuntimeError(
                    f"Component kernel variant budget {self.budget.max_kernel_variants} exceeded by "
                    f"{code.entry_definition[1]}; adjust the declared configurations or build budget"
                )
            self.kernels[code.semantic_key] = dict(**facts, uses=0)
        self.kernels[code.semantic_key]["uses"] += 1

    def report(self):
        return dict(
            version=1,
            target=self.target,
            requested_configurations=self.requested,
            completed_configurations=self.completed,
            missing_configurations=[value for value in self.requested if value not in self.completed],
            kernel_variants=self.kernels,
            elapsed_seconds=time.perf_counter() - self.started,
            budget=vars(self.budget),
            time_limit_scope="checked_between_stages_not_native_preemption",
            request_admission_complete=self.admission_complete,
            blocked_variant=self.blocked_variant,
        )


class ComponentVariants:
    def __init__(self, roots, domain, build):
        self._roots, self._domain = roots, domain
        self._runtime, self._target = build.runtime, build.target
        self._report = json.dumps(build.report(), sort_keys=True)

    @property
    def report(self):
        return json.loads(self._report)

    def select(self, configuration):
        """Query the compiler-produced root for an already built selection."""
        key = _digest({name: _static_value(value) for name, value in configuration.items()})
        for selection, root in self._roots:
            if _digest(selection) == key:
                return root
        raise ValueError("No built component variant for this configuration")

    def export(self, module, name):
        if not isinstance(name, str) or not name or "\0" in name or name in module._component_families:
            raise ValueError("Component family requires an unused nonempty name")
        if module._runtime is not self._runtime or module._arch.name != self._target:
            raise ValueError("Component family and export Module belong to different Runtime/targets")
        named = [
            (name + "__variant_" + _digest(configuration), configuration, root) for configuration, root in self._roots
        ]
        if any(
            root_name in module._component_roots or root_name in module._execution_plans for root_name, _, _ in named
        ):
            raise ValueError("Component family variant name is already exported")
        cases, kernels = [], set()
        try:
            for root_name, configuration, root in named:
                plan = root.export(module, root_name)
                kernels.update(plan.kernel_entries)
                cases.append(
                    dict(configuration=configuration, root=root_name, semantic_key=root.contract["semantic_key"])
                )
        except BaseException:
            # Native Graph publication has no rollback. Do not save a partial
            # family or an apparently complete module after a failed export.
            module._graph_export_failed = True
            raise
        manifest = dict(
            version=1,
            name=name,
            target=self._target,
            domain=self._domain,
            cases=cases,
            build_report=self.report,
            exported_kernel_entries=sorted(kernels),
            coverage="exact_requested_configurations",
            required_capabilities=["component_variant_selection_v1"],
        )
        module._component_families[name] = manifest
        module._content.append("component_family:" + name)
        return json.loads(json.dumps(manifest))


def build_variants(variants, *, budget=None):
    """Build the explicitly requested legal combinations of normal roots.

    A time budget is checked between stages; it does not interrupt an active
    native compiler. Complete frontend definitions may remain cached after a
    failure, but no incomplete family is returned or exported.
    """
    budget = budget or BuildBudget()
    if not isinstance(budget, BuildBudget):
        raise TypeError("Component variants require a BuildBudget")
    build = _Build(budget)
    requests, domain, seen = [], None, set()
    try:
        for variant in variants:
            if not isinstance(variant, Variant):
                raise TypeError("Component variants require normal Variant recording requests")
            if len(requests) >= budget.max_configurations:
                raise ValueError(f"Component configuration budget {budget.max_configurations} exceeded")
            build.check_time("during request admission")
            snapshot = _GraphCapture(variant.component, variant.record, variant.args, variant.kwargs).configuration
            if not snapshot:
                raise ValueError("Component variants require an explicit finite configuration domain")
            current_domain = {
                key: sorted(spec["choices"], key=lambda value: json.dumps(value)) for key, spec in snapshot.items()
            }
            if domain is None:
                domain = current_domain
            elif domain != current_domain:
                raise ValueError("Component variants declare different finite domains")
            configuration = {key: spec["selected"] for key, spec in snapshot.items()}
            identity = _digest(configuration)
            if identity in seen:
                raise ValueError("Component configuration has duplicate or ambiguous recording requests")
            seen.add(identity)
            build.requested.append(configuration)
            requests.append((variant, configuration))
        if not requests:
            raise ValueError("Component variant build requires at least one normal root")
        build.admission_complete = True
        roots = []
        for variant, configuration in requests:
            root = compile_component_graph(
                variant.component, variant.record, *variant.args, _build=build, **variant.kwargs
            )
            actual = {key: spec["selected"] for key, spec in root.contract["configuration"].items()}
            if json.dumps(actual, sort_keys=True) != json.dumps(configuration, sort_keys=True):
                raise ValueError("Component configuration changed after request admission; retry")
            roots.append((configuration, root))
            build.completed.append(configuration)
        return ComponentVariants(roots, domain, build)
    except Exception as error:
        raise ComponentBuildError(str(error), build.report()) from error
