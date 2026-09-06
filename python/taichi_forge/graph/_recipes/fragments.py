"""Provider-owned replacement fragments for complete Graph recipes."""

import json
from dataclasses import dataclass, field

from taichi_forge.graph._ir import ResourceEffect, RuntimeBinding, TemporaryRequirement
from taichi_forge.graph._recipes.definition import _canonical_json, _digest
from taichi_forge.graph._recipes.providers import RUNTIME_GRAPH_ASSEMBLY_V1

_FRAGMENT_SCHEMA = "taichi_forge.graph_recipe_fragment.v2"
_TASK_SCHEMA = "taichi_forge.graph_fragment_task.v1"
_PLANNED_FRAGMENT_SCHEMA = "taichi_forge.graph_fragment_planned_physical.v1"


def _required_text(value, role):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a non-empty string")
    return value


def _normalized_strings(values, role):
    values = tuple(values)
    for value in values:
        _required_text(value, role)
    if len(values) != len(set(values)):
        raise ValueError(f"{role} values must be unique")
    return tuple(sorted(values))


@dataclass(frozen=True)
class GraphFragmentTask:
    """One provider-local physical task in a replacement DAG."""

    task_id: str
    kind: str
    depends_on: tuple[str, ...] = ()
    effects: tuple[ResourceEffect, ...] = ()
    bindings: tuple[RuntimeBinding, ...] = ()
    temporaries: tuple[TemporaryRequirement, ...] = ()
    _physical_payload_json: str = field(default="{}", repr=False)

    @classmethod
    def create(
        cls,
        task_id,
        kind,
        *,
        depends_on=(),
        effects=(),
        bindings=(),
        temporaries=(),
        physical=None,
    ):
        _required_text(task_id, "Graph fragment task ID")
        _required_text(kind, "Graph fragment task kind")
        depends_on = _normalized_strings(depends_on, "Graph fragment dependency")
        if task_id in depends_on:
            raise ValueError("Graph fragment task cannot depend on itself")
        effects = tuple(effects)
        bindings = tuple(bindings)
        temporaries = tuple(temporaries)
        if not all(isinstance(item, ResourceEffect) for item in effects):
            raise TypeError("Graph fragment task effects must be ResourceEffect values")
        if not all(isinstance(item, RuntimeBinding) for item in bindings):
            raise TypeError(
                "Graph fragment task bindings must be RuntimeBinding values"
            )
        if not all(isinstance(item, TemporaryRequirement) for item in temporaries):
            raise TypeError(
                "Graph fragment task temporaries must be TemporaryRequirement values"
            )
        physical_payload_json = _canonical_json({} if physical is None else physical)
        return cls(
            task_id=task_id,
            kind=kind,
            depends_on=depends_on,
            effects=effects,
            bindings=bindings,
            temporaries=temporaries,
            _physical_payload_json=physical_payload_json,
        )

    @property
    def physical(self):
        import json

        return json.loads(self._physical_payload_json)

    def to_dict(self):
        return {
            "schema": _TASK_SCHEMA,
            "task_id": self.task_id,
            "kind": self.kind,
            "depends_on": self.depends_on,
            "effects": tuple(item.to_dict() for item in self.effects),
            "bindings": tuple(item.to_dict() for item in self.bindings),
            "temporaries": tuple(item.to_dict() for item in self.temporaries),
            "physical": self.physical,
        }


@dataclass(frozen=True)
class GraphFragmentBindingRequirement:
    """A public or fragment-internal symbolic binding requirement."""

    name: str
    kinds: tuple[str, ...] = ()
    required: bool = True
    scope: str = "public"

    def __post_init__(self):
        _required_text(self.name, "Graph fragment binding name")
        if self.scope not in ("public", "fragment_internal"):
            raise ValueError("Graph fragment binding scope is invalid")
        object.__setattr__(
            self,
            "kinds",
            _normalized_strings(self.kinds, "Graph fragment binding kind"),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "kinds": self.kinds,
            "required": self.required,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class GraphFragmentResourceRequirement:
    """Storage ownership and lifetime required by one fragment."""

    name: str
    kind: str
    bytes: int = 0
    alignment: int = 1
    ownership: str = "fragment"
    lifetime: str = "graph"
    exclusive_submission: bool = False

    def __post_init__(self):
        _required_text(self.name, "Graph fragment resource name")
        _required_text(self.kind, "Graph fragment resource kind")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise TypeError("Graph fragment resource bytes must be an integer")
        if self.bytes < 0:
            raise ValueError("Graph fragment resource bytes must be non-negative")
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
        ):
            raise ValueError("Graph fragment resource alignment must be positive")
        if self.ownership not in ("fragment", "graph_instance", "shared", "external"):
            raise ValueError("Graph fragment resource ownership is invalid")
        if self.lifetime not in ("task", "submission", "graph", "session"):
            raise ValueError("Graph fragment resource lifetime is invalid")

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "bytes": self.bytes,
            "alignment": self.alignment,
            "ownership": self.ownership,
            "lifetime": self.lifetime,
            "exclusive_submission": self.exclusive_submission,
        }


@dataclass(frozen=True)
class GraphFragmentSubmissionRequirement:
    """Backend-neutral recording, queue, barrier, and admission facts."""

    queue: str = "default"
    recording_scope: str = "graph"
    barrier_before: bool = False
    barrier_after: bool = False
    exclusive_submission: bool = False
    # An executor wraps the assembled computation; it does not replace the
    # covered semantic regions. Region providers must explicitly opt in.
    executor_kind: str = ""
    compatible_executor_kinds: tuple[str, ...] = ()

    def __post_init__(self):
        _required_text(self.queue, "Graph fragment queue")
        _required_text(self.recording_scope, "Graph fragment recording scope")
        if not isinstance(self.executor_kind, str):
            raise TypeError("Graph executor kind must be a string")
        object.__setattr__(
            self,
            "compatible_executor_kinds",
            _normalized_strings(self.compatible_executor_kinds, "Graph compatible executor kind"),
        )
        if self.executor_kind and self.compatible_executor_kinds:
            raise ValueError("Graph executor cannot also declare region compatibility")

    def to_dict(self):
        return {
            "queue": self.queue,
            "recording_scope": self.recording_scope,
            "barrier_before": self.barrier_before,
            "barrier_after": self.barrier_after,
            "exclusive_submission": self.exclusive_submission,
            **({"executor_kind": self.executor_kind} if self.executor_kind else {}),
            **({"compatible_executor_kinds": self.compatible_executor_kinds} if self.compatible_executor_kinds else {}),
        }


def _validate_task_dag(tasks):
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Graph fragment task IDs must be unique")
    for task in tasks:
        missing = set(task.depends_on).difference(by_id)
        if missing:
            raise ValueError(
                "Graph fragment task dependencies are absent: "
                + ", ".join(sorted(missing))
            )

    visiting = set()
    visited = set()

    def visit(task_id):
        if task_id in visited:
            return
        if task_id in visiting:
            raise ValueError("Graph fragment task dependencies contain a cycle")
        visiting.add(task_id)
        for dependency in by_id[task_id].depends_on:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task in tasks:
        visit(task.task_id)


def _planned_task_payloads(tasks):
    """Canonicalize task labels to local indices for physical identity."""

    index_by_id = {task.task_id: index for index, task in enumerate(tasks)}
    return tuple(
        {
            "task_index": index,
            "kind": task.kind,
            "depends_on": tuple(
                sorted(index_by_id[dependency] for dependency in task.depends_on)
            ),
            "effects": tuple(
                sorted(
                    (item.to_dict() for item in task.effects),
                    key=_canonical_json,
                )
            ),
            "bindings": tuple(
                sorted(
                    (item.to_dict() for item in task.bindings),
                    key=_canonical_json,
                )
            ),
            "temporaries": tuple(
                sorted(
                    (item.to_dict() for item in task.temporaries),
                    key=_canonical_json,
                )
            ),
            "physical": task.physical,
        }
        for index, task in enumerate(tasks)
    )


@dataclass(frozen=True)
class GraphRecipeFragment:
    """A complete provider-owned replacement for one or more semantic regions."""

    fragment_id: str
    planned_physical_id: str
    semantic_graph_id: str
    provider_namespace: str
    provider_version: str
    provider_domain_version: str
    fragment_key: str
    assembly_protocol: str
    assembly_provider_namespace: str
    coverage_region_ids: tuple[str, ...]
    semantic_region_digests: tuple[tuple[str, str], ...]
    tasks: tuple[GraphFragmentTask, ...]
    binding_requirements: tuple[GraphFragmentBindingRequirement, ...]
    resources: tuple[GraphFragmentResourceRequirement, ...]
    submission: GraphFragmentSubmissionRequirement
    backend_requirements: tuple[str, ...]
    capability_requirements: tuple[str, ...]
    _provider_metadata_json: str = field(default="{}", repr=False)

    @classmethod
    def create(
        cls,
        definition,
        *,
        provider_namespace,
        provider_version,
        provider_domain_version,
        fragment_key,
        coverage_region_ids,
        tasks,
        binding_requirements=(),
        resources=(),
        submission=None,
        backend_requirements=(),
        capability_requirements=(),
        assembly_protocol=RUNTIME_GRAPH_ASSEMBLY_V1,
        assembly_provider_namespace=None,
        provider_metadata=None,
    ):
        _required_text(provider_namespace, "Graph fragment provider namespace")
        _required_text(provider_version, "Graph fragment provider version")
        _required_text(
            provider_domain_version,
            "Graph fragment provider domain version",
        )
        _required_text(fragment_key, "Graph fragment key")
        _required_text(assembly_protocol, "Graph fragment assembly protocol")
        if assembly_provider_namespace is None:
            assembly_provider_namespace = provider_namespace
        _required_text(
            assembly_provider_namespace,
            "Graph fragment assembly provider namespace",
        )
        provider_metadata_json = _canonical_json(
            {} if provider_metadata is None else provider_metadata
        )
        requested_coverage = set(
            _normalized_strings(
                coverage_region_ids,
                "Graph fragment coverage region ID",
            )
        )
        if not requested_coverage:
            raise ValueError("Graph fragment coverage must not be empty")
        definition_regions = {region.region_id: region for region in definition.regions}
        unknown = requested_coverage.difference(definition_regions)
        if unknown:
            raise ValueError(
                "Graph fragment covers unknown semantic regions: "
                + ", ".join(sorted(unknown))
            )
        coverage = tuple(
            region.region_id
            for region in definition.regions
            if region.region_id in requested_coverage
        )
        semantic_region_digests = tuple(
            (region_id, definition_regions[region_id].semantic_digest)
            for region_id in coverage
        )
        tasks = tuple(tasks)
        if not tasks:
            raise ValueError("Graph fragment replacement must contain a physical task")
        if not all(isinstance(item, GraphFragmentTask) for item in tasks):
            raise TypeError("Graph fragment tasks must be GraphFragmentTask values")
        _validate_task_dag(tasks)
        bindings = tuple(binding_requirements)
        if not all(
            isinstance(item, GraphFragmentBindingRequirement) for item in bindings
        ):
            raise TypeError(
                "Graph fragment bindings must be GraphFragmentBindingRequirement values"
            )
        bindings = tuple(
            sorted(bindings, key=lambda item: (item.scope, item.name, item.kinds))
        )
        resources = tuple(resources)
        if not all(
            isinstance(item, GraphFragmentResourceRequirement) for item in resources
        ):
            raise TypeError(
                "Graph fragment resources must be GraphFragmentResourceRequirement values"
            )
        resources = tuple(sorted(resources, key=lambda item: (item.name, item.kind)))
        submission = submission or GraphFragmentSubmissionRequirement()
        if not isinstance(submission, GraphFragmentSubmissionRequirement):
            raise TypeError(
                "Graph fragment submission must be a GraphFragmentSubmissionRequirement"
            )
        backend_requirements = _normalized_strings(
            backend_requirements,
            "Graph fragment backend requirement",
        )
        capability_requirements = _normalized_strings(
            capability_requirements,
            "Graph fragment capability requirement",
        )
        # Compatibility is a composition certificate, not different device
        # code. Keep physical deduplication independent of that declaration.
        physical_submission = submission.to_dict()
        physical_submission.pop("compatible_executor_kinds", None)
        physical_payload = {
            "schema": _PLANNED_FRAGMENT_SCHEMA,
            "semantic_graph_id": definition.semantic_graph_id,
            "coverage_region_ids": coverage,
            "tasks": _planned_task_payloads(tasks),
            "binding_requirements": tuple(item.to_dict() for item in bindings),
            "resources": tuple(item.to_dict() for item in resources),
            "submission": physical_submission,
            "backend_requirements": backend_requirements,
            "capability_requirements": capability_requirements,
            "assembly_protocol": assembly_protocol,
            "assembly_provider_namespace": assembly_provider_namespace,
        }
        planned_physical_id = f"fragment-physical:{_digest(physical_payload)}"
        identity_payload = {
            "schema": _FRAGMENT_SCHEMA,
            "provider_namespace": provider_namespace,
            "provider_version": provider_version,
            "provider_domain_version": provider_domain_version,
            "semantic_graph_id": definition.semantic_graph_id,
            "semantic_region_digests": semantic_region_digests,
            "planned_physical_id": planned_physical_id,
            "fragment_key": fragment_key,
            "assembly_protocol": assembly_protocol,
            "assembly_provider_namespace": assembly_provider_namespace,
            "provider_metadata": json.loads(provider_metadata_json),
            **(
                {"compatible_executor_kinds": submission.compatible_executor_kinds}
                if submission.compatible_executor_kinds
                else {}
            ),
        }
        fragment_id = f"graph-fragment:{_digest(identity_payload)}"
        return cls(
            fragment_id=fragment_id,
            planned_physical_id=planned_physical_id,
            semantic_graph_id=definition.semantic_graph_id,
            provider_namespace=provider_namespace,
            provider_version=provider_version,
            provider_domain_version=provider_domain_version,
            fragment_key=fragment_key,
            assembly_protocol=assembly_protocol,
            assembly_provider_namespace=assembly_provider_namespace,
            coverage_region_ids=coverage,
            semantic_region_digests=semantic_region_digests,
            tasks=tasks,
            binding_requirements=bindings,
            resources=resources,
            submission=submission,
            backend_requirements=backend_requirements,
            capability_requirements=capability_requirements,
            _provider_metadata_json=provider_metadata_json,
        )

    @property
    def provider_metadata(self):
        return json.loads(self._provider_metadata_json)

    def to_dict(self):
        return {
            "schema": _FRAGMENT_SCHEMA,
            "fragment_id": self.fragment_id,
            "planned_physical_id": self.planned_physical_id,
            "semantic_graph_id": self.semantic_graph_id,
            "provider_namespace": self.provider_namespace,
            "provider_version": self.provider_version,
            "provider_domain_version": self.provider_domain_version,
            "fragment_key": self.fragment_key,
            "assembly_protocol": self.assembly_protocol,
            "assembly_provider_namespace": self.assembly_provider_namespace,
            "coverage_region_ids": self.coverage_region_ids,
            "semantic_region_digests": self.semantic_region_digests,
            "tasks": tuple(task.to_dict() for task in self.tasks),
            "binding_requirements": tuple(
                item.to_dict() for item in self.binding_requirements
            ),
            "resources": tuple(item.to_dict() for item in self.resources),
            "submission": self.submission.to_dict(),
            "backend_requirements": self.backend_requirements,
            "capability_requirements": self.capability_requirements,
            "provider_metadata": self.provider_metadata,
        }
