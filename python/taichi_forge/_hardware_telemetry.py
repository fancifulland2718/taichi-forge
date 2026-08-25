"""Low-overhead execution counters shared by hardware recordings and Graph."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from types import MappingProxyType


_COUNTER_NAMES = (
    "recordings",
    "graph_recordings",
    "attempted",
    "executed",
    "unsupported",
    "contract_failure",
    "provider_load_failure",
    "provider_plan_failure",
    "provider_execution_failure",
    "completion_failure",
    "fallback",
    "declared_backend_commands",
    "executed_backend_commands",
    "resource_first_uses",
    "resource_reuses",
)
_FAILURE_PHASES = (
    "contract_failure",
    "provider_load_failure",
    "provider_plan_failure",
    "provider_execution_failure",
    "completion_failure",
)
_UNSUPPORTED_MARKERS = (
    " requires ",
    "unavailable",
    "unsupported",
    "not the active",
    "only available on",
)


def _runtime_generation():
    try:
        from taichi_forge.lang import impl  # pylint: disable=C0415

        return int(impl.runtime_generation())
    except (AttributeError, RuntimeError):
        return 0


def _new_counters():
    return {name: 0 for name in _COUNTER_NAMES}


@dataclass(frozen=True)
class HardwareOperationExecutionSnapshot:
    operation_id: str
    recordings: int
    graph_recordings: int
    attempted: int
    executed: int
    unsupported: int
    contract_failure: int
    provider_load_failure: int
    provider_plan_failure: int
    provider_execution_failure: int
    completion_failure: int
    fallback: int
    declared_backend_commands: int
    executed_backend_commands: int
    resource_first_uses: int
    resource_reuses: int

    def to_dict(self):
        return {
            "operation_id": self.operation_id,
            **{name: getattr(self, name) for name in _COUNTER_NAMES},
        }


_lock = Lock()
_generation = None
_counters = {}
_known_operations = set()
_seen_resources = {}


def _ensure_generation_locked(generation):
    global _generation  # pylint: disable=W0603
    if generation == _generation:
        return
    _generation = generation
    _counters.clear()
    _seen_resources.clear()


def _record(operation_id, **increments):
    generation = _runtime_generation()
    with _lock:
        _ensure_generation_locked(generation)
        counters = _counters.setdefault(operation_id, _new_counters())
        for name, amount in increments.items():
            counters[name] += int(amount)


def _classify_failure(error):
    phase = getattr(error, "_taichi_forge_hardware_failure_phase", None)
    if phase in _FAILURE_PHASES:
        return phase
    message = f" {str(error).lower()} "
    if any(marker in message for marker in _UNSUPPORTED_MARKERS):
        return "unsupported"
    return "contract_failure"


@contextmanager
def hardware_failure_phase(phase):
    """Attribute an exception to the explicit hardware stage that raised it."""

    if phase not in _FAILURE_PHASES:
        raise ValueError(f"unknown hardware failure phase {phase!r}")
    try:
        yield
    except Exception as error:
        try:
            setattr(error, "_taichi_forge_hardware_failure_phase", phase)
        except (AttributeError, TypeError):
            pass
        raise


def _provider_library_loaded(provider_id):
    try:
        from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415

        return bool(
            _ti_core.cuda_external_library_status(provider_id)["library_loaded"]
        )
    except (AttributeError, KeyError, RuntimeError, TypeError):
        return None


@contextmanager
def hardware_provider_call(provider_id, *, failure_phase="provider_execution_failure"):
    """Attribute a lazy provider call without parsing the provider error text."""

    if not isinstance(provider_id, str) or not provider_id:
        raise ValueError("hardware provider_id must be nonempty")
    if failure_phase not in _FAILURE_PHASES:
        raise ValueError(f"unknown hardware failure phase {failure_phase!r}")
    loaded_before = _provider_library_loaded(provider_id)
    try:
        yield
    except Exception as error:
        loaded_after = _provider_library_loaded(provider_id)
        phase = (
            "provider_load_failure"
            if loaded_before is False and loaded_after is False
            else failure_phase
        )
        try:
            setattr(error, "_taichi_forge_hardware_failure_phase", phase)
        except (AttributeError, TypeError):
            pass
        raise


def _resource_keys(recording, operation_id, runtime_resource):
    if runtime_resource:
        return (("runtime", operation_id),)
    resources = []
    pipelines = getattr(recording, "pipelines", ())
    resources.extend(pipelines)
    for name in ("pipeline", "plan", "scene", "matrix", "_owner"):
        value = getattr(recording, name, None)
        if value is not None and all(value is not item for item in resources):
            resources.append(value)
    keys = []
    for resource in resources:
        handle = getattr(resource, "_handle", None)
        generation = getattr(resource, "_runtime_generation", None)
        keys.append(
            (type(resource).__name__, generation, handle)
            if handle is not None
            else (type(resource).__name__, id(resource))
        )
    return tuple(keys)


def _record_resource_uses(operation_id, keys):
    if not keys:
        return
    generation = _runtime_generation()
    first_uses = 0
    reuses = 0
    with _lock:
        _ensure_generation_locked(generation)
        seen = _seen_resources.setdefault(operation_id, set())
        for key in keys:
            if key in seen:
                reuses += 1
            else:
                seen.add(key)
                first_uses += 1
        counters = _counters.setdefault(operation_id, _new_counters())
        counters["resource_first_uses"] += first_uses
        counters["resource_reuses"] += reuses


def instrument_hardware_recording(operation_id, *, runtime_resource=False):
    """Decorate one recording without changing its public type or contract."""

    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError("hardware telemetry operation_id must be nonempty")

    def decorate(cls):
        _known_operations.add(operation_id)
        cls.hardware_operation_id = operation_id
        original_init = cls.__init__
        original_execute = cls.execute

        @wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _record(
                operation_id,
                recordings=1,
                declared_backend_commands=int(self.command_count),
            )

        @wraps(original_execute)
        def execute(self, *args, **kwargs):
            _record(operation_id, attempted=1)
            try:
                result = original_execute(self, *args, **kwargs)
            except Exception as error:
                _record(operation_id, **{_classify_failure(error): 1})
                raise
            _record(
                operation_id,
                executed=1,
                executed_backend_commands=int(self.command_count),
            )
            _record_resource_uses(
                operation_id,
                _resource_keys(self, operation_id, runtime_resource),
            )
            return result

        cls.__init__ = init
        cls.execute = execute
        return cls

    return decorate


def record_graph_recording(recording):
    operation_id = getattr(recording, "hardware_operation_id", None)
    if operation_id is not None:
        _record(operation_id, graph_recordings=1)


def operation_executed(operation_id):
    generation = _runtime_generation()
    with _lock:
        _ensure_generation_locked(generation)
        return _counters.get(operation_id, {}).get("executed", 0) > 0


def execution_snapshot():
    generation = _runtime_generation()
    with _lock:
        _ensure_generation_locked(generation)
        result = {}
        for operation_id in sorted(_known_operations):
            values = dict(_counters.get(operation_id, _new_counters()))
            result[operation_id] = HardwareOperationExecutionSnapshot(
                operation_id=operation_id,
                **values,
            )
        return MappingProxyType(result)


__all__ = [
    "HardwareOperationExecutionSnapshot",
    "execution_snapshot",
    "hardware_failure_phase",
    "hardware_provider_call",
    "instrument_hardware_recording",
    "operation_executed",
    "record_graph_recording",
]
