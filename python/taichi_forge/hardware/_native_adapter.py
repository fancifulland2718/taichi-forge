"""Shared adapters for explicit hardware recordings in root Graphs."""

from taichi_forge.graph._ir import ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.hardware._retained import validate_retained_execution_contract
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang import impl


def validate_exact_bindings(recording, bindings, operation):
    """Reject missing and unexpected bindings with one stable diagnostic."""

    required = frozenset(recording.binding_names)
    provided = frozenset(bindings)
    if provided == required:
        return
    missing = sorted(required.difference(provided))
    unexpected = sorted(provided.difference(required))
    details = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise TaichiRuntimeError(
        f"{operation} bindings do not match the recording: "
        + "; ".join(details)
    )


def runtime_generation_matches(owner):
    """Returns whether a provider object belongs to the active Program."""

    return (
        impl.get_runtime().prog is owner._runtime_prog
        and int(impl.runtime_generation()) == owner._runtime_generation
    )


def validate_runtime_generation(owner, message):
    if not runtime_generation_matches(owner):
        raise TaichiRuntimeError(message)


def static_resource_effect(resource, access, *, subresource=None):
    return ResourceEffect(
        resource,
        access,
        runtime_bound=False,
        subresource=subresource,
    )


def _resolve(value, recording):
    return value(recording) if callable(value) else value


class HardwareRecordingExecutable(NativeGraphExecutable):
    def __init__(
        self,
        recording,
        *,
        runtime_bindings,
        lifetime_leases,
        debug_info,
    ):
        self._recording = recording
        self._runtime_bindings = runtime_bindings
        self._lifetime_leases = lifetime_leases
        self._debug_info = debug_info
        validate_retained_execution_contract(
            recording, tuple(_resolve(lifetime_leases, recording))
        )
        self._action = BackendCommandGraphAction(recording)

    def run(self, runtime_args):
        return self._recording.execute(runtime_args)

    @property
    def runtime_arg_schema(self):
        bindings = _resolve(self._runtime_bindings, self._recording)
        return tuple(RuntimeBinding(name, kind) for name, kind in bindings)

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def lifetime_leases(self):
        return tuple(_resolve(self._lifetime_leases, self._recording))

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return dict(_resolve(self._debug_info, self._recording))


class HardwareRecordingNode(NativeGraphNode):
    def __init__(
        self,
        recording,
        *,
        runtime_bindings,
        lifetime_leases,
        debug_info,
    ):
        self._recording = recording
        self._options = {
            "runtime_bindings": runtime_bindings,
            "lifetime_leases": lifetime_leases,
            "debug_info": debug_info,
        }

    def compile(self):
        return HardwareRecordingExecutable(self._recording, **self._options)


def native_recording_node(
    recording,
    *,
    runtime_bindings=None,
    lifetime_leases=(),
    debug_info=None,
):
    """Builds the common recordable-action Graph adapter."""

    if runtime_bindings is None:
        runtime_bindings = lambda item: tuple(
            (name, "ndarray") for name in item.binding_names
        )
    if debug_info is None:
        debug_info = {}
    return HardwareRecordingNode(
        recording,
        runtime_bindings=runtime_bindings,
        lifetime_leases=lifetime_leases,
        debug_info=debug_info,
    )


__all__ = [
    "HardwareRecordingExecutable",
    "HardwareRecordingNode",
    "native_recording_node",
    "runtime_generation_matches",
    "static_resource_effect",
    "validate_exact_bindings",
    "validate_runtime_generation",
]
