"""Cold, provider-owned matmul configuration discovery and reconstruction.

Opaque algorithm bytes are process-local execution data. Only documented
configuration attributes and observed workspace become persistent facts.
"""

import ctypes
from dataclasses import dataclass

from taichi_forge.hardware._cublaslt import (
    CublasLtMatmulPlan,
    _HeuristicResult,
    _MatmulAlgo,
    _set_attribute,
)
from taichi_forge.lang.exception import TaichiRuntimeError

# cuBLASLt public AlgoConfig ABI. The two shape attributes use uint16, not
# the uint32 representation used by most of the earlier attributes.
_CONFIG = (
    ("id", ctypes.c_int32),
    ("tile", ctypes.c_uint32),
    ("split_k", ctypes.c_int32),
    ("reduction", ctypes.c_uint32),
    ("swizzle", ctypes.c_uint32),
    ("custom", ctypes.c_uint32),
    ("stages", ctypes.c_uint32),
    ("inner_shape", ctypes.c_uint16),
    ("cluster_shape", ctypes.c_uint16),
)


@dataclass(frozen=True)
class _AlgorithmChoice:
    configuration: tuple
    workspace_bytes: int

    def __post_init__(self):
        values = tuple(self.configuration)
        if len(values) != len(_CONFIG):
            raise ValueError("Incomplete cuBLASLt algorithm configuration")
        for index, (value, (_, dtype)) in enumerate(zip(values, _CONFIG)):
            if value is None and index >= 7:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or dtype(value).value != value
            ):
                raise ValueError("Invalid cuBLASLt algorithm configuration value")
        if (
            isinstance(self.workspace_bytes, bool)
            or not isinstance(self.workspace_bytes, int)
            or self.workspace_bytes < 0
        ):
            raise ValueError("Invalid cuBLASLt algorithm workspace")
        object.__setattr__(self, "configuration", values)

    def to_dict(self):
        return {
            "configuration": {
                name: value for (name, _), value in zip(_CONFIG, self.configuration)
            },
            "workspace_bytes": self.workspace_bytes,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {
            "configuration",
            "workspace_bytes",
        }:
            raise ValueError("Invalid cuBLASLt algorithm facts")
        values = data["configuration"]
        if not isinstance(values, dict) or set(values) != {name for name, _ in _CONFIG}:
            raise ValueError("Incomplete cuBLASLt algorithm configuration")
        return cls(tuple(values[name] for name, _ in _CONFIG), data["workspace_bytes"])


class _AlgorithmApi:
    def __init__(self, library):
        self.library = library
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        size = ctypes.c_size_t
        algo = ctypes.POINTER(_MatmulAlgo)
        symbol = "cublasLtMatmulAlgoConfigGetAttribute"
        self.get = getattr(library.library, symbol, None)
        if self.get is None:
            raise TaichiRuntimeError(f"cuBLASLt recipe preparation requires {symbol}")
        self.get.argtypes = (algo, integer, pointer, size, ctypes.POINTER(size))
        self.get.restype = integer

    def describe(self, algorithm, workspace_bytes):
        values = []
        for attribute, (_, dtype) in enumerate(_CONFIG):
            value, written = dtype(), ctypes.c_size_t()
            status = self.get(
                ctypes.byref(algorithm),
                attribute,
                ctypes.byref(value),
                ctypes.sizeof(value),
                ctypes.byref(written),
            )
            if attribute >= 7 and status in (
                7,
                15,
            ):  # Older vendor API lacks this optional attribute.
                values.append(None)
                continue
            self.library.require(status, "algorithm configuration query")
            if written.value != ctypes.sizeof(value):
                raise TaichiRuntimeError(
                    "cuBLASLt algorithm configuration ABI size differs"
                )
            values.append(int(value.value))
        return _AlgorithmChoice(tuple(values), int(workspace_bytes))

    def shortlist(self, plan, count):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 0x7FFFFFFF
        ):
            raise ValueError(
                "cuBLASLt preparation shortlist count must fit positive int32"
            )
        library = self.library
        preference = ctypes.c_void_p()
        library.require(
            library.preference_create(ctypes.byref(preference)),
            "recipe preference creation",
        )
        try:
            _set_attribute(
                library,
                library.preference_set_attribute,
                preference,
                1,
                ctypes.c_size_t(plan.workspace_limit_bytes),
                "recipe workspace limit",
            )
            results, returned = (_HeuristicResult * count)(), ctypes.c_int()
            library.require(
                library.heuristic(
                    plan.provider._handle,
                    plan._matmul_desc,
                    *plan._layouts,
                    preference,
                    count,
                    results,
                    ctypes.byref(returned),
                ),
                "recipe heuristic shortlist",
            )
            if not 0 <= returned.value <= count:
                raise TaichiRuntimeError("cuBLASLt returned an invalid shortlist count")
            choices = {}
            for result in results[: returned.value]:
                if (
                    result.state != 0
                    or result.workspace_size > plan.workspace_limit_bytes
                ):
                    continue
                choice = self.describe(result.algo, result.workspace_size)
                if choice not in choices:
                    choices[choice] = _HeuristicResult.from_buffer_copy(result)
            if not choices:
                raise TaichiRuntimeError(
                    "cuBLASLt found no compatible physical matmul strategy"
                )
            return choices
        finally:
            library.require(
                library.preference_destroy(preference), "recipe preference destruction"
            )


class _MatmulRecipePlan(CublasLtMatmulPlan):
    """Graph-only physical plan, or host-only preparation descriptors."""

    def __init__(
        self,
        provider,
        semantics,
        *,
        layout,
        epilogue,
        workspace_limit_bytes,
        choice=None,
        shortlist_size=8,
        preparation_only=False,
    ):
        self._requested_layout, self._requested_epilogue = layout, epilogue
        self._selected_choice = choice
        self._shortlist_size = shortlist_size
        self._preparation_only = preparation_only
        with provider._lock:
            provider._validate_lifetime()
            super().__init__(
                provider, **semantics, workspace_limit_bytes=workspace_limit_bytes
            )
            provider._plans.add(self)

    def _create_native_plan(self):
        if self._selected_choice is None:
            library = self.provider._library
            if not hasattr(library, "_recipe_algorithm_api"):
                library._recipe_algorithm_api = _AlgorithmApi(library)
            self._algorithm_api = library._recipe_algorithm_api
            self._create_descriptors(
                layout=self._requested_layout, epilogue=self._requested_epilogue
            )
            found = self._algorithm_api.shortlist(self, self._shortlist_size)
            self.choices = tuple(found)
            self._algorithm_choice = self.choices[0]
            self.workspace_bytes = self._algorithm_choice.workspace_bytes
            if self._preparation_only:
                return
            self._close_native()
        else:
            self.choices = (self._selected_choice,)
        self._create_shared_plan(
            layout=self._requested_layout,
            epilogue=self._requested_epilogue,
            choice=self.choices[0],
            allocate_workspace=not self._preparation_only,
        )

    def execute(self, bindings):
        raise TaichiRuntimeError("cuBLASLt recipe plans require retained Graph capture")

    def close(self):
        if self._preparation_only:
            # Descriptors never submitted mathematics or owned GPU scratch.
            # No unrelated device synchronization is needed to retire them.
            with self.provider._lock, self._lock:
                self._close_native()
            return None
        return super().close()
