"""Private, optional CompileIQ boundary for Forge kernel variants."""

from dataclasses import dataclass
import math
import statistics
from types import MappingProxyType


_FORGE_VARIANT_PARAMETER = "forge_variant"
_SEARCH_STAGES = ("structural", "launch", "full")


def _compileiq_import_error():
    return RuntimeError(
        "CompileIQ is an optional external dependency; install it in the "
        "qualification environment before requesting a CompileIQ search space"
    )


@dataclass(frozen=True)
class _CompileIQVariantSelection:
    variant_id: str
    compilation_id: str


@dataclass(frozen=True)
class _CompileIQPairedTrial:
    variant_id: str
    block: int
    order: tuple[str, str]


@dataclass(frozen=True)
class _CompileIQCandidateEvidence:
    variant_id: str
    ratios: tuple[float, ...]
    median_ratio: float
    worst_ratio: float

    @property
    def worst_positive(self):
        return self.worst_ratio < 1.0


class _CompileIQVariantAdapter:
    """Expose stable Forge variant IDs without importing CompileIQ at startup.

    The adapter deliberately keeps Forge structural and PTXAS searches as
    separate stages.  This avoids coupling the wheel to CompileIQ and also
    works on CompileIQ releases whose Windows core rejects mixed search-space
    lists.
    """

    def __init__(self, session):
        variants = tuple(
            session.variant(variant_id) for variant_id in session.variant_ids()
        )
        if not variants:
            raise ValueError("a CompileIQ adapter requires at least one Forge variant")
        groups = tuple(session.compilation_groups)
        if not groups:
            raise ValueError("a CompileIQ adapter requires compilation groups")

        self._session = session
        self._variants = MappingProxyType(
            {variant.variant_id: variant for variant in variants}
        )
        self._groups = MappingProxyType(
            {group.compilation_id: group for group in groups}
        )
        self._structural_ids = tuple(
            group.representative_variant_id for group in groups
        )

    def variant_ids(self, stage="structural", *, compilation_id=None):
        if stage not in _SEARCH_STAGES:
            raise ValueError("stage must be 'structural', 'launch', or 'full'")
        if stage == "structural":
            if compilation_id is not None:
                raise ValueError("structural search does not accept compilation_id")
            return self._structural_ids
        if stage == "full":
            if compilation_id is not None:
                raise ValueError("full search does not accept compilation_id")
            return tuple(self._variants)
        if compilation_id is None:
            raise ValueError("launch search requires compilation_id")
        try:
            return self._groups[compilation_id].variant_ids
        except KeyError as error:
            raise KeyError(
                f"unknown Forge compilation group {compilation_id!r}"
            ) from error

    def search_space(self, stage="structural", *, compilation_id=None):
        """Build a raw CompileIQ choice space through a lazy import."""

        try:
            from compileiq.search_spaces.base import choice
        except ImportError as error:
            raise _compileiq_import_error() from error
        return {
            _FORGE_VARIANT_PARAMETER: choice(
                self.variant_ids(stage, compilation_id=compilation_id)
            )
        }

    def paired_schedule(
        self,
        stage="structural",
        *,
        compilation_id=None,
        blocks=2,
    ):
        """Return a deterministic AB/BA plan for a finite Forge stage.

        Forge variant stages are deliberately enumerated instead of delegated
        to CompileIQ's genetic core.  This prevents a noisy first-generation
        sample from permanently eliminating a legal structural or launch
        candidate.
        """

        if isinstance(blocks, bool) or not isinstance(blocks, int):
            raise TypeError("blocks must be an integer")
        if blocks < 2 or blocks % 2:
            raise ValueError("blocks must be an even integer >= 2")
        return tuple(
            _CompileIQPairedTrial(
                variant_id=variant_id,
                block=block,
                order=(
                    ("baseline", "candidate")
                    if block % 2 == 0
                    else ("candidate", "baseline")
                ),
            )
            for variant_id in self.variant_ids(stage, compilation_id=compilation_id)
            for block in range(blocks)
        )

    def rank_paired(
        self,
        measurements,
        stage="structural",
        *,
        compilation_id=None,
        blocks=2,
    ):
        """Rank complete paired evidence by worst ratio, then median ratio."""

        expected = self.variant_ids(stage, compilation_id=compilation_id)
        if not isinstance(measurements, dict):
            raise TypeError("measurements must map variant IDs to paired ratios")
        missing = tuple(
            variant_id for variant_id in expected if variant_id not in measurements
        )
        extra = tuple(
            variant_id for variant_id in measurements if variant_id not in expected
        )
        if missing or extra:
            raise ValueError(
                f"paired measurements do not match stage candidates; missing={missing}, extra={extra}"
            )

        evidence = []
        for variant_id in expected:
            raw_ratios = measurements[variant_id]
            if not isinstance(raw_ratios, (tuple, list)):
                raise TypeError("paired ratios must be a tuple or list")
            if len(raw_ratios) != blocks:
                raise ValueError(
                    f"variant {variant_id!r} requires exactly {blocks} paired ratios"
                )
            ratios = tuple(float(value) for value in raw_ratios)
            if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
                raise ValueError("paired ratios must be finite and positive")
            evidence.append(
                _CompileIQCandidateEvidence(
                    variant_id=variant_id,
                    ratios=ratios,
                    median_ratio=float(statistics.median(ratios)),
                    worst_ratio=max(ratios),
                )
            )
        return tuple(
            sorted(
                evidence,
                key=lambda item: (
                    item.worst_ratio,
                    item.median_ratio,
                    item.variant_id,
                ),
            )
        )

    @staticmethod
    def ptxas_search_space(*, version="13.3", variant="default", tag="latest"):
        """Create a separate optional PTXAS provider through a lazy import."""

        try:
            from compileiq.search_spaces.compilers import PtxasSearchSpace
        except ImportError as error:
            raise _compileiq_import_error() from error
        return PtxasSearchSpace(version=version, variant=variant, tag=tag)

    def select(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError("CompileIQ parameters must be a dictionary")
        if _FORGE_VARIANT_PARAMETER not in parameters:
            raise KeyError(f"CompileIQ parameters require {_FORGE_VARIANT_PARAMETER!r}")
        variant_id = parameters[_FORGE_VARIANT_PARAMETER]
        if not isinstance(variant_id, str):
            raise TypeError("forge_variant must be a string")
        try:
            variant = self._variants[variant_id]
        except KeyError as error:
            raise KeyError(f"unknown Forge kernel variant {variant_id!r}") from error
        return _CompileIQVariantSelection(
            variant_id=variant_id,
            compilation_id=variant.compilation_id,
        )

    def bind(self, parameters):
        return self._session.bind(self.select(parameters).variant_id)

    def manifest(self):
        """Return a dependency-free, serializable worker/replay manifest."""

        return {
            "schema_version": 1,
            "parameter": _FORGE_VARIANT_PARAMETER,
            "structural_variant_ids": self._structural_ids,
            "variants": tuple(
                {
                    "variant_id": variant.variant_id,
                    "compilation_id": variant.compilation_id,
                    "spec": variant.spec.stable_payload,
                }
                for variant in self._variants.values()
            ),
        }


__all__ = [
    "_CompileIQVariantAdapter",
    "_CompileIQCandidateEvidence",
    "_CompileIQPairedTrial",
    "_CompileIQVariantSelection",
]
