"""Provider-honest memory reports for explicit hardware operations."""

from dataclasses import dataclass
from typing import Optional

HARDWARE_MEMORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HardwareMemoryComponent:
    """One provider resource using requested-byte, not allocator-raw, accounting."""

    name: str
    requested_bytes: Optional[int]
    requested_bytes_exact: bool
    lifetime: str
    ownership: str
    resident: bool = False
    reusable: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("hardware memory component name must be nonempty")
        if self.requested_bytes is not None and (
                isinstance(self.requested_bytes, bool)
                or not isinstance(self.requested_bytes, int)
                or self.requested_bytes < 0):
            raise ValueError(
                "hardware memory component requested_bytes must be nonnegative or None"
            )
        if self.requested_bytes is None and self.requested_bytes_exact:
            raise ValueError("unknown hardware memory bytes cannot be exact")
        if not isinstance(self.resident, bool):
            raise TypeError("hardware memory component resident must be bool")
        if self.lifetime not in ("runtime", "provider_generation",
                                 "invocation"):
            raise ValueError("unsupported hardware memory component lifetime")
        if self.ownership not in (
                "runtime",
                "provider",
                "driver",
                "shared_user_object",
        ):
            raise ValueError("unsupported hardware memory component ownership")

    def to_dict(self):
        return {
            "name": self.name,
            "requested_bytes": self.requested_bytes,
            "requested_bytes_exact": self.requested_bytes_exact,
            "lifetime": self.lifetime,
            "ownership": self.ownership,
            "resident": self.resident,
            "reusable": self.reusable,
        }


@dataclass(frozen=True)
class HardwareMemoryReport:
    """Immutable requested-byte report for one provider-owned generation.

    ``None`` is intentional: driver objects and vendor-library workspaces are
    not assigned fabricated byte counts when the loaded provider cannot expose
    them. Runtime-wide allocator statistics remain the source for raw/committed
    device memory.
    """

    schema_version: int
    provider: str
    backend: str
    lifecycle_state: str
    ownership_scope: str
    components: tuple

    def __post_init__(self):
        if self.schema_version != HARDWARE_MEMORY_SCHEMA_VERSION:
            raise ValueError("hardware memory report schema version mismatch")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("hardware memory provider must be nonempty")
        if self.backend not in ("cpu", "cuda", "vulkan"):
            raise ValueError("unsupported hardware memory backend")
        if self.lifecycle_state not in ("ready", "closed", "runtime_invalid"):
            raise ValueError("unsupported hardware memory lifecycle state")
        components = tuple(self.components)
        if not all(
                isinstance(item, HardwareMemoryComponent)
                for item in components):
            raise ValueError(
                "hardware memory report components must be HardwareMemoryComponent values"
            )
        object.__setattr__(self, "components", components)

    @property
    def known_resident_requested_bytes(self):
        return sum(item.requested_bytes for item in self.components
                   if item.resident and item.requested_bytes is not None)

    @property
    def known_capacity_requested_bytes(self):
        return sum(item.requested_bytes for item in self.components
                   if item.requested_bytes is not None)

    @property
    def resident_requested_bytes_complete(self):
        return all(not item.resident or item.requested_bytes is not None
                   for item in self.components)

    @property
    def opaque_component_count(self):
        return sum(item.resident and item.requested_bytes is None
                   for item in self.components)

    def to_dict(self):
        return {
            "schema_version":
            self.schema_version,
            "provider":
            self.provider,
            "backend":
            self.backend,
            "lifecycle_state":
            self.lifecycle_state,
            "ownership_scope":
            self.ownership_scope,
            "components":
            tuple(item.to_dict() for item in self.components),
            "known_resident_requested_bytes":
            (self.known_resident_requested_bytes),
            "known_capacity_requested_bytes":
            (self.known_capacity_requested_bytes),
            "resident_requested_bytes_complete":
            (self.resident_requested_bytes_complete),
            "opaque_component_count":
            self.opaque_component_count,
        }


def make_memory_report(
    provider,
    backend,
    components,
    *,
    lifecycle_state="ready",
    ownership_scope="provider_generation",
):
    return HardwareMemoryReport(
        schema_version=HARDWARE_MEMORY_SCHEMA_VERSION,
        provider=provider,
        backend=backend,
        lifecycle_state=lifecycle_state,
        ownership_scope=ownership_scope,
        components=tuple(components),
    )


__all__ = [
    "HARDWARE_MEMORY_SCHEMA_VERSION",
    "HardwareMemoryComponent",
    "HardwareMemoryReport",
]
