"""Private transport for the compatible modified CompileIQ protocol."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path
from types import MappingProxyType


_CAPABILITY_SCHEMA = "compileiq.taichi-forge-recipe-search-capability.v2"
_FORK_BUILD_ID = "compileiq-taichi-forge-complete-recipes.v2"
_PACKAGE_VERSION = "1.0.0dev6+taichiforge.report1"
_REVIEWED_FORK_REPOSITORY = "https://github.com/fancifulland2718/CompileIQ"
_REVIEWED_FORK_REF = "refs/heads/main"
_REVIEWED_FORK_COMMIT = "f604a79934792a5b17cade81299603fbdb626130"
_REVIEWED_WHEEL_PLATFORM = "win32/amd64"
_REVIEWED_WHEEL_FILENAME = (
    "compileiq-1.0.0.dev6+taichiforge.report1-py3-none-win_amd64.whl"
)
_REVIEWED_WHEEL_SHA256 = (
    "fe8c45f71341736609cc9b2374d7c79e6d9e25e984c5fad2fd087714b2608c9c"
)
_REVIEWED_WHEEL_CORE_LOCK = (
    "sha256:b4838970b7b913bbb7ce6bd50aaa0d132b0df8b11765bd76284736be8a16040b"
)
_REVIEWED_WHEEL_CAPABILITY_ID = (
    "ciq-forge-cap-v2:36e64005bfc7d99fcbed6654f476086482d509c5820640028f3c4af03e5bad08"
)
_DOMAIN_SCHEMA = "compileiq.opaque-recipe-domain.v1"
_BATCH_SCHEMA = "compileiq.opaque-recipe-batch.v2"
_AUDIT_SCHEMA = "compileiq.opaque-recipe-selection.v1"
_CAPABILITY_ID_PREFIX = "ciq-forge-cap-v2:"
_DOMAIN_FINGERPRINT_KEY = "domain_fingerprint"
_RECIPE_ID_KEY = "recipe_id"
_MAX_RECIPES = 4096
_MAX_FIELD_UTF8_BYTES = 4096
_MAX_CANONICAL_BYTES = 4 * 1024 * 1024
_CORE_VERIFICATION = (
    "bundled_manifest_lock_and_platform_hashes_at_search_start_no_override"
)
_OPAQUE_DOMAIN_BINDING = "capability_id_core_commit_core_lock"
_EXPECTED_CORE_COMMIT = "a5a0b8b9414ea62d1d4f6d6bca8dd8904f9518bd"
_EXPECTED_CORE_LOCK = (
    "sha256:0bc59bcd0864ce77dcae75aa00af3f7d641737e9abd0bd3cdb21c78425f127aa"
)
_EXPECTED_CAPABILITY_ID = (
    "ciq-forge-cap-v2:66659a63e3082ef84551c7ff282ab11e0c0f601118a3f7bf1351466eb2edc363"
)
_OBJECTIVE_WORKER = "forge_main_thread_serial_v1"
_EXPECTED_PYTHON_SOURCE_LOCK = (
    "ciq-python-source-v1:"
    "bc8e09772724ac9a741eba7bbed8ac093090bc45878b1ae519736cfd9f0f2144"
)
_REQUIRED_CAPABILITY_KEYS = frozenset(
    (
        "schema",
        "protocol_revision",
        "fork_build_id",
        "package_version",
        "opaque_recipe_domain_schema",
        "opaque_recipe_batch_schema",
        "opaque_dynamic_recipe_domain_schema",
        "selection_audit_schema",
        "opaque_target_contract_schema",
        "opaque_target_selection",
        "trial_outcome_schema",
        "search_checkpoint_schema",
        "evaluation_context_schema",
        "search_finalization_schema",
        "search_status_schema",
        "optimization_report_schema",
        "optimization_report_renderer",
        "max_recipe_ids",
        "max_field_utf8_bytes",
        "max_canonical_bytes",
        "provider_recipe_ids_cross_core_boundary",
        "core_verification",
        "opaque_domain_binding",
        "objective_worker",
        "opaque_recipe_search",
        "opaque_recipe_search_v1",
        "core_manifest_schema_version",
        "core_commit",
        "core_lock",
        "capability_id",
    )
)


class CompileIQOpaqueUnavailableError(RuntimeError):
    """The installed CompileIQ is not compatible with Forge recipe search."""


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(prefix, value):
    encoded = _canonical_json(value).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()


def _installed_compileiq_source_lock(
    support, *, error_type=CompileIQOpaqueUnavailableError
):
    try:
        package_root = Path(support.__file__).resolve().parent
    except (AttributeError, OSError, TypeError) as error:
        raise error_type(
            "modified CompileIQ package source location is unavailable"
        ) from error
    try:
        installed_files = tuple(
            sorted(
                path.relative_to(package_root).as_posix()
                for path in package_root.rglob("*.py")
                if path.is_file()
            )
        )
    except (OSError, ValueError) as error:
        raise error_type(
            "modified CompileIQ Python source manifest is unavailable"
        ) from error
    if not installed_files:
        raise error_type(
            "modified CompileIQ Python source manifest contains no source files"
        )

    entries = []
    for relative_path in installed_files:
        path = package_root / Path(relative_path)
        try:
            normalized = (
                path.read_text(encoding="utf-8")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )
        except (OSError, UnicodeError) as error:
            raise error_type(
                f"modified CompileIQ source file {relative_path!r} is unavailable"
            ) from error
        entries.append(
            (
                relative_path,
                "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            )
        )
    return _identity(
        "ciq-python-source-v1:",
        {
            "schema": "compileiq.taichi-forge-python-source-lock.v1",
            "files": tuple(entries),
        },
    )


def _validated_compileiq_capability(
    *, importer=import_module, error_type=CompileIQOpaqueUnavailableError
):
    try:
        support = importer("compileiq.forge_support")
        recipes = importer("compileiq.recipes")
    except ImportError as error:
        raise error_type(
            "opaque recipe search requires a compatible modified CompileIQ fork "
            "with the V2 complete-recipe capability"
        ) from error

    capability_factory = getattr(support, "forge_recipe_search_capability", None)
    worker_type = getattr(support, "ForgeMainThreadWorker", None)
    session_type = getattr(support, "ForgeOpaqueSearchSessionV2", None)
    budget_type = getattr(support, "ForgeOpaqueSearchBudgetV2", None)
    outcome_type = getattr(support, "TrialOutcomeV2", None)
    cleanup_type = getattr(support, "TrialCleanupV2", None)
    evaluation_context_type = getattr(
        support, "ForgeOpaqueEvaluationContextV1", None
    )
    finalization_type = getattr(support, "ForgeOpaqueSearchFinalizationV1", None)
    status_type = getattr(support, "ForgeOpaqueSearchStatusV2", None)
    report_type = getattr(support, "OpaqueOptimizationReportV1", None)
    report_schema_factory = getattr(
        support, "opaque_optimization_report_json_schema", None
    )
    exhaustive_search_type = getattr(
        support, "ForgeOpaqueRecipeExhaustiveSearchV1", None
    )
    target_contract_type = getattr(support, "ForgeOpaqueTargetContractV1", None)
    domain_type = getattr(recipes, "OpaqueRecipeDomainV1", None)
    batch_type = getattr(recipes, "OpaqueRecipeBatchV2", None)
    fidelity_type = getattr(recipes, "OpaqueRecipeFidelityV2", None)
    lineage_type = getattr(recipes, "OpaqueRecipeLineageV2", None)
    dynamic_domain_type = getattr(recipes, "OpaqueDynamicRecipeDomainV2", None)
    if (
        not callable(capability_factory)
        or domain_type is None
        or not isinstance(batch_type, type)
        or getattr(batch_type, "SCHEMA", None) != _BATCH_SCHEMA
        or not isinstance(fidelity_type, type)
        or not isinstance(lineage_type, type)
        or not isinstance(dynamic_domain_type, type)
        or getattr(dynamic_domain_type, "SCHEMA", None)
        != "compileiq.opaque-dynamic-recipe-domain.v2"
        or not isinstance(worker_type, type)
        or getattr(worker_type, "PROTOCOL", None) != _OBJECTIVE_WORKER
        or not isinstance(session_type, type)
        or getattr(session_type, "PROTOCOL", None)
        != "dynamic_batch_pareto_racing_main_thread_v2"
        or not isinstance(budget_type, type)
        or not isinstance(outcome_type, type)
        or getattr(outcome_type, "SCHEMA", None)
        != "compileiq.taichi-forge-trial-outcome.v2"
        or not isinstance(cleanup_type, type)
        or not isinstance(evaluation_context_type, type)
        or getattr(evaluation_context_type, "SCHEMA", None)
        != "compileiq.taichi-forge-evaluation-context.v1"
        or not isinstance(finalization_type, type)
        or getattr(finalization_type, "SCHEMA", None)
        != "compileiq.taichi-forge-search-finalization.v1"
        or not isinstance(status_type, type)
        or getattr(status_type, "SCHEMA", None)
        != "compileiq.taichi-forge-search-status.v2"
        or not isinstance(report_type, type)
        or getattr(report_type, "SCHEMA", None)
        != "compileiq.opaque-optimization-report.v1"
        or not callable(report_schema_factory)
        or not isinstance(exhaustive_search_type, type)
        or getattr(exhaustive_search_type, "PROTOCOL", None)
        != "bounded_exhaustive_main_thread_v1"
        or not isinstance(target_contract_type, type)
        or getattr(target_contract_type, "SCHEMA", None)
        != "compileiq.taichi-forge-opaque-target-contract.v1"
    ):
        raise error_type(
            "installed CompileIQ does not publish the complete Forge opaque-recipe "
            "capability and main-thread worker"
        )
    try:
        capability = capability_factory().as_dict()
    except Exception as error:
        raise error_type("modified CompileIQ capability negotiation failed") from error
    if not isinstance(capability, dict) or not _REQUIRED_CAPABILITY_KEYS.issubset(
        capability
    ):
        raise error_type("modified CompileIQ capability has a missing required field")

    expected = {
        "schema": _CAPABILITY_SCHEMA,
        "protocol_revision": 6,
        "fork_build_id": _FORK_BUILD_ID,
        "opaque_recipe_domain_schema": _DOMAIN_SCHEMA,
        "opaque_recipe_batch_schema": _BATCH_SCHEMA,
        "opaque_dynamic_recipe_domain_schema": (
            "compileiq.opaque-dynamic-recipe-domain.v2"
        ),
        "selection_audit_schema": _AUDIT_SCHEMA,
        "opaque_target_contract_schema": (
            "compileiq.taichi-forge-opaque-target-contract.v1"
        ),
        "opaque_target_selection": (
            "uncertainty_aware_pareto_layers_no_scalarization_v2"
        ),
        "trial_outcome_schema": "compileiq.taichi-forge-trial-outcome.v2",
        "search_checkpoint_schema": (
            "compileiq.taichi-forge-search-checkpoint.v2"
        ),
        "evaluation_context_schema": (
            "compileiq.taichi-forge-evaluation-context.v1"
        ),
        "search_finalization_schema": (
            "compileiq.taichi-forge-search-finalization.v1"
        ),
        "search_status_schema": "compileiq.taichi-forge-search-status.v2",
        "optimization_report_schema": "compileiq.opaque-optimization-report.v1",
        "optimization_report_renderer": (
            "json_fact_source_markdown_projection_v1"
        ),
        "max_recipe_ids": _MAX_RECIPES,
        "max_field_utf8_bytes": _MAX_FIELD_UTF8_BYTES,
        "max_canonical_bytes": _MAX_CANONICAL_BYTES,
        "provider_recipe_ids_cross_core_boundary": False,
        "core_verification": _CORE_VERIFICATION,
        "opaque_domain_binding": _OPAQUE_DOMAIN_BINDING,
        "objective_worker": _OBJECTIVE_WORKER,
        "opaque_recipe_search": (
            "dynamic_batch_pareto_racing_main_thread_v2"
        ),
        "opaque_recipe_search_v1": "bounded_exhaustive_main_thread_v1",
    }
    if any(capability.get(name) != value for name, value in expected.items()):
        raise error_type(
            "installed CompileIQ capability is incompatible with Forge recipe search"
        )
    package_version = capability["package_version"]
    if not isinstance(package_version, str) or not package_version:
        raise error_type("installed CompileIQ package has no version provenance")
    schema_version = capability["core_manifest_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise error_type("CompileIQ capability has no valid core manifest version")
    core_commit = capability["core_commit"]
    if not isinstance(core_commit, str) or not core_commit:
        raise error_type("CompileIQ capability has no valid bundled-core commit")
    core_lock = capability["core_lock"]
    if (
        not isinstance(core_lock, str)
        or len(core_lock) != len("sha256:") + 64
        or not core_lock.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in core_lock[7:])
    ):
        raise error_type("CompileIQ capability has no valid bundled-core lock")
    identity_payload = {
        name: value for name, value in capability.items() if name != "capability_id"
    }
    if capability["capability_id"] != _identity(
        _CAPABILITY_ID_PREFIX, identity_payload
    ):
        raise error_type("CompileIQ capability identity mismatch")
    if getattr(domain_type, "SCHEMA", None) != _DOMAIN_SCHEMA:
        raise error_type("CompileIQ opaque recipe domain schema mismatch")
    for name, value in (
        ("MAX_RECIPE_IDS", _MAX_RECIPES),
        ("MAX_FIELD_UTF8_BYTES", _MAX_FIELD_UTF8_BYTES),
        ("MAX_CANONICAL_BYTES", _MAX_CANONICAL_BYTES),
    ):
        if getattr(domain_type, name, None) != value:
            raise error_type(f"CompileIQ opaque recipe domain {name} mismatch")
    source_lock = _installed_compileiq_source_lock(support, error_type=error_type)
    return MappingProxyType(dict(capability)), domain_type, worker_type, source_lock


def _reviewed_distribution_manifest():
    return {
        "repository": _REVIEWED_FORK_REPOSITORY,
        "ref": _REVIEWED_FORK_REF,
        "acceptance": "compatible_capability_not_commit_or_wheel_hash",
        "compatibility": {
            "schema": _CAPABILITY_SCHEMA,
            "protocol_revision": 6,
            "fork_build_id": _FORK_BUILD_ID,
        },
        "qualified_snapshot": {
            "commit": _REVIEWED_FORK_COMMIT,
            "package_version": _PACKAGE_VERSION,
            "wheel_filename": _REVIEWED_WHEEL_FILENAME,
            "wheel_platform": _REVIEWED_WHEEL_PLATFORM,
            "wheel_sha256": _REVIEWED_WHEEL_SHA256,
            "capability_id": _REVIEWED_WHEEL_CAPABILITY_ID,
            "core_commit": _EXPECTED_CORE_COMMIT,
            "core_lock": _REVIEWED_WHEEL_CORE_LOCK,
            "python_source_lock": _EXPECTED_PYTHON_SOURCE_LOCK,
        },
        "runtime_verification": (
            "capability_identity_core_manifest_and_session_source_identity"
        ),
    }


class _CompileIQOpaqueRecipeTransport:
    """Bind one provider-owned recipe set to a compatible modified fork."""

    __slots__ = (
        "_baseline_recipe_id",
        "_capability",
        "_domain",
        "_dynamic_domain",
        "_domain_owner",
        "_python_source_lock",
        "_recipe_description",
        "_exhaustive_search_type",
        "_worker_type",
    )

    def __init__(
        self,
        *,
        provider_namespace,
        domain_version,
        provider_semantic_fingerprint,
        recipe_ids,
        baseline_recipe_id,
        capability_components,
        domain_owner,
        recipe_description,
        generation_domain_id=None,
        provider_registry_id=None,
        assembly_protocols=("provider_owned_opaque_recipe.v1",),
        recipe_schema="taichi_forge.opaque_recipe.v1",
        search_strategy_id="manual_dynamic_batches.v2",
    ):
        capability, domain_type, worker_type, python_source_lock = capability_components
        recipe_ids = tuple(recipe_ids)
        if (
            not recipe_ids
            or len(set(recipe_ids)) != len(recipe_ids)
            or any(
                not isinstance(recipe_id, str) or not recipe_id
                for recipe_id in recipe_ids
            )
        ):
            raise ValueError("opaque recipe IDs must be nonempty and unique")
        if baseline_recipe_id not in recipe_ids:
            raise ValueError("opaque recipe domain must contain its baseline")
        if len(recipe_ids) > _MAX_RECIPES:
            raise ValueError("opaque recipe domain exceeds the 4096-recipe limit")
        for name, value in (
            ("domain_owner", domain_owner),
            ("recipe_description", recipe_description),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")

        try:
            domain = domain_type(
                provider_namespace=provider_namespace,
                domain_version=domain_version,
                provider_semantic_fingerprint=provider_semantic_fingerprint,
                compileiq_capability_id=capability["capability_id"],
                compileiq_core_commit=capability["core_commit"],
                compileiq_core_lock=capability["core_lock"],
                recipe_ids=recipe_ids,
            )
        except Exception as error:
            raise ValueError(
                "modified CompileIQ rejected the complete opaque recipe domain"
            ) from error
        try:
            recipes = import_module("compileiq.recipes")
            dynamic_domain_type = getattr(recipes, "OpaqueDynamicRecipeDomainV2")
            dynamic_domain = dynamic_domain_type(
                provider_namespace=provider_namespace,
                domain_version=domain_version,
                generation_domain_id=(
                    provider_semantic_fingerprint
                    if generation_domain_id is None
                    else generation_domain_id
                ),
                provider_registry_id=(
                    _identity(
                        "forge-provider-registry-v1:",
                        {
                            "provider_namespace": provider_namespace,
                            "domain_version": domain_version,
                        },
                    )
                    if provider_registry_id is None
                    else provider_registry_id
                ),
                assembly_protocols=tuple(assembly_protocols),
                recipe_schema=recipe_schema,
                search_strategy_id=search_strategy_id,
                compileiq_capability_id=capability["capability_id"],
                compileiq_core_commit=capability["core_commit"],
                compileiq_core_lock=capability["core_lock"],
            )
        except Exception as error:
            raise ValueError(
                "modified CompileIQ rejected the dynamic opaque recipe domain"
            ) from error
        expected_binding = {
            "compileiq_capability_id": capability["capability_id"],
            "compileiq_core_commit": capability["core_commit"],
            "compileiq_core_lock": capability["core_lock"],
        }
        if any(
            getattr(domain, name, None) != value
            for name, value in expected_binding.items()
        ):
            raise ValueError(
                "modified CompileIQ did not bind the opaque domain to its exact core"
            )
        if frozenset(domain.recipe_ids) != frozenset(recipe_ids):
            raise ValueError(
                "modified CompileIQ changed the opaque recipe identity set"
            )

        compiled_space = domain.to_search_space()
        if not isinstance(compiled_space, dict) or frozenset(compiled_space) != {
            _DOMAIN_FINGERPRINT_KEY,
            _RECIPE_ID_KEY,
        }:
            raise ValueError(
                "modified CompileIQ exposed fields outside the opaque recipe transport"
            )
        core_tokens = getattr(compiled_space[_RECIPE_ID_KEY], "vals", None)
        expected_tokens = [
            f"ciq-recipe-v1-{ordinal:04d}" for ordinal in range(len(domain.recipe_ids))
        ]
        if (
            getattr(compiled_space[_DOMAIN_FINGERPRINT_KEY], "value", None)
            != domain.domain_fingerprint
            or core_tokens != expected_tokens
            or any(not token.isascii() for token in core_tokens or ())
        ):
            raise ValueError("modified CompileIQ changed the safe opaque token set")

        self._baseline_recipe_id = baseline_recipe_id
        self._capability = MappingProxyType(dict(capability))
        self._domain = domain
        self._dynamic_domain = dynamic_domain
        self._domain_owner = domain_owner
        self._python_source_lock = python_source_lock
        self._recipe_description = recipe_description
        self._exhaustive_search_type = getattr(
            worker_type, "OPAQUE_EXHAUSTIVE_SEARCH_TYPE", None
        )
        self._worker_type = worker_type

    @property
    def capability(self):
        return MappingProxyType(dict(self._capability))

    @property
    def search_space(self):
        return self._domain

    @property
    def worker_type(self):
        return self._worker_type

    def exhaustive_search(
        self,
        objective_function,
        *,
        problem_type="min",
        target_contract=None,
    ):
        """Create the exact fork's complete finite-domain search."""

        exhaustive_search_type = self._exhaustive_search_type
        if exhaustive_search_type is None:
            try:
                support = import_module("compileiq.forge_support")
                exhaustive_search_type = getattr(
                    support, "ForgeOpaqueRecipeExhaustiveSearchV1"
                )
            except (ImportError, AttributeError) as error:
                raise CompileIQOpaqueUnavailableError(
                    "modified CompileIQ does not expose bounded exhaustive "
                    "opaque search"
                ) from error
        return exhaustive_search_type(
            objective_function=objective_function,
            search_space=self._domain,
            baseline_recipe_id=self.baseline_recipe_id,
            problem_type=problem_type,
            target_contract=target_contract,
        )

    def batch_v2(
        self,
        *,
        recipe_ids,
        stage_index,
        stage_fingerprint,
        parent_batch=None,
        parent_recipe_ids=None,
        fidelity_name,
        fidelity_ordinal,
        repeat_count,
        work_scale=1.0,
        terminal=False,
        estimated_materialized_bytes=None,
        planned_physical_ids=None,
    ):
        """Build one exact V2 batch without exposing recipe internals."""

        try:
            recipes = import_module("compileiq.recipes")
            batch_type = getattr(recipes, "OpaqueRecipeBatchV2")
            fidelity_type = getattr(recipes, "OpaqueRecipeFidelityV2")
            lineage_type = getattr(recipes, "OpaqueRecipeLineageV2")
        except (ImportError, AttributeError) as error:
            raise CompileIQOpaqueUnavailableError(
                "modified CompileIQ does not expose opaque recipe batches V2"
            ) from error
        recipe_ids = tuple(recipe_ids)
        if not recipe_ids or len(set(recipe_ids)) != len(recipe_ids):
            raise ValueError("V2 batch recipes must be nonempty and unique")
        if len(recipe_ids) > _MAX_RECIPES or any(
            not isinstance(recipe_id, str) or not recipe_id for recipe_id in recipe_ids
        ):
            raise ValueError("V2 batch recipe identities are invalid")
        if self.baseline_recipe_id not in recipe_ids:
            raise ValueError("every V2 batch must retain the frozen baseline")
        parent_recipe_ids = parent_recipe_ids or {}
        estimated_materialized_bytes = estimated_materialized_bytes or {}
        planned_physical_ids = planned_physical_ids or {}
        unexpected_parents = set(parent_recipe_ids) - set(recipe_ids)
        unexpected_estimates = set(estimated_materialized_bytes) - set(recipe_ids)
        unexpected_plans = set(planned_physical_ids) - set(recipe_ids)
        if unexpected_parents or unexpected_estimates or unexpected_plans:
            raise ValueError("V2 batch metadata contains an unknown recipe")
        return batch_type(
            provider_namespace=self._dynamic_domain.provider_namespace,
            domain_version=self._dynamic_domain.domain_version,
            provider_semantic_fingerprint=(
                self._dynamic_domain.generation_domain_id
            ),
            compileiq_capability_id=self._capability["capability_id"],
            compileiq_core_commit=self._capability["core_commit"],
            compileiq_core_lock=self._capability["core_lock"],
            stage_index=stage_index,
            stage_fingerprint=stage_fingerprint,
            parent_batch_fingerprint=(
                None if parent_batch is None else parent_batch.batch_fingerprint
            ),
            fidelity=fidelity_type(
                name=fidelity_name,
                ordinal=fidelity_ordinal,
                repeat_count=repeat_count,
                work_scale=work_scale,
                terminal=terminal,
            ),
            recipes=tuple(
                lineage_type(
                    recipe_id=recipe_id,
                    planned_physical_id=planned_physical_ids.get(
                        recipe_id,
                        _identity("opaque-planned-recipe-v1:", recipe_id),
                    ),
                    parent_recipe_ids=tuple(parent_recipe_ids.get(recipe_id, ())),
                    estimated_materialized_bytes=int(
                        estimated_materialized_bytes.get(recipe_id, 0)
                    ),
                )
                for recipe_id in recipe_ids
            ),
        )

    def search_session_v2(
        self,
        objective_function,
        *,
        target_contract,
        budget,
        deterministic_seed=0,
        halving_factor=2,
        minimum_survivors=1,
        evaluation_context=None,
        checkpoint=None,
    ):
        try:
            support = import_module("compileiq.forge_support")
            session_type = getattr(support, "ForgeOpaqueSearchSessionV2")
            evaluation_context_type = getattr(
                support,
                "ForgeOpaqueEvaluationContextV1",
            )
        except (ImportError, AttributeError) as error:
            raise CompileIQOpaqueUnavailableError(
                "modified CompileIQ does not expose staged opaque search V2"
            ) from error
        if evaluation_context is None:
            evaluation_context = evaluation_context_type(
                reuse_scope="session_only",
                workload_context_id="unspecified-workload-context",
                evaluation_contract_id="unspecified-evaluation-contract",
                backend_environment_id="unspecified-backend-environment",
            )
        elif not isinstance(evaluation_context, evaluation_context_type):
            raise TypeError(
                "evaluation_context must be a ForgeOpaqueEvaluationContextV1"
            )
        return session_type(
            objective_function=objective_function,
            dynamic_domain=self._dynamic_domain,
            evaluation_context=evaluation_context,
            baseline_recipe_id=self.baseline_recipe_id,
            target_contract=target_contract,
            budget=budget,
            deterministic_seed=deterministic_seed,
            halving_factor=halving_factor,
            minimum_survivors=minimum_survivors,
            checkpoint=checkpoint,
        )

    @property
    def python_source_lock(self):
        return self._python_source_lock

    @property
    def domain_fingerprint(self):
        return self._domain.domain_fingerprint

    @property
    def dynamic_domain(self):
        return self._dynamic_domain.model_copy(deep=True)

    @property
    def recipe_ids(self):
        return tuple(self._domain.recipe_ids)

    @property
    def baseline_recipe_id(self):
        return self._baseline_recipe_id

    def decode(self, parameters):
        if not isinstance(parameters, dict) or frozenset(parameters) != {
            _DOMAIN_FINGERPRINT_KEY,
            _RECIPE_ID_KEY,
        }:
            raise ValueError(
                "CompileIQ selection must contain exactly domain_fingerprint "
                "and recipe_id"
            )
        if parameters[_DOMAIN_FINGERPRINT_KEY] != self.domain_fingerprint:
            raise ValueError(
                f"CompileIQ selection belongs to another {self._domain_owner} domain"
            )
        recipe_id = parameters[_RECIPE_ID_KEY]
        if not isinstance(recipe_id, str):
            raise TypeError("CompileIQ recipe_id must be a string")
        if recipe_id not in self.recipe_ids:
            raise KeyError(
                f"CompileIQ selected an unknown {self._recipe_description} "
                f"{recipe_id!r}"
            )
        return recipe_id

    def search_coverage(self, compileiq_search):
        capability = getattr(compileiq_search, "opaque_recipe_capability", None)
        if capability != dict(self._capability):
            raise ValueError(
                "CompileIQ search capability does not match this opaque domain"
            )
        provenance = getattr(compileiq_search, "opaque_recipe_core_provenance", None)
        if not isinstance(provenance, dict) or any(
            provenance.get(name) != self._capability[name]
            for name in ("core_commit", "core_lock")
        ):
            raise ValueError(
                "CompileIQ search has no matching verified core provenance"
            )
        records = getattr(compileiq_search, "opaque_recipe_audit_records", None)
        if not isinstance(records, tuple):
            raise TypeError(
                "CompileIQ search does not expose immutable opaque recipe audits"
            )

        observed = set()
        token_by_recipe = {
            recipe_id: f"ciq-recipe-v1-{ordinal:04d}"
            for ordinal, recipe_id in enumerate(self.recipe_ids)
        }
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("CompileIQ opaque recipe audit must be a dictionary")
            expected = {
                "schema": _AUDIT_SCHEMA,
                "provider_namespace": self._domain.provider_namespace,
                "domain_version": self._domain.domain_version,
                "provider_semantic_fingerprint": (
                    self._domain.provider_semantic_fingerprint
                ),
                "compileiq_capability_id": self._capability["capability_id"],
                "compileiq_core_commit": self._capability["core_commit"],
                "compileiq_core_lock": self._capability["core_lock"],
                _DOMAIN_FINGERPRINT_KEY: self.domain_fingerprint,
            }
            if any(record.get(name) != value for name, value in expected.items()):
                raise ValueError(
                    "CompileIQ opaque recipe audit does not match this domain"
                )
            recipe_id = record.get(_RECIPE_ID_KEY)
            if recipe_id not in self.recipe_ids:
                raise ValueError(
                    "CompileIQ opaque recipe audit contains an unknown recipe"
                )
            if (
                frozenset(record)
                != frozenset(
                    (
                        "param_id",
                        *expected,
                        "core_recipe_token",
                        _RECIPE_ID_KEY,
                    )
                )
                or isinstance(record.get("param_id"), bool)
                or not isinstance(record.get("param_id"), int)
                or record.get("core_recipe_token") != token_by_recipe[recipe_id]
            ):
                raise ValueError(
                    "CompileIQ opaque recipe audit token mapping is invalid"
                )
            observed.add(recipe_id)

        observed_ids = tuple(
            recipe_id for recipe_id in self.recipe_ids if recipe_id in observed
        )
        missing_ids = tuple(
            recipe_id for recipe_id in self.recipe_ids if recipe_id not in observed
        )
        return MappingProxyType(
            {
                "complete": not missing_ids,
                "baseline_observed": self.baseline_recipe_id in observed,
                "evaluation_count": len(records),
                "observed_recipe_ids": observed_ids,
                "missing_recipe_ids": missing_ids,
                "verified_core": True,
            }
        )

    def require_complete_search(self, compileiq_search):
        coverage = self.search_coverage(compileiq_search)
        if not coverage["complete"]:
            raise RuntimeError(
                "modified CompileIQ did not evaluate the complete frozen "
                f"{self._recipe_description} domain; "
                f"missing={coverage['missing_recipe_ids']!r}"
            )
        return coverage

    def select_best_recipe_id(self, compileiq_search, result):
        self.require_complete_search(compileiq_search)
        best_result = getattr(result, "get_best_result", None)
        if not callable(best_result):
            raise TypeError("CompileIQ result does not provide get_best_result()")
        best = best_result()
        if not isinstance(best, dict) or not isinstance(best.get("params"), dict):
            raise TypeError("CompileIQ best result has no decoded parameter dictionary")
        return self.decode(best["params"])

    def manifest(self):
        return {
            "transport": "modified_compileiq_complete_recipe_v2",
            "capability": dict(self._capability),
            "compileiq_python_source_lock": self._python_source_lock,
            "reviewed_compileiq_distribution": (_reviewed_distribution_manifest()),
            "domain": self._domain.model_dump(by_alias=True),
            "domain_fingerprint": self.domain_fingerprint,
            "dynamic_domain": self._dynamic_domain.model_dump(by_alias=True),
            "dynamic_domain_fingerprint": self._dynamic_domain.domain_fingerprint,
            "provider_semantic_fingerprint": self._dynamic_domain.generation_domain_id,
            "baseline_recipe_id": self.baseline_recipe_id,
            "recipe_count": len(self.recipe_ids),
            "search_coverage": "budgeted_partial_frontier_with_lineage",
            "search_mode": "dynamic_batch_pareto_racing_main_thread_v2",
            "v1_compatibility": "bounded_exhaustive_main_thread_v1",
            "fallback": "disabled",
        }


__all__ = [
    "CompileIQOpaqueUnavailableError",
]
