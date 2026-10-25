"""Protocol facts for transport tests that intentionally do not import CompileIQ."""

import hashlib
import json
from types import MappingProxyType


SOURCE_LOCK = "ciq-python-source-v1:" + "e" * 64


def capability():
    payload = {
        "schema": "compileiq.taichi-forge-recipe-search-capability.v2",
        "protocol_revision": 7,
        "fork_build_id": "compileiq-taichi-forge-complete-recipes.v2",
        "package_version": "1.1.0b2+taichiforge.1",
        "opaque_recipe_domain_schema": "compileiq.opaque-recipe-domain.v1",
        "opaque_recipe_batch_schema": "compileiq.opaque-recipe-batch.v2",
        "opaque_dynamic_recipe_domain_schema": "compileiq.opaque-dynamic-recipe-domain.v2",
        "selection_audit_schema": "compileiq.opaque-recipe-selection.v1",
        "opaque_target_contract_schema": "compileiq.taichi-forge-opaque-target-contract.v1",
        "opaque_target_selection": "uncertainty_aware_pareto_layers_no_scalarization_v2",
        "trial_outcome_schema": "compileiq.taichi-forge-trial-outcome.v2",
        "search_checkpoint_schema": "compileiq.taichi-forge-search-checkpoint.v2",
        "evaluation_context_schema": "compileiq.taichi-forge-evaluation-context.v1",
        "search_finalization_schema": "compileiq.taichi-forge-search-finalization.v1",
        "search_status_schema": "compileiq.taichi-forge-search-status.v2",
        "optimization_report_schema": "compileiq.opaque-optimization-report.v1",
        "optimization_report_renderer": "json_fact_source_markdown_projection_v1",
        "max_recipe_ids": 4096,
        "max_field_utf8_bytes": 4096,
        "max_canonical_bytes": 4 * 1024 * 1024,
        "provider_recipe_ids_cross_core_boundary": False,
        "core_verification": "python_scheduler_source_digest_at_search_start",
        "execution_engine": "forge_complete_recipes_python_v1",
        "opaque_domain_binding": "capability_id_scheduler_source_lock",
        "objective_worker": "forge_main_thread_serial_v1",
        "opaque_recipe_search": "dynamic_batch_pareto_racing_main_thread_v2",
        "opaque_recipe_search_v1": "bounded_exhaustive_main_thread_v1",
        "core_manifest_schema_version": 1,
        "core_commit": "python:fixture-revision-7",
        "core_lock": "sha256:" + "d" * 64,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["capability_id"] = "ciq-forge-cap-v2:" + hashlib.sha256(encoded).hexdigest()
    return MappingProxyType(payload)
