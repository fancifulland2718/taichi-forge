"""Opt-in budgeted trials for compact, provider-owned offload plan domains."""

from dataclasses import asdict
from types import MappingProxyType

from taichi_forge._compileiq_opaque import _identity, _submit_recipe_pages


class _OffloadPlanSearchSessionV2:
    PROTOCOL = "dynamic_batch_pareto_racing_main_thread_v2"

    def __init__(self, plans, objective, *, budget, problem_type, checkpoint, evaluation_context, repeat_count):
        from compileiq import forge_support as support

        if not callable(objective):
            raise TypeError("objective_function must be callable")
        if problem_type not in ("min", "max"):
            raise ValueError("problem_type must be 'min' or 'max'")
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 1:
            raise ValueError("repeat_count must be a positive integer")
        if isinstance(budget, dict):
            budget = support.ForgeOpaqueSearchBudgetV2(**budget)
        if checkpoint is not None and getattr(evaluation_context, "reuse_scope", None) != "portable":
            raise ValueError("resuming offload trials requires an explicit portable evaluation_context")
        self._plans, self._objective = plans, objective
        self._repeat_count = repeat_count
        self._support = support
        self._session = plans._transport.search_session_v2(
            self._evaluate,
            budget=budget,
            checkpoint=checkpoint,
            evaluation_context=evaluation_context,
            target_contract=support.ForgeOpaqueTargetContractV1(
                objectives=(support.ForgeOpaqueObjectiveV1(name="score", direction=problem_type),)
            ),
        )

    def _evaluate(self, request):
        plans, support = self._plans, self._support
        parameters = {"domain_fingerprint": plans.domain_fingerprint, "recipe_id": request.recipe_id}
        cleanup = support.TrialCleanupV2(
            status="complete",
            released_resources=True,
            detail_code="trial_wrapper_released_program_cache_retained",
        )
        provenance = {
            "compileiq_python_source_lock": plans._transport.python_source_lock,
            "memory_scope": "trial_owned_device_allocations_only_excludes_program_caches_and_caller_storage",
            "identity_scope": "validated_compiled_task_manifest_and_complete_plan",
        }
        phase = "materialization"
        try:
            # The native plan remains Program-owned, as for ordinary explicit
            # plans. This adapter owns no arrays/workspace or execution queue.
            bound = plans.bind(parameters)
            report = bound.report(*plans._sample_args)
            facts = tuple(
                {key: value for key, value in asdict(task).items() if key not in ("task_id", "task_name")}
                for task in report.tasks
            )
            observed = _identity(
                "forge-observed-offload-plan-v1:",
                {
                    "plan": bound.plan.identity,
                    "backend": report.backend,
                    "tasks": facts,
                },
            )
            phase = "objective"
            score = self._objective(parameters)
            phase = "protocol"
            return support.TrialOutcomeV2(
                planned_physical_id=_identity("opaque-planned-recipe-v1:", request.recipe_id),
                materialized_physical_id=observed,
                metrics={"score": score},
                materialized_memory_bytes=0,
                provenance=provenance,
                cleanup=cleanup,
            )
        except Exception as error:
            return support.TrialOutcomeV2(
                planned_physical_id=_identity("opaque-planned-recipe-v1:", request.recipe_id),
                provenance=provenance,
                cleanup=cleanup,
                failure=support.TrialFailureV2(
                    category=phase, code=type(error).__name__, message=str(error)[:1000] or type(error).__name__
                ),
            )

    def _pages(self):
        transport = self._plans._transport
        group = _identity(
            "forge-offload-pages-v1:",
            {
                "semantic_fingerprint": self._plans.semantic_fingerprint,
                "repeat_count": self._repeat_count,
            },
        )
        pages = iter(transport.recipe_pages())
        current, following = next(pages), next(pages, None)
        paged = following is not None
        index, previous = 0, None
        while True:
            batch = transport.batch_v2(
                recipe_ids=current,
                stage_index=index,
                stage_fingerprint=_identity("forge-offload-page-v1:", (group, index, current)),
                parent_batch=previous,
                parent_recipe_ids={},
                fidelity_name="full",
                fidelity_ordinal=0,
                repeat_count=self._repeat_count,
                terminal=True,
                page_group_fingerprint=group if paged else None,
                page_index=index,
                page_final=following is None,
            )
            yield batch
            if following is None:
                return
            previous, index = batch, index + 1
            current, following = following, next(pages, None)

    def start(self):
        if self.checkpoint().finalization is not None:
            return self.result()
        result = _submit_recipe_pages(self._session, self._pages())
        checkpoint = self.checkpoint()
        if checkpoint.stages[-1].complete and getattr(checkpoint.batches[-1], "page_final", True):
            return self._session.finalize(
                self._support.ForgeOpaqueSearchFinalizationV1(
                    generation_status="exhaustive",
                    terminal_fidelity_status="complete",
                    reason="complete_offload_plan_domain",
                )
            )
        return result

    def checkpoint(self):
        return self._session.checkpoint()

    def result(self):
        return self._session.result()

    @property
    def status(self):
        return self._session.status

    @property
    def opaque_recipe_capability(self):
        return self._session.opaque_recipe_capability

    @property
    def opaque_recipe_core_provenance(self):
        return self._session.opaque_recipe_core_provenance

    def coverage(self, plans):
        if plans is not self._plans:
            raise ValueError("offload search belongs to another frozen plan domain")
        checkpoint = self.checkpoint()
        observed = set(checkpoint.stages[-1].evaluated_recipe_ids) if checkpoint.stages else set()
        status = self.status
        return MappingProxyType(
            {
                "complete": status.generation_status == "exhaustive"
                and status.evaluation_status == "complete"
                and status.terminal_fidelity_status == "complete",
                "baseline_observed": status.baseline_status == "available",
                "evaluation_count": checkpoint.evaluation_count,
                "observed_recipe_ids": tuple(key for key in plans.recipe_ids if key in observed),
                "missing_recipe_ids": tuple(key for key in plans.recipe_ids if key not in observed),
                "verified_core": True,
            }
        )
