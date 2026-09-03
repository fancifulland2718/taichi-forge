"""Staged catalog for complete Graph recipes without Cartesian expansion."""

from dataclasses import dataclass
from taichi_forge.graph._recipes.composer import (
    GraphExecutableRecipe,
    GraphRecipeComposer,
)
from taichi_forge.graph._recipes.fragments import GraphRecipeFragment
from taichi_forge.graph._recipes.providers import (
    GraphRecipeProvider as GraphFragmentProvider,
    GraphRecipeProviderError,
    GraphRecipeProviderSet,
)


@dataclass(frozen=True)
class GraphRecipeCatalogEntry:
    """One admitted complete recipe and its staged-search lineage."""

    stage: str
    recipe: GraphExecutableRecipe
    parent_recipe_ids: tuple[str, ...] = ()

    def to_dict(self):
        return {
            "stage": self.stage,
            "recipe": self.recipe.to_dict(),
            "parent_recipe_ids": self.parent_recipe_ids,
        }


@dataclass(frozen=True)
class GraphExactRecipeEnumeration:
    """Result of a bounded proof attempt over the provider recipe domain."""

    exhaustive: bool
    recipe_ids: tuple[str, ...]
    reason: str


class GraphRecipeCatalog:
    """Build complete recipes in explicit stages and deduplicate physical plans."""

    def __init__(
        self,
        definition,
        *,
        available_capabilities=(),
        provider_set=None,
    ):
        self.definition = definition
        if provider_set is not None and not isinstance(
            provider_set,
            GraphRecipeProviderSet,
        ):
            raise TypeError(
                "Graph recipe catalog provider_set must be GraphRecipeProviderSet"
            )
        if provider_set is not None and provider_set.definition is not definition:
            raise ValueError(
                "Graph recipe catalog provider set belongs to another definition"
            )
        self.provider_set = provider_set
        self.composer = GraphRecipeComposer(
            definition,
            available_capabilities=available_capabilities,
        )
        self._fragments = {}
        self._entries = {}
        self._ordered_recipe_ids = []
        self._planned_physical_representatives = {}
        self._physical_duplicates = {}
        baseline = self.composer.compose()
        self._admit("baseline", baseline, ())

    @property
    def baseline(self):
        return self._entries[self.definition.baseline_recipe.recipe_id]

    @property
    def fragments(self):
        return tuple(
            self._fragments[fragment_id]
            for fragment_id in sorted(
                self._fragments,
                key=lambda item: item.encode("utf-8"),
            )
        )

    @property
    def physical_duplicates(self):
        return dict(self._physical_duplicates)

    def entries(self, *, stage=None):
        entries = tuple(
            self._entries[recipe_id] for recipe_id in self._ordered_recipe_ids
        )
        if stage is None:
            return entries
        return tuple(entry for entry in entries if entry.stage == stage)

    def entry(self, recipe_id):
        representative = self._physical_duplicates.get(recipe_id, recipe_id)
        return self._entries[representative]

    def resolve(self, recipe_id):
        """Re-resolve one admitted recipe through this catalog's provider set."""

        entry = self.entry(recipe_id)
        recipe = entry.recipe
        if not recipe.fragments:
            return recipe
        if self.provider_set is None:
            raise GraphRecipeProviderError(
                "Graph recipe resolution requires the catalog provider set",
                error_key="provider_registry_missing",
            )
        resolved = tuple(
            self.provider_set.resolve(fragment) for fragment in recipe.fragments
        )
        canonical = self.composer.compose(resolved)
        if canonical.to_dict() != recipe.to_dict():
            raise GraphRecipeProviderError(
                "Graph recipe changed while resolving stable fragment keys",
                error_key="recipe_resolution_drift",
            )
        return canonical

    def register_fragment(self, fragment):
        if not isinstance(fragment, GraphRecipeFragment):
            raise TypeError("Graph recipe catalog accepts GraphRecipeFragment values")
        if self.provider_set is not None:
            self.provider_set.validate_fragment(fragment)
        previous = self._fragments.get(fragment.fragment_id)
        if previous is not None and previous != fragment:
            raise ValueError("Graph fragment ID collides with a different fragment")
        self.composer.compose((fragment,))
        self._fragments[fragment.fragment_id] = fragment
        return fragment

    @property
    def provider_registry_id(self):
        return (
            "" if self.provider_set is None else self.provider_set.provider_registry_id
        )

    @property
    def generation_domain_id(self):
        return (
            "" if self.provider_set is None else self.provider_set.generation_domain_id
        )

    def discover(self, providers=None):
        if providers is not None:
            requested = GraphRecipeProviderSet(
                self.definition,
                providers,
                available_capabilities=self.composer.available_capabilities,
            )
            if (
                self.provider_set is not None
                and requested.provider_registry_id
                != self.provider_set.provider_registry_id
            ):
                raise GraphRecipeProviderError(
                    "Graph recipe catalog provider set cannot change after creation",
                    error_key="provider_registry_drift",
                )
            self.provider_set = requested
        if self.provider_set is None:
            raise GraphRecipeProviderError(
                "Graph recipe catalog requires an explicit provider set",
                error_key="provider_registry_missing",
            )
        discovered = []
        for fragment in self.provider_set.discover():
            discovered.append(self.register_fragment(fragment))
        return tuple(discovered)

    def _admit(self, stage, recipe, parent_recipe_ids):
        existing = self._entries.get(recipe.recipe_id)
        if existing is not None:
            if existing.recipe != recipe:
                raise ValueError(
                    "Graph recipe ID collides with different recipe contents"
                )
            return existing
        representative_id = self._planned_physical_representatives.get(
            recipe.planned_physical_id
        )
        if representative_id is not None:
            self._physical_duplicates[recipe.recipe_id] = representative_id
            return self._entries[representative_id]
        entry = GraphRecipeCatalogEntry(
            stage=stage,
            recipe=recipe,
            parent_recipe_ids=tuple(parent_recipe_ids),
        )
        self._entries[recipe.recipe_id] = entry
        self._ordered_recipe_ids.append(recipe.recipe_id)
        self._planned_physical_representatives[recipe.planned_physical_id] = (
            recipe.recipe_id
        )
        return entry

    def compose(self, fragment_ids, *, stage, parent_recipe_ids=()):
        fragments = tuple(self._fragments[fragment_id] for fragment_id in fragment_ids)
        recipe = self.composer.compose(fragments)
        return self._admit(stage, recipe, parent_recipe_ids)

    def build_single_region_stage(self):
        """Admit one complete recipe per fragment; baseline fills every gap."""

        entries = []
        baseline_id = self.definition.baseline_recipe.recipe_id
        recipes = tuple(
            sorted(
                (self.composer.compose((fragment,)) for fragment in self.fragments),
                key=lambda item: item.recipe_id.encode("utf-8"),
            )
        )
        for recipe in recipes:
            entries.append(self._admit("single-region", recipe, (baseline_id,)))
        return tuple(entries)

    def build_exact_stage(self, *, candidate_limit):
        """Enumerate the complete reachable domain only when it is provably small.

        Neighbor expansion is explored to a fixed point before compatible
        fragment subsets are composed.  The catalog is left unchanged when the
        proof exceeds ``candidate_limit`` so callers can fall back to staged
        generation without leaking a partial exact-domain probe.
        """

        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("Graph exact candidate limit must be an integer")
        if candidate_limit < 1:
            raise ValueError("Graph exact candidate limit must be positive")
        if self.provider_set is None:
            raise GraphRecipeProviderError(
                "Graph exact enumeration requires the catalog provider set",
                error_key="provider_registry_missing",
            )

        fragments = {item.fragment_id: item for item in self.fragments}
        pending = list(self.fragments)
        expanded = set()
        while pending:
            current = pending.pop(0)
            if current.fragment_id in expanded:
                continue
            expanded.add(current.fragment_id)
            for neighbor in self.provider_set.expand(current):
                if neighbor.fragment_id in fragments:
                    if fragments[neighbor.fragment_id] != neighbor:
                        raise ValueError(
                            "Graph fragment ID collides during exact expansion"
                        )
                    continue
                fragments[neighbor.fragment_id] = neighbor
                pending.append(neighbor)
                pending.sort(key=lambda item: item.fragment_id.encode("utf-8"))
                # Every valid fragment creates at least one complete singleton
                # recipe, in addition to the baseline.
                if len(fragments) + 1 > candidate_limit:
                    return GraphExactRecipeEnumeration(
                        False,
                        (),
                        "exact_candidate_limit_exceeded",
                    )

        from taichi_forge.graph._recipes.composer import (
            GraphRecipeCompositionError,
        )

        ordered_fragments = tuple(
            fragments[fragment_id]
            for fragment_id in sorted(
                fragments,
                key=lambda item: item.encode("utf-8"),
            )
        )
        recipes = {
            self.baseline.recipe.recipe_id: self.baseline.recipe,
        }
        overflow = False

        def extend(selected, start):
            nonlocal overflow
            if overflow:
                return
            for index in range(start, len(ordered_fragments)):
                candidate = selected + (ordered_fragments[index],)
                try:
                    recipe = self.composer.compose(candidate)
                except GraphRecipeCompositionError:
                    continue
                recipes.setdefault(recipe.recipe_id, recipe)
                if len(recipes) > candidate_limit:
                    overflow = True
                    return
                extend(candidate, index + 1)
                if overflow:
                    return

        extend((), 0)
        if overflow:
            return GraphExactRecipeEnumeration(
                False,
                (),
                "exact_candidate_limit_exceeded",
            )

        for fragment in ordered_fragments:
            self.register_fragment(fragment)
        for recipe_id in sorted(recipes, key=lambda item: item.encode("utf-8")):
            if recipe_id == self.baseline.recipe.recipe_id:
                continue
            self._admit("exact-compatible", recipes[recipe_id], ())
        return GraphExactRecipeEnumeration(
            True,
            tuple(entry.recipe.recipe_id for entry in self.entries()),
            "exact_domain_enumerated",
        )

    def build_survivor_stage(
        self,
        survivor_recipe_ids,
        *,
        seed_fragment_ids,
        candidate_limit,
    ):
        """Generate bounded new physical recipes from one measured frontier."""

        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("Graph survivor candidate limit must be an integer")
        if candidate_limit < 0:
            raise ValueError("Graph survivor candidate limit must be non-negative")
        if candidate_limit == 0:
            return ()

        from taichi_forge.graph._recipes.composer import (
            GraphRecipeCompositionError,
        )

        survivor_ids = tuple(
            sorted(
                dict.fromkeys(survivor_recipe_ids),
                key=lambda item: item.encode("utf-8"),
            )
        )
        survivors = tuple(
            (recipe_id, self.entry(recipe_id)) for recipe_id in survivor_ids
        )
        seeds = tuple(
            self._fragments[fragment_id]
            for fragment_id in sorted(
                dict.fromkeys(seed_fragment_ids),
                key=lambda item: item.encode("utf-8"),
            )
        )
        candidates = {}
        physical_candidates = {}

        def offer(fragments, parents, stage):
            try:
                recipe = self.composer.compose(tuple(fragments))
            except GraphRecipeCompositionError:
                return
            if recipe.recipe_id in self._entries:
                return
            if recipe.planned_physical_id in self._planned_physical_representatives:
                return
            previous_id = physical_candidates.get(recipe.planned_physical_id)
            if previous_id is not None:
                if previous_id.encode("utf-8") <= recipe.recipe_id.encode("utf-8"):
                    return
                candidates.pop(previous_id, None)
            physical_candidates[recipe.planned_physical_id] = recipe.recipe_id
            candidates[recipe.recipe_id] = (
                recipe,
                tuple(sorted(set(parents), key=lambda item: item.encode("utf-8"))),
                stage,
            )

        for left_index, (left_id, left) in enumerate(survivors):
            for right_id, right in survivors[left_index + 1 :]:
                by_id = {
                    fragment.fragment_id: fragment
                    for fragment in (*left.recipe.fragments, *right.recipe.fragments)
                }
                offer(by_id.values(), (left_id, right_id), "survivor-merge")

        for survivor_id, survivor in survivors:
            selected = {
                fragment.fragment_id: fragment for fragment in survivor.recipe.fragments
            }
            for seed in seeds:
                if seed.fragment_id in selected:
                    continue
                offer(
                    (*selected.values(), seed),
                    (survivor_id,),
                    "survivor-addition",
                )

        if self.provider_set is not None:
            for survivor_id, survivor in survivors:
                selected = survivor.recipe.fragments
                for current in selected:
                    for neighbor in self.provider_set.expand(current):
                        by_id = {
                            fragment.fragment_id: fragment
                            for fragment in selected
                            if fragment.fragment_id != current.fragment_id
                        }
                        by_id[neighbor.fragment_id] = neighbor
                        offer(
                            by_id.values(),
                            (survivor_id,),
                            "neighbor-expansion",
                        )

        admitted = []
        for recipe_id in sorted(candidates, key=lambda item: item.encode("utf-8"))[
            :candidate_limit
        ]:
            recipe, parents, stage = candidates[recipe_id]
            for fragment in recipe.fragments:
                self.register_fragment(fragment)
            before = len(self._ordered_recipe_ids)
            entry = self._admit(stage, recipe, parents)
            if len(self._ordered_recipe_ids) != before:
                admitted.append(entry)
        return tuple(admitted)

    def build_compatible_stage(self, *, candidate_limit):
        """Admit budget-bounded multi-fragment complete recipes.

        The limit is supplied by the search budget, not by a family-specific
        heuristic.  Compatibility is checked incrementally so overlapping
        alternatives never inflate the materialized search domain.
        """

        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("Graph compatible candidate limit must be an integer")
        if candidate_limit < 0:
            raise ValueError("Graph compatible candidate limit must be non-negative")
        if candidate_limit == 0:
            return ()

        from taichi_forge.graph._recipes.composer import (
            GraphRecipeCompositionError,
        )

        fragments = tuple(
            sorted(self._fragments.values(), key=lambda item: item.fragment_id)
        )
        singleton_by_fragment = {
            entry.recipe.fragments[0].fragment_id: entry.recipe.recipe_id
            for entry in self.entries(stage="single-region")
            if len(entry.recipe.fragments) == 1
        }
        admitted = []

        def extend(selected, start):
            if len(admitted) >= candidate_limit:
                return
            for index in range(start, len(fragments)):
                candidate = selected + (fragments[index],)
                try:
                    recipe = self.composer.compose(candidate)
                except GraphRecipeCompositionError:
                    continue
                if len(candidate) >= 2:
                    parent_recipe_ids = tuple(
                        singleton_by_fragment[item.fragment_id] for item in candidate
                    )
                    before = len(self._ordered_recipe_ids)
                    entry = self._admit(
                        "compatible-composition",
                        recipe,
                        parent_recipe_ids,
                    )
                    if len(self._ordered_recipe_ids) != before:
                        admitted.append(entry)
                        if len(admitted) >= candidate_limit:
                            return
                extend(candidate, index + 1)
                if len(admitted) >= candidate_limit:
                    return

        extend((), 0)
        return tuple(admitted)

    def compose_compatible(self, fragment_groups, *, parent_recipe_ids=()):
        """Compose only caller-selected survivor groups, never a full product."""

        return tuple(
            self.compose(
                tuple(fragment_ids),
                stage="compatible-composition",
                parent_recipe_ids=parent_recipe_ids,
            )
            for fragment_ids in fragment_groups
        )

    def expand_neighbors(self, recipe_id):
        """Replace one selected fragment at a time with provider-owned neighbors."""

        if self.provider_set is None:
            raise GraphRecipeProviderError(
                "Graph neighbor expansion requires the catalog provider set",
                error_key="provider_registry_missing",
            )
        parent = self.entry(recipe_id)
        entries = []
        selected = parent.recipe.fragments
        for current in selected:
            for neighbor in self.provider_set.expand(current):
                self.register_fragment(neighbor)
                fragment_ids = tuple(
                    (
                        neighbor.fragment_id
                        if fragment.fragment_id == current.fragment_id
                        else fragment.fragment_id
                    )
                    for fragment in selected
                )
                entries.append(
                    self.compose(
                        fragment_ids,
                        stage="neighbor-expansion",
                        parent_recipe_ids=(recipe_id,),
                    )
                )
        return tuple(entries)
