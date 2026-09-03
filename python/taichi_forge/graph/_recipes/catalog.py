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
        return tuple(self._fragments.values())

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
            ""
            if self.provider_set is None
            else self.provider_set.provider_registry_id
        )

    @property
    def generation_domain_id(self):
        return (
            ""
            if self.provider_set is None
            else self.provider_set.generation_domain_id
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
        for fragment in self._fragments.values():
            entries.append(
                self.compose(
                    (fragment.fragment_id,),
                    stage="single-region",
                    parent_recipe_ids=(baseline_id,),
                )
            )
        return tuple(entries)

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
                    neighbor.fragment_id
                    if fragment.fragment_id == current.fragment_id
                    else fragment.fragment_id
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
