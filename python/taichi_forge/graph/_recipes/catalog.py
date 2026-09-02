"""Staged catalog for complete Graph recipes without Cartesian expansion."""

from dataclasses import dataclass
from typing import Protocol

from taichi_forge.graph._recipes.composer import (
    GraphExecutableRecipe,
    GraphRecipeComposer,
)
from taichi_forge.graph._recipes.fragments import GraphRecipeFragment


class GraphFragmentProvider(Protocol):
    """Minimal provider contract owned by Forge, not by CompileIQ."""

    def fragments(self, definition) -> tuple[GraphRecipeFragment, ...]: ...


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

    def __init__(self, definition, *, available_capabilities=()):
        self.definition = definition
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
        previous = self._fragments.get(fragment.fragment_id)
        if previous is not None and previous != fragment:
            raise ValueError("Graph fragment ID collides with a different fragment")
        self.composer.compose((fragment,))
        self._fragments[fragment.fragment_id] = fragment
        return fragment

    def discover(self, providers):
        discovered = []
        for provider in providers:
            for fragment in provider.fragments(self.definition):
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

        parent = self.entry(recipe_id)
        entries = []
        selected = parent.recipe.fragments
        for current in selected:
            for neighbor in current.neighbors():
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
