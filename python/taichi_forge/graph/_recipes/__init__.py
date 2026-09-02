"""Internal complete-Graph recipe foundations."""

from .catalog import (
    GraphFragmentProvider,
    GraphRecipeCatalog,
    GraphRecipeCatalogEntry,
)
from .composer import (
    GraphExecutableRecipe,
    GraphRecipeComposer,
    GraphRecipeCompositionError,
    GraphRecipeExecutionStep,
    GraphRegionSelection,
)
from .definition import (
    GraphBaselineRecipe,
    GraphBindingABIEntry,
    GraphCompileProvenance,
    GraphDefinition,
    GraphDefinitionSource,
    GraphSemanticRegion,
)
from .fragments import (
    GraphFragmentBindingRequirement,
    GraphFragmentResourceRequirement,
    GraphFragmentSubmissionRequirement,
    GraphFragmentTask,
    GraphRecipeFragment,
)
from .families import (
    GraphExistingFamilyProvider,
    GraphFamilySelection,
    assemble_existing_family_recipe,
    materialize_existing_family_baseline,
)

from .materialize import (
    GraphMaterializationContext,
    GraphMaterializationError,
    GraphMaterializationProduct,
    GraphMaterializationScope,
    GraphMaterializedAllocation,
    GraphMaterializedFragment,
    GraphMaterializedRecipe,
)

from .physical import (
    CompiledGraphPhysicalManifest,
    GraphPhysicalBindingManifest,
    GraphPhysicalCommandManifest,
    GraphPhysicalKernelManifest,
    GraphPhysicalManifestError,
    GraphPhysicalResourceManifest,
    GraphPhysicalSubmissionManifest,
    GraphPhysicalTaskManifest,
    observe_graph_physical_manifest,
)

__all__ = [
    "CompiledGraphPhysicalManifest",
    "GraphBaselineRecipe",
    "GraphBindingABIEntry",
    "GraphCompileProvenance",
    "GraphDefinition",
    "GraphDefinitionSource",
    "GraphExecutableRecipe",
    "GraphFragmentBindingRequirement",
    "GraphFragmentProvider",
    "GraphFragmentResourceRequirement",
    "GraphFragmentSubmissionRequirement",
    "GraphFragmentTask",
    "GraphExistingFamilyProvider",
    "GraphFamilySelection",
    "GraphMaterializationContext",
    "GraphMaterializationError",
    "GraphMaterializationProduct",
    "GraphMaterializationScope",
    "GraphMaterializedAllocation",
    "GraphMaterializedFragment",
    "GraphMaterializedRecipe",
    "GraphPhysicalBindingManifest",
    "GraphPhysicalCommandManifest",
    "GraphPhysicalKernelManifest",
    "GraphPhysicalManifestError",
    "GraphPhysicalResourceManifest",
    "GraphPhysicalSubmissionManifest",
    "GraphPhysicalTaskManifest",
    "GraphRecipeCatalog",
    "GraphRecipeCatalogEntry",
    "GraphRecipeComposer",
    "GraphRecipeCompositionError",
    "GraphRecipeExecutionStep",
    "GraphRecipeFragment",
    "GraphRegionSelection",
    "GraphSemanticRegion",
    "assemble_existing_family_recipe",
    "materialize_existing_family_baseline",
    "observe_graph_physical_manifest",
]
