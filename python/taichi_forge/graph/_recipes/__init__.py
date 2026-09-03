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
    GraphRuntimeAssemblyProvider,
    assemble_existing_family_recipe,
    default_graph_recipe_providers,
    materialize_existing_family_baseline,
)
from .graph_memory import GraphMemoryRecipeProvider

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
from .providers import (
    GraphRecipeProvider,
    GraphRecipeProviderDescriptor,
    GraphRecipeProviderError,
    GraphRecipeProviderSet,
    PROVIDER_OWNED_WHOLE_GRAPH_V1,
    RUNTIME_GRAPH_ASSEMBLY_V1,
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
    "GraphRuntimeAssemblyProvider",
    "GraphMaterializationContext",
    "GraphMaterializationError",
    "GraphMaterializationProduct",
    "GraphMaterializationScope",
    "GraphMaterializedAllocation",
    "GraphMaterializedFragment",
    "GraphMaterializedRecipe",
    "GraphMemoryRecipeProvider",
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
    "GraphRecipeProvider",
    "GraphRecipeProviderDescriptor",
    "GraphRecipeProviderError",
    "GraphRecipeProviderSet",
    "GraphRegionSelection",
    "GraphSemanticRegion",
    "PROVIDER_OWNED_WHOLE_GRAPH_V1",
    "RUNTIME_GRAPH_ASSEMBLY_V1",
    "assemble_existing_family_recipe",
    "default_graph_recipe_providers",
    "materialize_existing_family_baseline",
    "observe_graph_physical_manifest",
]
