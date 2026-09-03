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
    GraphFamilySelection,
    GraphRuntimeAssemblyProvider,
    GraphRuntimeFragmentProvider,
    assemble_runtime_graph_recipe,
    default_graph_recipe_providers,
    materialize_runtime_graph_baseline,
)
from .runtime_assembly import GraphRuntimeRecipeAssembly
from .branch_join import GraphBranchJoinRecipeProvider
from .dispatch_families import (
    GraphOffloadPhaseFusionRecipeProvider,
    GraphSparseTraversalRecipeProvider,
)
from .graph_memory import GraphMemoryRecipeProvider
from .map_fusion import GraphMapFusionRecipeProvider
from .semantic_families import (
    GraphBoundedExecutionRecipeProvider,
    GraphNativeAlgorithmRecipeProvider,
    GraphReductionRecipeProvider,
    GraphStructuredControlRecipeProvider,
)
from .submission_families import (
    GraphRecordingPartitionRecipeProvider,
    GraphWorkspaceConcurrencyRecipeProvider,
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
    "GraphBranchJoinRecipeProvider",
    "GraphBoundedExecutionRecipeProvider",
    "GraphCompileProvenance",
    "GraphDefinition",
    "GraphDefinitionSource",
    "GraphExecutableRecipe",
    "GraphFragmentBindingRequirement",
    "GraphFragmentProvider",
    "GraphFragmentResourceRequirement",
    "GraphFragmentSubmissionRequirement",
    "GraphFragmentTask",
    "GraphFamilySelection",
    "GraphMapFusionRecipeProvider",
    "GraphNativeAlgorithmRecipeProvider",
    "GraphOffloadPhaseFusionRecipeProvider",
    "GraphRecordingPartitionRecipeProvider",
    "GraphReductionRecipeProvider",
    "GraphRuntimeAssemblyProvider",
    "GraphRuntimeFragmentProvider",
    "GraphRuntimeRecipeAssembly",
    "GraphSparseTraversalRecipeProvider",
    "GraphStructuredControlRecipeProvider",
    "GraphWorkspaceConcurrencyRecipeProvider",
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
    "assemble_runtime_graph_recipe",
    "default_graph_recipe_providers",
    "materialize_runtime_graph_baseline",
    "observe_graph_physical_manifest",
]
