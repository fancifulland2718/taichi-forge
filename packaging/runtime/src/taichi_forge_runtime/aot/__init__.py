"""Load Forge AOT kernels and ordinary CGraphs without the Python DSL.

Importing this module does not load a native library or initialize a device.
Use an explicit Runtime context manager to retire modules and arrays safely.
"""

from ._runtime import AotError, BoundCall, Graph, Kernel, Module, NativeResource, NdArray, PreparedCall, Runtime, Scalar
from ._plan import ExecutionPlan
from ._cuda_context import CudaVulkanContextBinding, configure_cuda_context, cuda_context_info
from ._external_ready import CudaExternalReadyGate
from ._completion import SubmissionTicket
from ._wait_set import CompletionSet, CompletionWaitResult
from ._workspace import WorkspaceBusyError, WorkspaceLease, WorkspacePool
from ._domain import Completion, ExecutionDomain, CudaExecutionDomain, VulkanExecutionDomain
from ._struct import StructLayout, StructMember

__all__ = ["AotError", "Runtime", "Module", "NativeResource", "NdArray", "Kernel", "Graph", "BoundCall", "PreparedCall", "Scalar", "ExecutionPlan", "SubmissionTicket", "WorkspaceBusyError", "WorkspaceLease", "WorkspacePool", "Completion", "ExecutionDomain", "CudaExecutionDomain", "VulkanExecutionDomain"]
__all__ += ["CudaVulkanContextBinding", "configure_cuda_context", "cuda_context_info"]
__all__ += ["CudaExternalReadyGate"]
__all__ += ["StructLayout", "StructMember"]
__all__ += ["CompletionSet", "CompletionWaitResult"]
