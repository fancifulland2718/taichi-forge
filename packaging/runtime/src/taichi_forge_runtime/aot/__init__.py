"""Load Forge AOT kernels and ordinary CGraphs without the Python DSL.

Importing this module does not load a native library or initialize a device.
Use an explicit Runtime context manager to retire modules and arrays safely.
"""

from ._runtime import AotError, BoundCall, Graph, Kernel, Module, NativeResource, NdArray, PreparedCall, Runtime, Scalar
from ._plan import ExecutionPlan
from ._completion import SubmissionTicket
from ._workspace import WorkspaceBusyError, WorkspaceLease, WorkspacePool
from ._domain import Completion, CudaExecutionDomain

__all__ = ["AotError", "Runtime", "Module", "NativeResource", "NdArray", "Kernel", "Graph", "BoundCall", "PreparedCall", "Scalar", "ExecutionPlan", "SubmissionTicket", "WorkspaceBusyError", "WorkspaceLease", "WorkspacePool", "Completion", "CudaExecutionDomain"]
