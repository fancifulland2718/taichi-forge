"""Load Forge AOT kernels and ordinary CGraphs without the Python DSL.

Importing this module does not load a native library or initialize a device.
Use an explicit Runtime context manager to retire modules and arrays safely.
"""

from ._runtime import AotError, BoundCall, Graph, Kernel, Module, NdArray, PreparedCall, Runtime, Scalar
from ._plan import ExecutionPlan

__all__ = ["AotError", "Runtime", "Module", "NdArray", "Kernel", "Graph", "BoundCall", "PreparedCall", "Scalar", "ExecutionPlan"]
