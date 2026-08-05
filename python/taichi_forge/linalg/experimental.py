"""Experimental solver execution APIs.

The runtime-bound operator model and its storage, composition, and
qualification helpers are public from ``taichi_forge.linalg``. Persistent
solver and preconditioner execution plans remain in this namespace.
"""

from taichi_forge.linalg._runtime import (
    BatchedSolvePlan,
    BatchedSolveResult,
    PreconditionerPlan,
    PreconditionerSession,
    SolveGraphTerminal,
    SolveGraphTerminalPacket,
    SolveGraphTerminalSnapshot,
    SolvePlan,
    SolveQualificationReport,
    SolveResult,
    SolveSubmission,
    qualify_solve_plan,
    summarize_solve_qualifications,
)

__all__ = [
    "BatchedSolvePlan",
    "BatchedSolveResult",
    "PreconditionerPlan",
    "PreconditionerSession",
    "SolveGraphTerminal",
    "SolveGraphTerminalPacket",
    "SolveGraphTerminalSnapshot",
    "SolvePlan",
    "SolveQualificationReport",
    "SolveResult",
    "SolveSubmission",
    "qualify_solve_plan",
    "summarize_solve_qualifications",
]
