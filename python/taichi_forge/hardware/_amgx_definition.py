"""Cold AmgX facts shared by direct creation and deployment reconstruction."""

from pathlib import Path

from taichi_forge.graph._native_definition import NativeDefinition


def solver_definition(rows, nonzeros, dtype, config, *, config_file, compute_residual):
    # Materialize a file once; both vendor creation and export use these exact
    # contents. A producer path or a later edit must not become deployment state.
    if config_file:
        config = Path(config).expanduser().read_text(encoding="utf-8-sig")
    if not isinstance(config, str) or not config or "\0" in config:
        raise ValueError("AmgX configuration must be nonempty text without NUL")
    return NativeDefinition(
        "amgx",
        "solver_csr",
        dict(rows=rows, nonzeros=nonzeros, dtype=dtype, config=config, compute_residual=compute_residual),
    )


def solve_definition(zero_initial_guess, retain_initial_guess):
    return NativeDefinition(
        "amgx",
        "solve",
        dict(initial_guess="zero" if zero_initial_guess else "upload", retain_initial_guess=retain_initial_guess),
        version=2,
    )
