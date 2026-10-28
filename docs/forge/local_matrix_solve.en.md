# Local dense matrix solves

[中文](local_matrix_solve.zh.md) · [Documentation](index.en.md)

This page describes unreleased FG-MATH-01. Published 0.7.0 restricts `ti.solve`
to dimensions 2/3 and `Matrix.inverse()` to dimensions 1–4.

Focused validation covers dimensions 1, 5, 9, 16 and 32 in f32/f64 on CPU/CUDA,
the same dimensions in f32 on Vulkan, and CPU dimension 64 in f32. Results are
compared with NumPy solutions/inverses and normalized residuals, including row
swaps for a zero first pivot. A CUDA dimension-64 probe did not finish compilation
within a 90-second bound; an earlier long run already consumed substantial compile
time and host memory. That GPU size is not qualified here and is not artificially rejected.

Current source accepts positive fixed square dimensions for `ti.solve(A, b, dt=...)`
with a matching vector, and fixed square matrices for `A.inverse()`. **There is
no fixed upper dimension limit such as 16.** Existing 2/3 solves and 1–4 inverse
formulas are preserved; additional dimensions use partial-pivot Gaussian elimination.

```python
@ti.kernel
def solve_block(a: ti.types.matrix(9, 9, ti.f64),
                b: ti.types.vector(9, ti.f64)) -> ti.types.vector(9, ti.f64):
    return ti.solve(a, b, dt=ti.f64)
```

- New solve paths use f32/f64. Omitted `dt` still uses `default_fp`, not inference
  from A. Inverse preserves floating input precision and promotes integer inputs.
  The backend must support the requested precision.
- Pivot search, elimination and substitution use runtime loops, avoiding cubic
  static expansion. They stay serial within each thread and can be called inside
  an outer parallel loop. Inverse processes all identity columns in one elimination.
- Inputs must be finite and nonsingular. No regularization or fixed pivot epsilon
  is applied. A zero pivot asserts in debug mode. New paths return NaNs in non-debug
  mode with `fast_math=False`; fast math gives no singular-result guarantee.
  The new Python-scope inverse path raises an assertion.
- Qualification covers primal numerical solves, not AD through new dimensions,
  arbitrary ill-conditioned systems or a throughput guarantee.

Dimension support does not make arbitrarily large per-thread matrices efficient.
Input expansion and local storage grow quadratically; elimination work grows
cubically. Larger matrices can increase compilation cost and device local-memory
traffic. For runtime-variable dimensions or large systems, use ndarrays and a
structure-appropriate [LinearOperator / SolvePlan](linear_operator.en.md). CUDA
dense SPD systems also have an explicit [cuSOLVERDn provider](external_hardware_providers.en.md).
Their backend/structure contracts still apply; this enhancement is not a general
dynamic-dimension dense LU provider.
