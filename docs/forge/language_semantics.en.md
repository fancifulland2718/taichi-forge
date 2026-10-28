# Branch variable scope

[中文](language_semantics.zh.md) · [Documentation](index.en.md)

FG-LANG-01 is an unreleased enhancement. Published 0.7.0 still requires defining
variables before runtime branches. Current source permits a new variable defined
in both runtime `if/else` branches to be read afterwards:

```python
@ti.kernel
def choose(flag: ti.i32) -> ti.i32:
    if flag:
        value = 1
    else:
        value = 2
    return value
```

- Both branches must define the name. Nested branches follow the same rule.
  One-sided assignments, loop-only definitions and reads of the opposite
  branch's locals remain errors. Early returns do not relax this requirement.
- Types and shapes must match exactly; there is no implicit cross-branch type
  promotion. Scalars, vectors, matrices, structs with matching members/methods,
  and lists/tuples of these runtime values are supported.
- Common new runtime names merge even if unused afterwards. Rename differently
  typed branch temporaries or explicitly initialize a common variable before the
  conditional. Assignments to existing outer variables keep their previous rules.
- `ti.static` conditionals and static Python objects retain their block scope.
- Values are written to shared local storage on their respective execution paths;
  the unselected branch is not additionally evaluated. Existing AD handles the local storage.

The contract follows [Triton's all-path assignment rule](https://triton-lang.org/main/python-api/triton-semantics.html)
and [Warp's branch merging implementation](https://github.com/NVIDIA/warp/blob/main/warp/_src/codegen.py).
This is a language enhancement, not a defect in upstream Taichi's block scope.
