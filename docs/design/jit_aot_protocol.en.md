# JIT/AOT protocol comparison and convergence plan

[中文](jit_aot_protocol.zh.md)

Date: 2026-11-12. Source audit baseline: Forge `7420977ca`.
Status: protocol comparison, convergence decisions and a minimal consistency-test selection;
implementation unification is not started. This describes development sources, not released-wheel qualification.

## 1. Scope

JIT and AOT should give consistent meanings to compilation semantics, function boundaries,
dependencies, artifacts and consumption. They may retain different implementations, caches,
storage and execution lifetimes. **Agreement does not require identical bytes, inlining policies
or build entry points.**

This phase introduces no general compiler framework, large IR, global Artifact Store or
cross-kernel frontend sharing. Ordinary JIT does not serialize and reload TCM files. No default
diagnostics or per-call protocol checks are added; existing necessary ABI, target and resource
validity checks retain their current placement. `jit_function_library` remains experimental
and off by default. Existing JIT and AOT consumption behavior remains unchanged.

The contract fields below are semantic projections of existing objects, not four new large
objects or registration APIs. Requirements and implemented behavior are distinguished;
documenting a requirement does not claim that a complete common interface already exists.

## 2. Alignment overview

| Contract | Existing shared foundation | Difference or gap | Decision |
| --- | --- | --- | --- |
| Compile Request | Owned configuration, target and provenance snapshots through `KernelCompileRequest` and its resolver | Purpose-dependent target defaults; static inputs, ABI and resources have other owners | Define the existing fields and defaults; do not put entire functions/models into the request |
| Function Contract | `FunctionLibrary` typed boundaries and value-copy behavior; native `Function` types/IR | `JitFunctionLibrary` and `PackageCompilation` differ in eligibility, dependency proofs, construction and retirement; no complete common effects/specialization record | Align meanings and fallback classes; retain separate proofs and construction |
| Artifact Contract | LLVM `FunctionCodeCache` and AOT code-unit export/linking; existing target, ABI and integrity checks | In-memory lookup keys, content hashes, component identities and specialization identities differ; SPIR-V has no matching cross-module library | Define identity layers and correspondence without merging caches or formats |
| Consumption Contract | Resource binding, prepared specialization, backend optimization and executables | Immediate JIT and AOT load/prepare have different lifetimes and backend-specific deferred work | Specify binding times, preserving those lifetimes |

## 3. Compile Request

Sources: [request](../../taichi/program/kernel_compile_request.h),
[resolver](../../taichi/program/kernel_compile_request.cpp),
[Python initialization](../../python/taichi_forge/lang/misc.py),
[AOT Module](../../python/taichi_forge/aot/module.py).

| Information | Meaning and current rule |
| --- | --- |
| Purpose | `CompilePurpose::{jit,aot}` selects applicable default policies; the label alone is not complete code identity |
| Target | `CompileConfig.arch` and resolved capabilities describe the effective target; provenance does not substitute for those values |
| Optimization | The snapshot records effective tier, advanced optimization, LLVM/external levels, fast math and simplification limits; equal tier names need not imply equal backend passes |
| Overrides | Resolved base configuration → kernel tier → AOT export options; export options are AOT-only. Explicit `False` and `0` differ from omission |
| Provenance | Purpose, tier source, normalization and option sources explain decisions. Differences they cause in effective values matter to identity; different origins alone do not |
| Static inputs | Existing template instances, function dependencies, package definitions and preparation records carry them; equal request configuration does not prove equal static inputs |
| Types/resources | Existing function, kernel and resource owners retain ABI, layout, dependencies and lifetimes; the request does not take ownership |

Preserve these defaults and distinctions:

- Without an explicit CUDA target, JIT resolves the current device's supported target; AOT defaults to SM80. Equal Python option strings do not establish compatibility.
- Explicit CUDA targets and PTX versions must agree. AOT does not silently choose another target and currently rejects targets below SM60.
- AOT removes producer-specific default launch hints (`gpu_compute_unit_count=0`, `gpu_subgroup_size=32`). Explicit task plans retain their geometry; JIT may keep current-device hints.
- Full tier normalizes the legacy default simplification cap from `1` to `0` only when not explicitly selected. An explicit `1` remains `1`.
- CPU/Vulkan retain supplied capability snapshots without acquiring CUDA requirements. Vulkan interprets SPIR-V optimization levels through its own backend policy.

Compare effective resolved values together with function, ABI and resource dependencies.
Resolution must not temporarily mutate the Program or caller capabilities. Discovery stays distinct:
JIT obtains specialization inputs from calls; AOT obtains definitions and dependencies from the
formal export scope, not from a single test's execution trace as a substitute for engine coverage.

## 4. Function Contract

Sources: [common boundary](../../python/taichi_forge/lang/_function_library.py),
[JIT policy](../../python/taichi_forge/lang/_jit_function_library.py),
[package transaction](../../python/taichi_forge/aot/_package_compilation.py),
[native Function](../../taichi/program/function.h).

| Field | Shared interpretation | Present boundary |
| --- | --- | --- |
| Arguments/results | Define order, dtype, scalar/aggregate shapes, value/reference/array passing and return structure; value-parameter mutation stays local | Kernel-entry ABI and internal function ABI may differ; entry RuntimeContext offsets cannot be applied to a callee's private context |
| Static dependencies | Types and actually code-affecting constants/dependencies constrain specialization; owner membership alone does not make ordinary data static | JIT dependency proofs and AOT definition/preparation classification differ. Not every static branch is deferrable; layout/initialization guards remain meaningful |
| Resources | Identify storage, layout, aliases and reads/writes; current Field contents/addresses are not generic persistent identity | JIT may use objects for local lookup; AOT uses package contracts and relocations/bindings. General resource-slot unification is incomplete |
| Effects | Preserve write order, atomics, aliasing, synchronization and parallel scope | Existing AST admission, IR analysis and resource contracts jointly describe effects; no complete common exported summary exists |
| Specialization | Separate production-fixed types/structure, facts current mechanisms may fix at prepare, and per-call data | JIT specializes immediately; AOT defers only what its format and dependency proof support |
| Optimization | Extraction/inlining/optimization may not change the contract | Automatic JIT LLVM helpers use late forced inlining; AOT selectively marks proven single-use helpers during specialization; Vulkan supplies Inline hints. These policies need not match |

Distinguish three outcomes:

1. A supported ABI and dependency proof permit an independent function boundary.
2. Extraction is ineligible but ordinary expansion is legal: JIT expands normally; AOT may
   expand during production and seal the result. An optimization miss is not whole-definition rejection.
3. The definition cannot be exported or consumed legally: reject at the relevant production,
   loading or preparation stage. Consumers do not fall back to private numerical source.
   A caller script's own JIT is separate from such a fallback.

Current automatic boundaries still restrict AD, unknown callbacks/attribute effects, unproven
dependencies and some return inference. Kernel-root parallel helpers must not become serial
per-thread functions merely to obtain reuse. SIMT/shared-memory/task-dependent functions cannot
be shared solely because their parameter types match. Explicit `ti.real_func` keeps its existing
backend support; automatic function-unit support does not extend it by implication. Eligibility
remains a compile-time decision, with no new warm-call checks.

## 5. Artifact Contract

Sources: [LLVM function cache](../../taichi/runtime/llvm/function_code_cache.h),
[AOT code units](../../taichi/runtime/llvm/aot_code_units.h),
[LLVM compiler](../../taichi/codegen/llvm/kernel_compiler.cpp),
[SPIR-V compiler](../../taichi/codegen/spirv/kernel_compiler.cpp).

### Identity and correspondence

| Layer | Required interpretation | Invalid shortcut |
| --- | --- | --- |
| Definition/specialization | Numerical definition, actual static dependencies, types/call ABI, relevant resource/layout constraints, effective configuration and target establish compatibility | A whole-component digest, example name or Python address alone cannot establish persistent equivalence |
| Compilation unit | Source identity, symbols/dependencies, backend/target, ABI, format and producer toolchain version must be recorded or traceable; outer metadata need not be duplicated in every unit | Equal Kernel semantic keys do not authorize cross-backend/ABI substitution |
| Content | `code_units/<sha256>.bc` hashes establish content/reference integrity; current reference metadata is `taichi.aot.code_units.v1` | Content hash alone is not full semantic/runtime compatibility; different bytes need not mean different mathematics |
| Final specialization | Refer to the original artifact and relevant target, configuration, constants, layout/alias facts and backend requirements | Do not implicitly bake instance addresses or mutable contents into shared code |
| Lifetime | Code may be shared while resources and in-flight leases remain independent | Equal code identity does not imply equal resource ownership or retirement |

LLVM `FunctionCodeCache` holds immutable symbol/bitcode entries under a Program compiler or
producer transaction. Lookup accounts for serialized function information, parameter/result
types, target context and binding constraints. Units without the required task-independence
or resource proof are not forced into reuse. The persistent JIT kernel cache still stores
self-contained modules. AOT `bc-library` persists units and references; consumption resolves
dependencies, checks integrity and target triple/data layout, then optimizes the linked result.

An existing connection from memory units to exportable compatible units does **not** establish
universal interchangeability between all memory and persistent artifacts. Python-local keys,
native cache keys and archive content hashes remain separate namespaces. There is no new
cross-process function cache. Toolchain/compiler version, runtime ABI and format version are
distinct invalidation domains.

### Backend differences

| Path | Current representation/reuse | Preserved limit |
| --- | --- | --- |
| CPU JIT/AOT | Program-local LLVM function reuse; ordinary AOT bitcode and prepared specialization | Public `artifact_format="bc-library"` is currently CUDA-only; sharing LLVM does not establish CPU format support |
| CUDA JIT/AOT | LLVM function units; AOT bitcode, experimental `bc-library`, PTX and cubin | Formats have different deferred-specialization/linking capabilities; PTX/cubin are not editable LLVM bitcode |
| Vulkan JIT/AOT | Automatic JIT helpers share within each task's SPIR-V module through `OpFunctionCall`; AOT SPIR-V has its existing consumption specialization | No cross-task/independent-module SPIR-V function cache. Automatic value boundaries support scalars, fixed vectors/matrices and template resources; array/ref parameters stay inline |

This is an implemented-path inventory, not qualification of every language feature. Selecting
a backend does not imply cross-arch Module compilation: current Module creation requires the
initialized arch to match the requested arch.

## 6. Consumption Contract

Sources: [specialization constants](../../taichi/aot/specialization.h),
[LLVM loader](../../taichi/runtime/llvm/llvm_aot_module_loader.cpp),
[Gfx loader](../../taichi/runtime/gfx/aot_module_loader_impl.cpp),
[Vulkan specialization](../../taichi/rhi/vulkan/shader_specialization.cpp),
[hybrid resource borrowing](../../python/taichi_forge/aot/_hybrid.py).

The common logical order is dependency resolution → ABI/target checks and resource binding →
fixing permitted specialization facts → backend optimization/executable generation → execution.
This is a dependency order, not a requirement for identical APIs or repeated preparation.

| Binding time | Information | Allowed differences |
| --- | --- | --- |
| Production/definition compilation | Numerical closure, types/call ABI, necessarily fixed mathematical structure/layout, target requirements and external symbol/provider contracts | JIT compiles the script's definitions; AOT seals source-free numerical artifacts. Unspecified types/branches are not left for consumer AST rediscovery |
| Consumption/prepare | Supported shapes, strides, explicitly selected integer constants, proven resource relations and launch policy; remaining linking/executable dependencies | JIT may do this immediately. AOT load/bind/prepare may be separate, and supported formats may defer backend generation. Check each format/backend rather than assuming universal deferral |
| Execution | Mutable model contents, ordinary runtime arguments, valid bindings and ordering | Content changes do not implicitly trigger recompilation. Prepared descriptors/leases remain valid; binding changes follow existing update/reprepare APIs |

Shared specialization caches hold code, not one instance's current pointers for another to
switch. Generations, close and in-flight retirement retain existing ownership/lease rules.
Runtime/provider dependencies need not be embedded in TCM, but must resolve under their contracts.

For matching inputs, effective configuration and supported capabilities, preserve language,
value-copy, resource-effect and control semantics. Declared floating-point tolerance may reflect
optimization/backend differences. Comparisons with different targets, fast-math settings or
static domains do not establish protocol disagreement. Identical machine code, preparation
latency, cache-hit counts and final sizes are not required.

## 7. Compatibility and invalidation

| Change | Required action |
| --- | --- |
| Purpose/origin label only, effective inputs equal | Do not manufacture numerical variants solely from the label; distinguish any effective target/geometry differences |
| Definition, actual static dependency, ABI or code-affecting layout | Recompile/select the affected compatible unit and recheck dependent callers; do not assume whole-engine invalidation |
| Target capabilities/effective optimization | Check the compatibility domain and select/rebuild affected artifacts; missing requirements reject, even for the same arch |
| Schema/format, compiler/LLVM or runtime ABI | Check their existing version mechanisms independently; reject unknown required versions rather than relying on wheel version/content hash alone |
| Missing/corrupt unit or wrong triple/layout | Fail at the relevant link/load boundary, without private-source fallback or mutation of the original archive |
| Prepared constants/shapes/proven alias relations | Select a new compatible specialization; ordinary data changes do not become static implicitly |
| Resource instance/generation | Rebind/reprepare as required and honor leases; compatible code may remain shared, mutable state may not |

These are common requirements, not a claim of one complete version manifest or effects summary
today. Future cross-path cache access requires proof that existing distributed fields define a
sufficient compatibility domain; this phase does not enable it. Do not add compatibility layers
for historical experimental build decisions, or remove still-used artifact readers in this
documentation phase.

## 8. Minimal consistency checks

Select existing tests by the contract changed; this is **not** a mandatory full matrix for every
document/module update. Use device-free request/identity tests first, and add backend/consumer
checks only where affected.

| Concern | Existing selection | Boundary |
| --- | --- | --- |
| Snapshot, overrides, defaults/origins | [kernel_compile_request_test.cpp](../../tests/cpp/program/kernel_compile_request_test.cpp): `OwnsConfigAndTargetWithoutChangingTheCaller`, `ResolvesCudaTargetsWithoutBorrowingProducerCapabilities`, `PortableAotDoesNotBorrowProducerLaunchHints`, `PurposeAndEquivalentOriginsDoNotChangeCodeIdentity`, `PreservesExplicitCapAndIgnoresOriginsInCodeIdentity`, `ExportOverridesAreFrozenValuesWithoutJitMutation` | Explicit False/0, purpose-dependent targets and effective identity; not whole-function equivalence |
| Effective request reaches backend | [test_aot_compile_request.py](../../tests/python/test_aot_compile_request.py): `test_aot_resolves_kernel_tier_before_codegen_and_cache_lookup` | Fast/full and ordinary/template arguments on the affected backend |
| Value/resource/fallback semantics | [test_func_inline_ir.py](../../tests/python/test_func_inline_ir.py): `test_jit_function_library_preserves_tensor_copies`, `test_jit_function_library_keeps_field_state_and_bindings`, `test_jit_function_library_tracks_nested_owner_and_static_dependencies`, `test_jit_function_library_does_not_skip_python_attribute_effects` | JIT boundary evidence, not a complete paired JIT/AOT suite |
| Backend boundary | Same file: `test_jit_function_library_spirv_retains_one_typed_body`, `test_jit_function_library_spirv_keeps_array_abi_inline` | Vulkan module-local sharing and array fallback, not cross-module caching |
| Persistent units/errors | [test_aot_execution_context.py](../../tests/python/test_aot_execution_context.py): `test_bitcode_library_shares_real_function_and_validates_dependencies` | CUDA unit reuse, JIT numerical reference, instance isolation and missing/corrupt dependency rejection |
| AOT construction preserves JIT | [test_aot_package.py](../../tests/python/test_aot_package.py): `test_package_bitcode_library_preserves_jit_functions` | Success/failure retirement, continued JIT and producer-side inline fallback; select this node only |
| Consumption and instances | [test_aot_value_arguments.py](../../tests/python/test_aot_value_arguments.py): `test_prepared_bitcode_specialization_keeps_shapes_and_instances_independent`, `test_prepared_real_function_arguments_have_their_own_layout` | Shape/selected constants versus contents and independent internal ABI; the first test also has a Vulkan branch despite its name |

If shared contracts are subsequently changed, add one small paired JIT/AOT case in these
existing files, with explicit equal effective targets/configuration, one static change, one
ordinary-data change and one incompatible input. Reuse fixtures rather than building another
runner. Do not assert identical private key strings or binary bytes. Complete cross-format
version/capability rejection coverage was not established by this audit; add focused negative
cases when those contracts change.

## 9. Convergence decisions and stopping point

| Item | Decision/status | Future implementation boundary |
| --- | --- | --- |
| Request meaning/defaults | Comparison complete; correct the stale provenance comment that says purpose does not select policies | Do not change the resolver or request lifecycle |
| Function projection | Common meanings for ABI, static/resources/effects, specialization and fallback are defined; implementation remains distributed | Add a minimal read-only projection to existing objects only when needed for a concrete comparison; no new frontend or forced identical admission algorithm |
| Artifact projection | Separate semantic/content/final-specialization identities, version domains and formats | Add only consumer-required missing metadata when justified; no shared storage or global migration now |
| Consumption binding times | Production/prepare/execution responsibilities and rejection points are specified | Retain preparation reuse, binding and retirement; do not merge Program and AOT Runtime |
| Consistency evidence | Existing focused tests mapped; no complete common comparison suite claimed | Add minimal relevant counterexamples with actual contract changes; broaden regression at integration points |

Completion means a reviewable comparison/plan, clear semantics and preserved differences,
with no execution behavior change. It does not mean every interface field is now unified or
cross-path compilation has become faster. **Stop after this phase and wait for a decision
on next-release implementation unification.**

Unification is worthwhile only where measurements identify repeated shareable work. If typing
dominates, LLVM-body reuse alone is insufficient; if backend body generation dominates, existing
units may suffice; if final optimization/driver loading dominates, common frontend/storage does
not directly remove that cost. Final inlined code may legitimately remain the same size. Measure
producer AOT, user prepare/first call, JIT cold/warm compilation and execution separately.

Prerequisites are proven compatible identities and resource bindings, contracts covering the
real static domain, backend-specific composable representations, controlled immutable-unit
lifetimes and measured engineering benefit. Cross-kernel frontend reuse, a global Artifact
Store, complete cache/lifetime unification and whole-engine re-export are not started here.
