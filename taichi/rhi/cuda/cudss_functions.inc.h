// Minimal public cuDSS C ABI used by Forge's optional staged CSR solver.
// Keep this table independent of CUDA Toolkit headers: cuDSS is discovered
// and loaded from a user-managed shared library at runtime.
PER_CUDSS_FUNCTION(create, cudssCreate, void **)
PER_CUDSS_FUNCTION(destroy, cudssDestroy, void *)
PER_CUDSS_FUNCTION(set_stream, cudssSetStream, void *, void *)
PER_CUDSS_FUNCTION(config_create, cudssConfigCreate, void **)
PER_CUDSS_FUNCTION(config_destroy, cudssConfigDestroy, void *)
PER_CUDSS_FUNCTION(data_create, cudssDataCreate, const void *, void **)
PER_CUDSS_FUNCTION(data_destroy, cudssDataDestroy, void *, void *)
PER_CUDSS_FUNCTION(matrix_create_csr,
                   cudssMatrixCreateCsr,
                   void **,
                   int64_t,
                   int64_t,
                   int64_t,
                   const void *,
                   const void *,
                   const void *,
                   const void *,
                   int,
                   int,
                   int,
                   int,
                   int,
                   int)
PER_CUDSS_FUNCTION(matrix_create_dn,
                   cudssMatrixCreateDn,
                   void **,
                   int64_t,
                   int64_t,
                   int64_t,
                   const void *,
                   int,
                   int)
PER_CUDSS_FUNCTION(matrix_destroy, cudssMatrixDestroy, void *)
PER_CUDSS_FUNCTION(matrix_set_values,
                   cudssMatrixSetValues,
                   void *,
                   const void *)
PER_CUDSS_FUNCTION(execute,
                   cudssExecute,
                   void *,
                   int,
                   const void *,
                   void *,
                   const void *,
                   void *,
                   const void *)
