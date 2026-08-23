PER_CUFFT_FUNCTION(plan_1d, cufftPlan1d, int *, int, int, int);
PER_CUFFT_FUNCTION(plan_many, cufftPlanMany, int *, int, int *, int *, int,
                   int, int *, int, int, int, int);
PER_CUFFT_FUNCTION(destroy, cufftDestroy, int);
PER_CUFFT_FUNCTION(set_stream, cufftSetStream, int, CUstream);
PER_CUFFT_FUNCTION(exec_c2c, cufftExecC2C, int, void *, void *, int);
PER_CUFFT_FUNCTION(exec_r2c, cufftExecR2C, int, void *, void *);
PER_CUFFT_FUNCTION(exec_c2r, cufftExecC2R, int, void *, void *);
PER_CUFFT_FUNCTION(get_size, cufftGetSize, int, std::size_t *);
PER_CUFFT_FUNCTION(get_version, cufftGetVersion, int *);
