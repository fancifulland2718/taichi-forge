PER_CUFFT_FUNCTION(plan_1d, cufftPlan1d, int *, int, int, int);
PER_CUFFT_FUNCTION(destroy, cufftDestroy, int);
PER_CUFFT_FUNCTION(set_stream, cufftSetStream, int, CUstream);
PER_CUFFT_FUNCTION(exec_c2c, cufftExecC2C, int, void *, void *, int);
PER_CUFFT_FUNCTION(get_version, cufftGetVersion, int *);
