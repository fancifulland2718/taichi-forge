// CUDA APIs newer than Taichi's minimum supported driver must stay optional.
// Capability checks gate every call site before invoking these wrappers.

PER_CUDA_OPTIONAL_FUNCTION(stream_begin_capture_to_graph,
                           cuStreamBeginCaptureToGraph,
                           void *,
                           CUgraph,
                           const void *,
                           const void *,
                           std::size_t,
                           CUstreamCaptureMode);
PER_CUDA_OPTIONAL_FUNCTION(graph_conditional_handle_create,
                           cuGraphConditionalHandleCreate,
                           std::uint64_t *,
                           CUgraph,
                           void *,
                           unsigned int,
                           unsigned int);
PER_CUDA_OPTIONAL_FUNCTION(graph_add_node,
                           cuGraphAddNode,
                           void **,
                           CUgraph,
                           const void *,
                           std::size_t,
                           const TaichiCudaGraphNodeParams *);
PER_CUDA_OPTIONAL_FUNCTION(launch_kernel_ex,
                           cuLaunchKernelEx,
                           const TaichiCudaLaunchConfig *,
                           void *,
                           void **,
                           void **);
PER_CUDA_OPTIONAL_FUNCTION(graph_upload, cuGraphUpload, CUgraphExec, void *);
