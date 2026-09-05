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

// Keep the legacy entry-point ABI explicit: cuda.h maps current source names
// to newer layouts, but these unversioned driver exports remain available on
// older CUDA Graph drivers. A materializer must qualify availability once;
// loading these symbols does not enable a new replay policy by itself.
PER_CUDA_OPTIONAL_FUNCTION(graph_node_get_type,
                           cuGraphNodeGetType, void *, std::uint32_t *);
PER_CUDA_OPTIONAL_FUNCTION(graph_kernel_node_get_params_v1,
                           cuGraphKernelNodeGetParams, void *,
                           TaichiCudaKernelNodeParamsV1 *);
PER_CUDA_OPTIONAL_FUNCTION(graph_exec_kernel_node_set_params_v1,
                           cuGraphExecKernelNodeSetParams, CUgraphExec, void *,
                           const TaichiCudaKernelNodeParamsV1 *);
PER_CUDA_OPTIONAL_FUNCTION(graph_exec_update_v1,
                           cuGraphExecUpdate, CUgraphExec, CUgraph, void **,
                           std::uint32_t *);
