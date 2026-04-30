# ./taichi/runtime/gfx/CMakeLists.txt

# Phase 2a — pointer-on-Vulkan device-side allocator (default OFF, see
# compile_doc/SNode_Vulkan_规划.md §2.3). When OFF, snode_allocator.cpp does
# not enter the build at all, so vanilla behavior is bit-identical.
option(TI_WITH_VULKAN_POINTER "Enable WIP pointer-SNode-on-Vulkan allocator (Phase 2a)" OFF)

add_library(gfx_runtime)
target_sources(gfx_runtime
  PRIVATE
    runtime.cpp
    snode_tree_manager.cpp
    aot_module_builder_impl.cpp
    aot_module_loader_impl.cpp
    kernel_launcher.cpp
  )
if(TI_WITH_VULKAN_POINTER)
  target_sources(gfx_runtime PRIVATE snode_allocator.cpp)
  target_compile_definitions(gfx_runtime PRIVATE TI_WITH_VULKAN_POINTER=1)
endif()
#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(gfx_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
        ${LLVM_INCLUDE_DIRS}
  )
target_link_libraries(gfx_runtime PRIVATE taichi_common)
