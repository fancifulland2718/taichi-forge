# ./taichi/rhi/cuda/CMakeLists.txt

set(CUDA_RHI cuda_rhi)
ti_add_native_library(${CUDA_RHI})
target_sources(${CUDA_RHI}
  PRIVATE
    cuda_device.cpp
    cuda_context.cpp
    cuda_driver.cpp
    cuda_profiler.cpp
    cuda_primitives.cpp
    primitives/hierarchical_ptx.cpp
    primitives/linear_ptx.cpp
    primitives/solver_ptx.cpp
    cupti_toolkit.cpp
  )

target_include_directories(${CUDA_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
  )

target_link_libraries(${CUDA_RHI} PRIVATE interop_rhi)

if (TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if (TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE)
  target_sources(${CUDA_RHI} PRIVATE cuda_primitives.cu)
  target_include_directories(${CUDA_RHI} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_source_files_properties(cuda_primitives.cpp
    PROPERTIES COMPILE_DEFINITIONS
      "TI_WITH_CUDA_TOOLKIT;TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE")
  get_filename_component(TI_CUDA_TOOLKIT_ROOT
    "${CUDAToolkit_INCLUDE_DIRS}/.." ABSOLUTE)
  target_compile_definitions(${CUDA_RHI}
    PRIVATE
      TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE
      TI_CUDA_TOOLKIT_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
      TI_CUDA_TOOLKIT_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR})

  if (TI_CUDA_CUB_SORT_DYNAMIC_CUDART)
    target_link_libraries(${CUDA_RHI} PUBLIC CUDA::cudart)
    set_property(TARGET ${CUDA_RHI} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    target_compile_definitions(${CUDA_RHI}
      PRIVATE
        TI_CUDA_CUB_SORT_DYNAMIC_CUDART
        TI_CUDA_CUB_SORT_CUDART_DLL="cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
    if (MSVC)
      target_link_options(${CUDA_RHI}
        INTERFACE
          "/DELAYLOAD:cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
      target_link_libraries(${CUDA_RHI} INTERFACE delayimp)
    endif()
  else()
    find_library(TI_CUDA_CUDART_STATIC_LIBRARY
      NAMES cudart_static
      PATHS "${TI_CUDA_TOOLKIT_ROOT}/lib/x64"
      NO_DEFAULT_PATH)
    if (TI_CUDA_CUDART_STATIC_LIBRARY)
      target_link_libraries(${CUDA_RHI} PRIVATE ${TI_CUDA_CUDART_STATIC_LIBRARY})
    else()
      target_link_libraries(${CUDA_RHI} PRIVATE CUDA::cudart_static)
    endif()
    set_property(TARGET ${CUDA_RHI} PROPERTY CUDA_RUNTIME_LIBRARY Static)
  endif()

  set_property(TARGET ${CUDA_RHI} PROPERTY CUDA_STANDARD 17)
  set(TI_CUDA_RHI_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  set(TI_CUDA_RHI_BASELINE_ARCH "" CACHE STRING
      "Baseline CUDA architecture used by cuda_rhi when CMAKE_CUDA_ARCHITECTURES is unset")
  if (NOT TI_CUDA_RHI_ARCHITECTURES)
    if (TI_CUDA_RHI_BASELINE_ARCH)
      set(TI_CUDA_RHI_ARCHITECTURES
          ${TI_CUDA_RHI_BASELINE_ARCH}-real
          ${TI_CUDA_RHI_BASELINE_ARCH}-virtual)
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.2)
      # CUDA 13 no longer supports generating code below compute_75.  Keep the
      # default wheel broad and compact for that toolkit by shipping sm_75 SASS
      # plus compute_75 PTX; newer devices can JIT the PTX.
      set(TI_CUDA_RHI_ARCHITECTURES 75-real 75-virtual)
    else()
      # Lowering the Toolkit/runtime driver requirement must not silently
      # expand Forge's supported GPU range to legacy architectures.  Preserve
      # the current sm_75 baseline unless a release build supplies an explicit
      # architecture matrix.
      set(TI_CUDA_RHI_ARCHITECTURES 75-real 75-virtual)
    endif()
  endif()
  message(STATUS "CUDA RHI architectures: ${TI_CUDA_RHI_ARCHITECTURES}")
  set_property(TARGET ${CUDA_RHI}
    PROPERTY CUDA_ARCHITECTURES ${TI_CUDA_RHI_ARCHITECTURES})
  if (MSVC)
    target_compile_options(${CUDA_RHI}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
  endif()
endif()

if (TI_WITH_CUPTI)
  target_include_directories(${CUDA_RHI} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(${CUDA_RHI}
    PRIVATE
      "${CUDAToolkit_CUPTI_INCLUDE_DIR}"
  )
  target_compile_definitions(${CUDA_RHI}
    PRIVATE
      TI_WITH_CUDA_TOOLKIT_HEADERS
      TI_WITH_CUPTI)
endif()
