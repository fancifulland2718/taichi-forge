# Device bitcode is a compiler input, not the installed HIP runtime. In
# particular LLVM 22 bitcode from newer SDKs is not valid input to LLVM 20.
set(TI_AMDGPU_DEVICE_LIBS_DIR "" CACHE PATH
    "LLVM-compatible ROCm device bitcode directory (empty: pinned LLVM 20 package)")
set(TI_AMDGPU_DEVICE_LIBS_LICENSE "" CACHE FILEPATH
    "License accompanying a custom ROCm device bitcode directory")
if(NOT TI_AMDGPU_DEVICE_LIBS_DIR)
    if(NOT LLVM_VERSION_MAJOR EQUAL 20)
        message(FATAL_ERROR
            "AMDGPU: set TI_AMDGPU_DEVICE_LIBS_DIR to device libraries compatible with LLVM ${LLVM_VERSION_MAJOR}")
    endif()
    set(_ti_amd_root "${CMAKE_BINARY_DIR}/amdgpu-device-libs-7.0.2")
    set(_ti_amd_archive "${_ti_amd_root}/device-libs.deb")
    set(_ti_amd_bits "${_ti_amd_root}/opt/rocm-7.0.2/lib/llvm/lib/clang/20/lib/amdgcn/bitcode")
    if(NOT EXISTS "${_ti_amd_bits}/ocml.bc" OR
       NOT EXISTS "${_ti_amd_root}/opt/rocm-7.0.2/share/doc/ROCm-Device-Libs/LICENSE.TXT")
        file(MAKE_DIRECTORY "${_ti_amd_root}")
        file(DOWNLOAD
            "https://repo.radeon.com/rocm/apt/7.0.2/pool/main/r/rocm-device-libs/rocm-device-libs_1.0.0.70002-56~22.04_amd64.deb"
            "${_ti_amd_archive}"
            EXPECTED_HASH SHA256=aaa03e502841270da8bba08441360b9c1beef5a26d5fa06f0910f3817db68f82
            TLS_VERIFY ON)
        file(ARCHIVE_EXTRACT INPUT "${_ti_amd_archive}"
            DESTINATION "${_ti_amd_root}" PATTERNS "data.tar.gz")
        # The LLVM bitcode is host-OS independent. Do not extract Linux
        # symlinks or install any system libraries on the build machine.
        file(ARCHIVE_EXTRACT INPUT "${_ti_amd_root}/data.tar.gz"
            DESTINATION "${_ti_amd_root}"
            PATTERNS "*/bitcode/*.bc" "*/ROCm-Device-Libs/LICENSE.TXT")
    endif()
    set(TI_AMDGPU_DEVICE_LIBS_DIR "${_ti_amd_bits}")
    set(TI_AMDGPU_DEVICE_LIBS_LICENSE
        "${_ti_amd_root}/opt/rocm-7.0.2/share/doc/ROCm-Device-Libs/LICENSE.TXT")
endif()
foreach(_ti_amd_file ocml.bc ockl.bc opencl.bc oclc_abi_version_500.bc
        oclc_wavefrontsize64_off.bc oclc_wavefrontsize64_on.bc)
    if(NOT EXISTS "${TI_AMDGPU_DEVICE_LIBS_DIR}/${_ti_amd_file}")
        message(FATAL_ERROR "Missing AMDGPU device library: ${TI_AMDGPU_DEVICE_LIBS_DIR}/${_ti_amd_file}")
    endif()
endforeach()
if(NOT EXISTS "${TI_AMDGPU_DEVICE_LIBS_LICENSE}")
    message(FATAL_ERROR "AMDGPU device bitcode requires its redistributable license")
endif()
message(STATUS "AMDGPU device bitcode: ${TI_AMDGPU_DEVICE_LIBS_DIR}")
