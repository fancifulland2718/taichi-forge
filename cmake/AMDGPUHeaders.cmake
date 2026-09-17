# HIP is dynamically loaded. Only the host ABI header closure is needed here,
# not a HIP compiler, runtime library, or the full ROCm source distribution.
# Source: TheRock therock-7.14.1's pinned rocm-systems revision.
set(TI_HIP_INCLUDE_DIR "" CACHE PATH "HIP 7+ host headers; empty uses pinned 7.14.1 headers")
if(NOT TI_HIP_INCLUDE_DIR)
    set(_ti_hip_revision ca887ee80abfb82671fe1d6d8da708a713438e05)
    set(_ti_hip_root "${CMAKE_BINARY_DIR}/hip-host-7.14.1")
    set(TI_HIP_INCLUDE_DIR "${_ti_hip_root}/include")
    set(_ti_hip_files
        "hip_runtime_api.h|a075feec4c32d01b35dfe97e01ef7b55aa62bac7b7590ca7cefc2564668aa6a2"
        "hip_common.h|86fb0f9fe17aaa58bf6bf91ea8aaf3ea20721def159ec5af415eb03010dd9cbe"
        "linker_types.h|2b8236794a1c152ac07db0455c93b77557724f277d88fbd5fe0ace3b82cf54b7"
        "driver_types.h|c4d19db7d6e9f668c61df074368507582b037e3e6d597a16f528a75cbd996a58"
        "texture_types.h|7542d6d2ea648210c77753bfe67624084eaed935c7398bae932b9dddb0dd7d21"
        "surface_types.h|23c95bde443ddc4b6f9b463b2816018d102257e37125ea88a2481b100e731c33"
        "channel_descriptor.h|5b4afd231c3126b203b75e7b16179c107de53ccc342ad6a4039cc8907a9724a8"
        "amd_detail/host_defines.h|5b4592fba67e65d2fe10c41c0ec5fad73d28b5c7c9d2acd780a112c083ca3e93"
        "amd_detail/amd_hip_runtime_pt_api.h|ee8d8978886dbe8e27ed6c94a3f3e213ab0906242e509c5b91b4cfc1e745a5b3"
        "amd_detail/amd_channel_descriptor.h|9ac6ef411d5d32fdbce980e8a3157d6ca66fb98f3fd2b635c76955c52eac6813"
        "amd_detail/amd_hip_vector_types.h|efa048d0802df7b432db5d9426324916855b6f77dd770e5f76c94b2608752a33")
    function(_ti_fetch_hip_host_file source destination sha256)
        if(EXISTS "${destination}")
            file(SHA256 "${destination}" _actual)
            if(_actual STREQUAL sha256)
                return()
            endif()
        endif()
        get_filename_component(_directory "${destination}" DIRECTORY)
        file(MAKE_DIRECTORY "${_directory}")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/ROCm/rocm-systems/${_ti_hip_revision}/${source}"
            "${destination}" EXPECTED_HASH "SHA256=${sha256}" TLS_VERIFY ON)
    endfunction()
    foreach(_entry IN LISTS _ti_hip_files)
        string(REPLACE "|" ";" _parts "${_entry}")
        list(GET _parts 0 _name)
        list(GET _parts 1 _sha256)
        if(_name MATCHES "^amd_detail/")
            set(_source "projects/clr/hipamd/include/hip/${_name}")
        else()
            set(_source "projects/hip/include/hip/${_name}")
        endif()
        _ti_fetch_hip_host_file("${_source}" "${TI_HIP_INCLUDE_DIR}/hip/${_name}" "${_sha256}")
    endforeach()
    # These values come from projects/hip/VERSION at the revision above.
    configure_file("${CMAKE_CURRENT_LIST_DIR}/hip_version.h.in"
                   "${TI_HIP_INCLUDE_DIR}/hip/hip_version.h" @ONLY)
    _ti_fetch_hip_host_file("projects/hip/LICENSE.md" "${_ti_hip_root}/HIP-LICENSE.txt"
        b185aaa652b0bf066c37a0d6314ce4bf4521e4a3c9bf46edd2f6a777ac522223)
    install(FILES "${_ti_hip_root}/HIP-LICENSE.txt"
        DESTINATION "${INSTALL_LIB_DIR}/licenses/amdgpu" COMPONENT runtime)
endif()
if(NOT EXISTS "${TI_HIP_INCLUDE_DIR}/hip/hip_runtime_api.h")
    message(FATAL_ERROR "TI_HIP_INCLUDE_DIR must contain HIP 7+ host API headers")
endif()
message(STATUS "AMDGPU host headers: ${TI_HIP_INCLUDE_DIR}")
