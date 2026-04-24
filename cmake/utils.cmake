function(target_enable_function_level_linking TARGET)
    if(APPLE)
        target_link_options(${TARGET} PRIVATE -Wl,-dead_strip)
    elseif(MSVC) # WIN32
        target_link_options(${TARGET} PRIVATE /Gy)
    else() # Linux / *nix / gcc compatible
        target_link_options(${TARGET} PRIVATE -Wl,--gc-sections)
    endif()
endfunction()

# Silence LLVM 19/20 [[deprecated]] warnings on legacy PassManager /
# PassManagerBuilder APIs that Taichi still uses in the codegen-to-machine-
# code (`addPassesToEmitFile`) path. The migration away from
# `legacy::PassManager` is scheduled for Phase 6 (LLVM 21+). Applying the
# suppression here keeps `-Werror` builds green on LLVM 19 and 20.
function(target_silence_llvm_deprecated_warnings TARGET)
    if(MSVC)
        target_compile_options(${TARGET} PRIVATE /wd4996)
    else()
        target_compile_options(${TARGET} PRIVATE -Wno-deprecated-declarations)
    endif()
endfunction()
