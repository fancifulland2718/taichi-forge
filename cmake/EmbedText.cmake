if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR NOT DEFINED SYMBOL_NAME)
    message(FATAL_ERROR "EmbedText.cmake requires INPUT_FILE, OUTPUT_FILE, and SYMBOL_NAME")
endif()

file(READ "${INPUT_FILE}" _ti_embedded_text)
if(DEFINED EXPECTED_PTX_VERSION AND NOT EXPECTED_PTX_VERSION STREQUAL "")
    string(REGEX MATCH "\\.version[ \\t]+([0-9]+\\.[0-9]+)"
           _ti_ptx_version_line "${_ti_embedded_text}")
    if(NOT CMAKE_MATCH_1 STREQUAL EXPECTED_PTX_VERSION)
        message(FATAL_ERROR
            "Expected PTX ${EXPECTED_PTX_VERSION}, found ${CMAKE_MATCH_1} in ${INPUT_FILE}")
    endif()
endif()
if(DEFINED EXPECTED_PTX_TARGET AND NOT EXPECTED_PTX_TARGET STREQUAL "")
    string(REGEX MATCH "\\.target[ \\t]+([A-Za-z0-9_]+)"
           _ti_ptx_target_line "${_ti_embedded_text}")
    if(NOT CMAKE_MATCH_1 STREQUAL EXPECTED_PTX_TARGET)
        message(FATAL_ERROR
            "Expected PTX target ${EXPECTED_PTX_TARGET}, found ${CMAKE_MATCH_1} in ${INPUT_FILE}")
    endif()
endif()
string(FIND "${_ti_embedded_text}" ")TFOPTIXPTX\"" _ti_embedded_delimiter)
if(NOT _ti_embedded_delimiter EQUAL -1)
    message(FATAL_ERROR "Embedded input contains the reserved raw-string delimiter")
endif()
file(WRITE "${OUTPUT_FILE}"
    "#pragma once\n"
    "static const char ${SYMBOL_NAME}[] = R\"TFOPTIXPTX(${_ti_embedded_text})TFOPTIXPTX\";\n")
