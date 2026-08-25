if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR NOT DEFINED SYMBOL_NAME)
    message(FATAL_ERROR "EmbedText.cmake requires INPUT_FILE, OUTPUT_FILE, and SYMBOL_NAME")
endif()

file(READ "${INPUT_FILE}" _ti_embedded_text)
string(FIND "${_ti_embedded_text}" ")TFOPTIXPTX\"" _ti_embedded_delimiter)
if(NOT _ti_embedded_delimiter EQUAL -1)
    message(FATAL_ERROR "Embedded input contains the reserved raw-string delimiter")
endif()
file(WRITE "${OUTPUT_FILE}"
    "#pragma once\n"
    "static const char ${SYMBOL_NAME}[] = R\"TFOPTIXPTX(${_ti_embedded_text})TFOPTIXPTX\";\n")
