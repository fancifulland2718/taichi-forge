# Source archives carry the original commit without requiring a .git directory.
# Do not accidentally read a parent checkout when building an extracted archive.
if(EXISTS "${CMAKE_SOURCE_DIR}/SOURCE_COMMIT")
  file(READ "${CMAKE_SOURCE_DIR}/SOURCE_COMMIT" TI_COMMIT_HASH)
  string(STRIP "${TI_COMMIT_HASH}" TI_COMMIT_HASH)
  string(LENGTH "${TI_COMMIT_HASH}" _ti_commit_length)
  if(NOT _ti_commit_length EQUAL 40 OR NOT TI_COMMIT_HASH MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Invalid source archive SOURCE_COMMIT")
  endif()
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _ti_git_result OUTPUT_VARIABLE TI_COMMIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _ti_git_result EQUAL 0)
    message(FATAL_ERROR "Cannot read the source commit")
  endif()
else()
  set(TI_COMMIT_HASH "unknown")
endif()
string(SUBSTRING "${TI_COMMIT_HASH}" 0 8 TI_COMMIT_SHORT_HASH)
