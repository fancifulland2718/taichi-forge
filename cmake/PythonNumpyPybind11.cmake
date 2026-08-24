# Python, NumPy, and pybind11. scikit-build-core provides both the modern
# Python_EXECUTABLE and the legacy PYTHON_EXECUTABLE cache variables. Prefer
# CMake's FindPython mode so pybind11 does not depend on the removed
# FindPythonInterp/FindPythonLibs modules on newer CMake/Python versions.
if(NOT Python_EXECUTABLE AND PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Keep the few existing build diagnostics and helper commands on the legacy
# variable names until the rest of the CMake tree is migrated.
set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
set(PYTHON_VERSION_STRING "${Python_VERSION}")
set(PYTHON_INCLUDE_DIR "${Python_INCLUDE_DIRS}")
set(PYTHON_INCLUDE_DIRS "${Python_INCLUDE_DIRS}")
set(PYTHON_LIBRARIES "${Python_LIBRARIES}")
if(Python_LIBRARIES)
    list(GET Python_LIBRARIES 0 PYTHON_LIBRARY)
endif()

execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- Python: Using ${Python_EXECUTABLE} as the interpreter")
message("    version: ${PYTHON_VERSION_STRING}")
message("    include: ${PYTHON_INCLUDE_DIR}")
message("    library: ${PYTHON_LIBRARY}")
message("    numpy include: ${NUMPY_INCLUDE_DIR}")

include_directories(${NUMPY_INCLUDE_DIR})

find_package(pybind11 CONFIG REQUIRED)
