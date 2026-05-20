cmake_minimum_required(VERSION 3.20)

# Add custom modules to the search path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(ktir-mlir-frontend LANGUAGES C CXX)

# Set defaults for compiler features and options.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

# Declare new CMake build options and their defaults.
option(KTIR_ENABLE_PYTHON_BINDINGS "Build Python bindings" NO)
option(KTIR_BUILD_TOOLS "Build tool executables" YES)

# Detect if this is (supposed to be) a stand-alone build.
if(NOT KTDP_STAND_ALONE)
  string(COMPARE EQUAL 
    "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" 
    KTDP_STAND_ALONE
  )
endif()
if(KTDP_STAND_ALONE)
  # Find MLIR.
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  # Load LLVM and MLIR CMake modules.
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  # Apply target properties for LLVM/MLIR globally.
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})

  # Set shared output directories.
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

  # Set project-specific output directories.
  set(KTIR_BINARY_DIR ${CMAKE_BINARY_DIR})
endif()

# Apply target properties for this project globally.
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Include KTIR cmake module.
include(AddKTIR)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

if(KTIR_ENABLE_PYTHON_BINDINGS)
  set(MLIR_ENABLE_BINDINGS_PYTHON ON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()

  if(NOT MLIR_PYTHON_PACKAGE_PREFIX)
    set(MLIR_PYTHON_PACKAGE_PREFIX "mlir_ktdp" CACHE STRING "" FORCE)
  endif()
  if(NOT MLIR_BINDINGS_PYTHON_NB_DOMAIN)
    set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir_ktdp" CACHE STRING "" FORCE)
  endif()
  if(NOT MLIR_BINDINGS_PYTHON_INSTALL_PREFIX)
    set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX 
      "python_packages/ktdp/${MLIR_PYTHON_PACKAGE_PREFIX}" 
      CACHE STRING "" FORCE
    )
  endif()

  add_subdirectory(python)
endif()
