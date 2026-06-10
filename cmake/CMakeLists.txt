include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if (KTIR_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH YES PARENT_SCOPE)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" PARENT_SCOPE)
endif()

install(DIRECTORY 
    ${KTIR_SOURCE_DIR}/include/
    ${KTIR_BINARY_DIR}/include/
  TYPE INCLUDE
  COMPONENT ktir-headers
  FILES_MATCHING
    PATTERN "*.td"
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.inc"
    PATTERN "*.hpp.inc"
    REGEX "CMakeFiles" EXCLUDE 
)

install(DIRECTORY
    ${KTIR_BINARY_DIR}/doc/
  DESTINATION doc
  COMPONENT ktir-doc
  FILES_MATCHING
    PATTERN "*.md"
    REGEX "CMakeFiles" EXCLUDE
)

# NOTE: CMake adds LLVM/MLIR to the RPATH of the targets, so this is not needed.
#       If a bundled installation is desired, the user should do it manually.
# if(LLVM_LINK_LLVM_DYLIB)
#   install(IMPORTED_RUNTIME_ARTIFACTS LLVM)
# endif()
# if(MLIR_LINK_MLIR_DYLIB)
#   install(IMPORTED_RUNTIME_ARTIFACTS MLIR)
# endif()

install(EXPORT ${PROJECT_NAME}
  FILE "${PROJECT_NAME}Targets.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT ktir-cmake
)

export(EXPORT ${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  FILE "${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}Targets.cmake"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)
