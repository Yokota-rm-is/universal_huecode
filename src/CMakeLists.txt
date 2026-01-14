cmake_minimum_required(VERSION 3.16)
project(workspace_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# ---- ZBar ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZBAR REQUIRED zbar)
message(STATUS "ZBar version (pkg-config): ${ZBAR_VERSION}")
message(STATUS "ZBar include dirs: ${ZBAR_INCLUDE_DIRS}")
message(STATUS "ZBar libraries: ${ZBAR_LIBRARIES}")

# ---- Boost ----
find_package(Boost REQUIRED)

# ---- yaml-cpp ----
find_package(yaml-cpp REQUIRED)

# ---- common include root ----
add_library(workspace_tools_common INTERFACE)
target_include_directories(workspace_tools_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ZBAR_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# optional: compile options
target_compile_options(workspace_tools_common INTERFACE
  -O3
  -Wall
  -Wextra
)

# ---- executables ----
add_executable(detect
  main/detect.cpp
)
target_link_libraries(detect PRIVATE 
  workspace_tools_common 
  ${OpenCV_LIBS} 
  ${ZBAR_LIBRARIES} 
  yaml-cpp
)