cmake_minimum_required(VERSION 3.16)
project(cpp_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Ofast -march=native -fopenmp)

# 출력 디렉토리(원하면 바꿔도 됨)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- Flightmare root 잡기 (핵심) ---
# cpp_control -> tonedioapp -> flightmare (2단계 상위)
get_filename_component(FLIGHTMARE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

set(FLIGHTLIB_INCLUDE_DIR "${FLIGHTMARE_ROOT}/flightlib/include")
set(FLIGHTLIB_LIB "${FLIGHTMARE_ROOT}/flightlib/build_cpp/libflightlib.a")

message(STATUS "FLIGHTMARE_ROOT=${FLIGHTMARE_ROOT}")
message(STATUS "FLIGHTLIB_LIB=${FLIGHTLIB_LIB}")

if (NOT EXISTS "${FLIGHTLIB_LIB}")
  message(FATAL_ERROR "Not found: ${FLIGHTLIB_LIB} (build flightlib first)")
endif()

add_executable(fmp src/fmp_connection.cpp)


target_include_directories(fmp PUBLIC
  "${FLIGHTLIB_INCLUDE_DIR}"
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fmp PUBLIC
  ${FLIGHTLIB_LIB}
  yaml-cpp
  stdc++fs
  ${OpenCV_LIBS}
  zmq
  zmqpp
  pthread
)

# gcc 버전에 따라 -fopenmp 링크가 필요할 수도 있음
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(fmp PUBLIC OpenMP::OpenMP_CXX)
endif()