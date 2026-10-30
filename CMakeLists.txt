cmake_minimum_required(VERSION 3.16)
project(NeuroCDCL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

# Compiler optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -mavx2 -mfma -flto -Wall -Wextra)
endif()

# Include directories
include_directories(
    solvers/m3_5_v2
    solvers/m3_5_v2/src
    solvers/m3_5_v2/upstream_minisat
    solvers/m3_5_v2/upstream_minisat/minisat
)

# M3.5 Version 2 Executable
add_executable(m3_5_v2
    solvers/m3_5_v2/src/main.cc
    solvers/m3_5_v2/src/minisat_adapter.cc
    solvers/m3_5_v2/src/structural_mlp_engine.cc
    solvers/m3_5_v2/src/gnn_engine.cc
    solvers/m3_5_v2/upstream_minisat/minisat/core/Solver.cc
    solvers/m3_5_v2/upstream_minisat/minisat/simp/SimpSolver.cc
    solvers/m3_5_v2/upstream_minisat/minisat/utils/System.cc
    solvers/m3_5_v2/upstream_minisat/minisat/utils/Options.cc
)

find_package(ZLIB REQUIRED)
target_link_libraries(m3_5_v2 PRIVATE ZLIB::ZLIB)
target_compile_definitions(m3_5_v2 PRIVATE __STDC_LIMIT_MACROS __STDC_FORMAT_MACROS NDEBUG)
