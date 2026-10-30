# M3.5 Version 2 Portable Makefile (Linux gcc/clang & Windows MinGW)
CXX ?= g++
CXXFLAGS ?= -O3 -mavx2 -mfma -flto -std=c++17 -Wall -Wextra -DNDEBUG -fpermissive -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS
INCLUDES = -I. -I./src -I./upstream_minisat -I./upstream_minisat/minisat
LIBS = -lz

TARGET = m3_5_v2
ifeq ($(OS),Windows_NT)
    TARGET := m3_5_v2.exe
    LIBS += -static-libgcc -static-libstdc++
endif

SOURCES = src/main.cc src/minisat_adapter.cc src/structural_mlp_engine.cc src/gnn_engine.cc \
          upstream_minisat/minisat/core/Solver.cc \
          upstream_minisat/minisat/simp/SimpSolver.cc \
          upstream_minisat/minisat/utils/System.cc \
          upstream_minisat/minisat/utils/Options.cc

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o src/*.o
