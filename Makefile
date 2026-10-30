# Root Makefile for NeuroCDCL Research Artifact
SHELL := /bin/bash

all: build-all

build-all: build-m35 build-kissat

build-m35:
	@echo "=== Building M3.5 Version 2 (SIMD MiniSat) ==="
	$(MAKE) -C solvers/m3_5_v2

build-kissat:
	@echo "=== Building Folder 52 (NeuroCDCL Kissat) ==="
	@if [ -f solvers/neuro_cdcl_f52/configure ]; then \
		cd solvers/neuro_cdcl_f52 && ./configure && $(MAKE); \
	fi

test:
	@echo "=== Running Automated Regression Test Suite ==="
	pytest tests/

reproduce:
	@echo "=== Reproducing Paper Tables 1, 2, and 3 ==="
	python3 scripts/reproduce_table1_rlaf.py
	python3 scripts/reproduce_table2_imitsat.py
	python3 scripts/reproduce_table3_neuroback.py

clean:
	$(MAKE) -C solvers/m3_5_v2 clean
	@if [ -f solvers/neuro_cdcl_f52/Makefile ]; then \
		$(MAKE) -C solvers/neuro_cdcl_f52 clean; \
	fi
