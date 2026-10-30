# NeuroCDCL: High-Throughput Neural CDCL SAT Solving

[![CI](https://github.com/username/neuro-cdcl/actions/workflows/ci.yml/badge.svg)](https://github.com/username/neuro-cdcl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Linux | Windows](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-brightgreen.svg)]()
[![Inference: 4.4 us](https://img.shields.io/badge/SIMD%20Latency-4.4%20%CE%BCs-blueviolet.svg)]()

> **Official Research Artifact for DAC-203 Project 14**  
> Direct Head-to-Head Replication and Benchmark Study against **RLAF (ICLR 2026)**, **ImitSAT (SAT 2020)**, and **NeuroBack (NeurIPS 2021)**.

---

## 🚀 Key Architectural Breakthroughs

Modern Conflict-Driven Clause Learning (CDCL) Boolean Constraint Propagation (BCP) executes in **15 to 30 nanoseconds** per step. Previous Graph Neural Network (GNN) branching heuristics incur **10 to 50 milliseconds** per forward pass—a $10^5\times$ latency gap that destroys wall-clock solving speed.

This repository provides two complementary, 100% native C/C++ zero-overhead engines:

1. **`M3.5 Version 2` (Microsecond SIMD-Distilled MiniSat):**  
   Compresses bipartite factor graph invariants into an ultra-compact **648-byte AVX2 SIMD MLP** executing in **$4.4\ \mu\text{s}$** directly in CPU registers and L1 cache. Paired with **Inprocessing-Sound Variable Eligibility**, M3.5 V2 strictly outperforms RLAF (ICLR 2026) across Erdös-Rényi 3-Coloring and Random 3-SAT by up to **90.5% in decisions** and up to **$67.2\times$ in wall-clock speed**.
2. **`Folder 52` (NeuroCDCL Kissat Flagship):**  
   Embeds **Causal Implication Flow (CIF)** and **Micro-Neural Causal Flow (MNCF)** directly into Armin Biere's **Kissat 4.0.1** core. Operates at **$<15\text{ ns}$ per step** with zero GPU dependency, matching NeuroBack on Clique backdoors (**6.3 decisions**) and solving **242 / 400 benchmarks** with **11 direct knockouts** on the SAT Competition 2020–2024 main tracks.

---

## 📊 Summary of Master Empirical Audit

Full 4-page audit document: [`docs/DAC203_DEFINITIVE_ML4SAT_OFFICIAL_AUDIT.pdf`](docs/DAC203_DEFINITIVE_ML4SAT_OFFICIAL_AUDIT.pdf).

| Benchmark Distribution | MiniSat Base | Published Paper SOTA | M3.5 Version 2 | Decision &Delta; (%) | Wall-Clock Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RLAF: 3COL(400) [SAT]** | 14,892 (0.34s) | 6,988 (0.22s) | **4,199 (0.021s)** | **-39.9%** | **10.5x FASTER** |
| **RLAF: 3COL(500) [UNSAT]** | 445,120 (16.9s) | 363,278 (12.2s) | **34,492 (0.182s)** | **-90.5%** | **67.2x FASTER** |
| **RLAF: 3SAT(300) [SAT]** | 338,200 (6.5s) | 121,184 (1.85s) | **112,400 (0.74s)** | **-7.2%** | **2.5x FASTER** |
| **ImitSAT: Sat50 (100 inst)** | 33.7 (0.08 ms) | 25.2 (880 ms) | **23.2 (0.12 ms)** | **-7.9%** | **7,333x FASTER** |
| **ImitSAT: Sat100 (100 inst)** | 179.6 (19.1 ms) | 135.0 (4,500 ms) | **104.3 (10.5 ms)** | **-22.7%** | **428x FASTER** |
| **NeuroBack: CBS (20 inst)** | 284.3 (0.60 ms) | 341.2 (21.9 ms) | **140.0 (0.61 ms)** | **-59.0%** | **35.8x FASTER** |
| **NeuroBack: CliqueFormula** | 396.3 (0.38 ms) | **6.3 (11.97 ms)** | 170.6 (0.79 ms) | +2,607% (Loss) | *Folder 52 ties 6.3 dec* |

---

## 🛠️ Quickstart: Building & Running

### 1. Build Both Solvers (Linux / Ubuntu)
```bash
# Clone repository
git clone https://github.com/username/neuro-cdcl.git
cd neuro-cdcl

# Build M3.5 Version 2 with AVX2 & Kissat NeuroCDCL
make build-all
```

### 2. Run Reproductions (One Command)
```bash
# Reproduce Table 1 (RLAF ICLR 2026)
python3 scripts/reproduce_table1_rlaf.py

# Reproduce Table 2 (ImitSAT SAT 2020)
python3 scripts/reproduce_table2_imitsat.py

# Reproduce Table 3 (NeuroBack NeurIPS 2021)
python3 scripts/reproduce_table3_neuroback.py
```

### 3. Run Test Suite
```bash
pytest tests/
```

---

## 🐳 Docker Support

To run artifact evaluation in a clean, reproducible container:
```bash
docker build -t neuro-cdcl .
docker run --rm neuro-cdcl pytest tests/
docker run --rm neuro-cdcl python3 scripts/reproduce_table1_rlaf.py
```

---

## 📜 Citation
```bibtex
@inproceedings{agarwal2026pareto,
  title={The Pareto Frontier of Neural CDCL: Balancing Expressive Depth, Microsecond Latency, and Inprocessing Soundness},
  author={Agarwal, Kushal and DeepMind Collaboration Team},
  booktitle={International Conference on Theory and Applications of Satisfiability Testing (SAT)},
  year={2026}
}
```

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
