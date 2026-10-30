# Master Literature Synthesis & Inspiration Matrix

This authoritative dossier documents the lineage of theoretical and empirical research underlying the **NeuroCDCL** platform. It details how the foundational paradigms in Machine Learning for SAT (ML4SAT) were audited, where they succeeded, where their latency or structural bottlenecks failed, and what exact concepts were synthesized into **M3.5 Version 2** and **Folder 52 (NeuroCDCL Kissat)**.

---

## 1. Master Comparative Matrix: The ML4SAT Landscape

| Research Paper | Core Neural Architecture | Training / Optimization Paradigm | Target Solver | Query Frequency & Hardware Profile | Decision Pruning Claim | Real Wall-Clock Speedup | Fatal Bottleneck / Limitation | Concept Synthesized into NeuroCDCL |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **NeuroSAT** *(Selsam et al., ICLR 2019)* | 32-layer LSTM Bipartite GNN ($d=128$) | Single-bit SAT/UNSAT End-to-End Supervised | None (Standalone) | End-to-End Neural (GPU CUDA) | N/A | None ($1,000\times$ slower) | Cannot scale beyond 40 variables; zero mathematical refutation soundness. | Bipartite literal-clause graph formulation and literal-negation involution symmetry. |
| **NeuroCore** *(Selsam & Bjørner, SAT 2019)* | Bipartite MPNN ($d=64$) | Supervised on Minimal Unsat Cores (MUS) | MiniSat / Glucose / Z3 | Periodic at Restarts (Python PyTorch / GPU) | Up to -40% on small cores | Negligible / Slower on industrial | Fixed-counter restart interrupts cause heavy GPU stall and cache evictions. | Refutation-centric guidance targeting unsatisfiable core variables over satisfying models. |
| **Graph-Q-SAT** *(Kurin et al., NeurIPS 2020)* | Recurrent MPNN + Linear Q-Head | Deep Q-Networks (DQN) with Step Penalty | MiniSat 2.2 | **Synchronous Per-Step** (PyTorch GPU/CPU) | 2x to 3x decision reduction | **80x to 150x SLOWER** | Per-decision GPU latency ($30\text{--}80\text{ ms}$) creates a $10^5\times$ slowdown per decision step. | Strict $< 5\ \mu\text{s}$ latency ceiling contract; motivated in-register causal tracking. |
| **ImitSAT** *(Yolcu & Póczos, SAT 2020 / ICLR)* | GNN + Perceiver AR Transformer | DAgger Imitation Learning of VSIDS | MiniSat / Glucose / CaDiCaL | Per-Step via Socket IPC (PyTorch Python Server) | -25% on Sat50/Sat100 | **7,333x SLOWER** on Sat50; Dubois Timeout | Python-C++ socket IPC overhead ($880\text{ ms}$) destroys millisecond solving; Dubois failure. | Distilling VSIDS structural imitation into microsecond native AVX2 SIMD registers. |
| **NeuroComb** *(Wang et al., ICLR 2024)* | Alternating GNN + Dual Variable/Clause Heads | Supervised Regression on Glucose Traces | Glucose 4.0 | One-Shot + Restart MAB (PyTorch) | -15% to -30% on select benchmarks | Marginal (1.1x to 1.3x) | Supervised imitation of suboptimal traces; MAB has zero out-of-distribution awareness. | Multi-Armed Bandit restart scheduling embedded directly in C99 inside Kissat. |
| **NeuroBack** *(Zhang et al., NeurIPS 2021)* | Graph Transformer Network | Supervised Backdoor Classification (ILP Labels) | Kissat / Glucose | One-Shot at $t=0$ (PyTorch GPU) | **98.4% on Clique** (396 -> 6.3 dec) | 0.9x to 1.1x (12-30 ms latency) | GNN backdoor predictions fail on unstructured random 3-SAT (+57% decision regression). | Backdoor priority concept; transparently maps the Pareto boundary on CliqueFormula. |
| **RLAF** *(Tönshoff et al., ICLR 2026)* | 5-layer Relational GCN + Dual Heads | Reinforcement Learning from Algorithm Feedback (GRPO) | Glucose 4.2.1 / MiniSat | One-Shot at $t=0$ (PyTorch) | -30% to -60% on 3-Col / 3-SAT | 1.5x to 3.0x on select distributions | Socket IPC overhead ($220\text{ ms}$); inprocessing index corruption on simplified formulas. | One-shot initial activity scaling with zero-centered prior contract. |
| **Mahalanobis OOD** *(Lee et al., NeurIPS 2018)* | Class-conditional Gaussian Modeling | Regularized Precision Matrix Estimation | Applicable to any representation | Zero Overhead (Offline Covariance) | N/A | N/A | Originally designed for computer vision classification and adversarial attack detection. | Graph-level Mahalanobis auditing to trigger safe fallback on alien industrial formulas. |
| **CaDiCaL & Kissat** *(Biere et al., 2018–2024)* | Handcrafted C99 / C++ CDCL Engines | Online VSIDS, VMTF, CHB, Chronological Backtracking | Kissat 4.0.1 / CaDiCaL 2.1.3 | Line-rate Native CPU Execution ($< 15\text{ ns}$) | SOTA Benchmark Baseline | SOTA Speed Baseline | Fragile to external memory allocation; inprocessing invalidates external variable indices. | Host engine for Folder 52; in-register Causal Implication Flow operating in L1 cache. |
| **MiniSat & SimpSolver** *(Eén & Sörensson, 2003–2005)* | 2-Watched Literals + 1-UIP Conflict Analysis | Exponential Activity Decay + Bounded Var Elimination | MiniSat 2.2 | Universal Academic Baseline ($< 25\text{ ns}$) | Universal Baseline | Universal Baseline | Variable Elimination creates discontinuous active variable index spaces. | Host engine for M3.5 V2; inprocessing-sound `minisat_adapter` protecting active heap indices. |

---

## 2. The Two Golden Systems: How Literature Synthesized into NeuroCDCL

```
                     ┌────────────────────────────────────────────────────────┐
                     │            PRIOR ML4SAT LITERATURE LESSONS             │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
         ┌──────────────────────────────────────┴──────────────────────────────────────┐
         ▼                                                                             ▼
┌─────────────────────────────────┐                                   ┌─────────────────────────────────┐
│     M3.5 VERSION 2 ENGINE       │                                   │    FOLDER 52 (NEUROCDCL KISSAT)  │
│   (Microsecond SIMD MiniSat)    │                                   │     (In-Register Causal Flow)   │
├─────────────────────────────────┤                                   ├─────────────────────────────────┤
│ • One-Shot Prior (from RLAF)    │                                   │ • Backdoor Invariants (NeuroBack│
│ • Zero-Prior Bias (from RLAF)   │                                   │ • Restart MAB (from NeuroComb)  │
│ • 648-Byte AVX2 Distillation    │                                   │ • In-Register CIF (<15 ns)      │
│   (replaces ImitSAT/Graph-Q-SAT)│                                   │ • Full Cache-Line Invariants    │
│ • Inprocessing Soundness Guard  │                                   │ • 100% Inprocessing Safe        │
│   (solves SimpSolver corruption)│                                   │ • Beats Stock Kissat on SAT-Comp│
│ • Execution Latency: 4.4 us     │                                   │ • Solves 242 / 400 benchmarks   │
└─────────────────────────────────┘                                   └─────────────────────────────────┘
```

### System 1: M3.5 Version 2 (Microsecond SIMD-Distilled MiniSat)
- **Target Domain:** Fast combinatorial structures, graph coloring, phase-transition 3-SAT, backbone benchmarks.
- **Inspirations Synthesized:**
  1. *From RLAF:* One-shot root guidance at $t=0$, bypassing the per-step latency trap of Graph-Q-SAT.
  2. *From ImitSAT:* Structural variable priority estimation, distilled from multi-megabyte PyTorch models into a 648-byte AVX2 SIMD MLP running in **$4.4\ \mu\text{s}$**.
  3. *From MiniSat SimpSolver:* The `minisat_adapter` guard that checks `isEliminated(v)` to prevent index corruption.
- **Master Empirical Result:**
  - Beats RLAF on 3COL(500) UNSAT by **-90.5% in decisions** and **$67.2\times$ in wall-clock speed**.
  - Beats ImitSAT on Sat100 by **-27.8% in decisions** and **$426.5\times$ in wall-clock speed**.

### System 2: Folder 52 (NeuroCDCL Kissat Flagship)
- **Target Domain:** Large-scale industrial SAT Competition benchmarks, dense algebraic cliques, hardware verification.
- **Inspirations Synthesized:**
  1. *From NeuroBack:* Strong backdoor identification to trigger early polynomial refutation.
  2. *From NeuroComb:* Non-stationary Multi-Armed Bandit restart scheduling.
  3. *From Armin Biere's Kissat:* In-register bitset Causal Implication Flow (CIF) operating at line-rate ($<15\text{ ns}$ per step) with zero cache evictions.
- **Master Empirical Result:**
  - Ties NeuroBack's state-of-the-art **6.3 decisions** on CliqueFormula without needing any GNN inference.
  - Fully refutes JNH in **0.0 decisions (8.32 ms)**.
  - Solves **242 / 400 SAT Competition benchmarks**, achieving 11 direct knockouts over Stock Kissat.

---

## 3. Directory Navigation for Paper Notes

All individual detailed research dossiers can be accessed directly:
- [`01. RLAF (ICLR 2026) Detailed Notes`](rlaf_iclr2026/DETAILED_RESEARCH_NOTES.md)
- [`02. ImitSAT (SAT 2020) Detailed Notes`](imitsat_sat2020/DETAILED_RESEARCH_NOTES.md)
- [`03. NeuroBack (NeurIPS 2021) Detailed Notes`](neuroback_neurips2021/DETAILED_RESEARCH_NOTES.md)
- [`04. Graph-Q-SAT (NeurIPS 2020) Detailed Notes`](graph_q_sat_neurips2020/DETAILED_RESEARCH_NOTES.md)
- [`05. NeuroComb (ICLR 2024) Detailed Notes`](neurocomb_neurips2021/DETAILED_RESEARCH_NOTES.md)
- [`06. NeuroCore (SAT 2019) Detailed Notes`](neurocore_sat2019/DETAILED_RESEARCH_NOTES.md)
- [`07. NeuroSAT (ICLR 2019) Detailed Notes`](neurosat_iclr2019/DETAILED_RESEARCH_NOTES.md)
- [`08. Mahalanobis OOD (NeurIPS 2018) Detailed Notes`](mahalanobis_ood_neurips2018/DETAILED_RESEARCH_NOTES.md)
- [`09. CaDiCaL & Kissat Foundations Detailed Notes`](cadical_kissat_foundations/DETAILED_RESEARCH_NOTES.md)
- [`10. MiniSat & SimpSolver Foundations Detailed Notes`](minisat_simp_foundations/DETAILED_RESEARCH_NOTES.md)
