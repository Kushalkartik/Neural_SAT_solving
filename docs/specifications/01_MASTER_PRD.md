# ⚠️ V2.0 AUTHORITATIVE UPGRADE NOTICE — READ BEFORE USING THIS DOCUMENT

**This document is the Master PRD (The Project Constitution) for DAC 203 Project 14: Neural Branching Heuristics for SAT.**  
The V2.0 controller is the system of record for the final research candidate. Earlier formulations remain only as historical/baseline material unless explicitly re-used as an ablation.

**Canonical Final Controller Architecture:**  
`Trust (OOD + uncertainty + search-turbulence) → Refresh (empirically learned future-benefit estimate vs measured refresh cost) → Branch (neural / VSIDS; optional blending only as a secondary ablation).`

---

# DAC 203 | ARTIFICIAL INTELLIGENCE & ADVANCED ALGORITHMS
## PROJECT 14: MASTER PRODUCT REQUIREMENTS DOCUMENT (PRD) — THE PROJECT CONSTITUTION
### Neural Branching Heuristics for Boolean Satisfiability (SAT)

| **Field** | **Value** |
| :--- | :--- |
| **Document ID** | P14-D01 |
| **Version** | 2.0 (Authoritative) |
| **Status** | Project Constitution — System of Record |
| **Target Venues** | Premier AI/Formal Methods Conferences (AAAI / SAT / CAV / NeurIPS) |
| **Academic Course** | DAC 203 — Department of Computer Science & Engineering |

---

## 1. Executive Summary & Core Research Problem

Boolean Satisfiability (SAT) is the foundational NP-complete problem that powers modern hardware verification, automated software synthesis, bounded model checking (BMC), and cryptographic analysis. Modern complete solvers rely on Conflict-Driven Clause Learning (CDCL): Boolean Constraint Propagation (BCP) via two-literal watching, conflict analysis via the 1-Empirical Implication Graph (1-UIP), non-chronological backjumping, and dynamic variable branching heuristics such as VSIDS.

While deep reinforcement learning over bipartite graph representations (e.g., Graph-Q-SAT, Kurin et al., NeurIPS 2020) can reduce search decision counts by over 50%, practical deployment is hindered by the **Neural Latency Wall**:
1. Native C++ BCP executes in 15 to 30 nanoseconds per implication.
2. A single PyTorch GNN forward pass takes 10 to 50 milliseconds per graph query—a latency gap of $10^5\times$.
3. Unbudgeted per-step neural branching results in massive wall-clock slowdowns ($74.6\times$ on MiniSat) and catastrophic search degradation on early-solving formulas.

**Project 14 Objective:**
Design, implement, and rigorously evaluate an adaptive cost-aware neural guidance architecture that achieves genuine wall-clock speedup and significant decision reductions over reference classical CDCL baselines without falling into the latency trap.

---

## 2. Core Research Questions (RQs)

- **RQ1 (Faithful Baseline Reproduction):** Can a deep reinforcement learning policy over bipartite graphs (Graph-Q-SAT) be faithfully reproduced within a native CDCL solver to achieve >50% search decision reduction on standardized benchmarks?
- **RQ2 (Computational Overhead Quantification):** What is the exact empirical wall-clock slowdown imposed by recurrent GNN inference, and where does the cost-benefit trade-off break down?
- **RQ3 (Novel Cost-Aware Controller):** Can an adaptive, cost-aware controller (Trust $\to$ Refresh $\to$ Branch) retain structural neural guidance while eliminating unnecessary neural computation, achieving wall-clock parity or superiority over pure classical CDCL?
- **RQ4 (Generalization & SOTA Integration):** Can learned guidance scale to industrial competition benchmarks and modern SOTA solvers (CaDiCaL 2.1.3, Kissat 4.0.1) without catastrophic out-of-distribution failure?

---

## 3. The 3-Stage Controller Architecture (Trust → Refresh → Branch)

```text
                      ┌─────────────────────────────────────────┐
                      │              INPUT FORMULA              │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │    STAGE 1: TRUST GATE      │
                            │  • Mahalanobis Distance D_M │
                            │  • Policy Entropy H(pi)     │
                            │  • Conflict Spike Detection │
                            └──────────────┬──────────────┘
                                           │
                       ┌───────────────────┴───────────────────┐
                  OOD / High Uncertainty           In-Distribution
                       │                                       │
                       ▼                                       ▼
            ┌─────────────────────┐              ┌─────────────────────────────┐
            │ FALLBACK: C++ VSIDS │              │   STAGE 2: REFRESH GATE     │
            │ (Zero Neural Cost)  │              │  • Value-of-Computation ΔV │
            │                     │              │  • Horizon H in [200, 500]  │
            └─────────────────────┘              └─────────────┬───────────────┘
                                                               │
                                               ┌───────────────┴───────────────┐
                                          Stale / High Cost                Low Cost
                                               │                               │
                                               ▼                               ▼
                                    ┌─────────────────────┐       ┌─────────────────────────┐
                                    │ REUSE STORED PRIORS │       │ DISPATCH GNN REFRESH    │
                                    │ (Ranked Priority Q) │       │ (Extract Top-K Literals)│
                                    └──────────┬──────────┘       └────────────┬────────────┘
                                               │                               │
                                               └───────────────┬───────────────┘
                                                               │
                                                               ▼
                                                ┌─────────────────────────────┐
                                                │   STAGE 3: CDCL EXECUTION   │
                                                │ MiniSat BCP + 1-UIP Conflict│
                                                └─────────────────────────────┘
```

1. **Stage 1: Trust Gate (OOD & Epistemic Uncertainty):**
   - Evaluates regularized Mahalanobis distance $D_M(z)$ in latent graph embedding space.
   - Computes policy entropy $\mathcal{H}(\pi)$ across unassigned candidate literals.
   - Detects search turbulence (conflict spikes).
   - If untrusted, dispatches zero-overhead classical VSIDS immediately.

2. **Stage 2: Refresh Gate (Cost-Benefit Value of Computation):**
   - Compares predicted downstream conflict reduction against measured inference latency $T_{\text{GNN}}$.
   - Bounded by a strict continuation horizon ($H \in [200, 500]$ conflicts).
   - Prevents stale guidance accumulation while suppressing redundant GNN calls.

3. **Stage 3: Branch Gate (Zero-Allocation Execution):**
   - Bounded decision execution latency ($\le 0.1\text{ ms}$).
   - Integrates learned guidance directly into the native solver priority queue or phase cache.

---

## 4. Progressive Milestone Structure & Definitions of Done

- **Milestone M0: Classical MiniSat CDCL Baseline & Telemetry**
  - Native C-ABI DLL (`minisat_adapter.dll`) with microsecond timers.
  - Zero-heuristics degradation: measured telemetry overhead $< 2.0\%$.
  - 100% solve rate on golden validation suite.

- **Milestone M1: Faithful Graph-Q-SAT Reproduction**
  - Vectorized bipartite MetaLayer GNN + DQN training over 50,000 updates.
  - Verification of $\ge 50\%$ decision reduction on held-out SATLIB benchmarks.
  - Complete forensic audit documenting the exact $74.6\times$ wall-clock latency tax.

- **Milestone M1.5: Fixed-$K$ Neural-to-Classical Handoff Study**
  - Characterization of the root-dominance effect across 1,100 problem solves.
  - Demonstration that the first $K \in [5, 15]$ decisions capture the vast majority of neural search pruning.

- **Milestone M2: One-Shot Neural Priors (M2A, M2B, M2C)**
  - M2A: One-shot initialization of VSIDS activity scores from initial graph embedding ($99.5\%$ reduction in neural calls).
  - M2B: Dedicated dual-head GNN predicting both variable rankings and polarities via algorithm feedback.
  - M2C: True RLAF-style reinforcement learning (GRPO over 3 seeds) with dynamic conflict bump modulation.

- **Milestone M3 / M3.5: Compact Native C++ Engine & Adaptive Refresh**
  - Inlined zero-allocation inference in C++ ($< 25\text{ ns}$ per clause / $< 0.3\text{ ms}$ per formula).
  - Wall-clock parity and speedup over classical reference solvers.

- **Milestone M4 / M5: Full V2 Adaptive Controller & Industrial SOTA Validation**
  - Integration into state-of-the-art engines (CaDiCaL 2.1.3, Kissat 4.0.1).
  - Deployment of NeuroPrune: Deep clause utility ranking with an adaptive watchdog controller.
  - Hard timeout salvages and net decision reduction on SAT Competition benchmarks.

---

## 5. Mandatory Deliverables & Evidence Contract

1. **Working Codebase:** Clean, modular repository in `07_SRC/` with automated build scripts.
2. **Comprehensive Test Suite:** Unit, component, golden, and performance sanity tests passing in `08_TESTS/`.
3. **Verified Empirical Results:** All headline claims backed by frozen CSV/JSON data in `12_RESULTS/` and reproducible execution configs.
4. **Publication-Ready Final Report:** Complete 6-page academic paper in `18_REPORT/` adhering to Document 15.
5. **Final Slide Deck:** Professional presentation package in `20_FINAL_DELIVERY/PRESENTATION/`.
6. **Automated Reproducibility:** Single-command replication scripts in `17_REPRODUCIBILITY/`.
