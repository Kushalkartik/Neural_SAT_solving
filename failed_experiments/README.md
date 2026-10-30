# The Master Archive of Failed Experiments & Negative Results

> **Repository:** `neuro-cdcl/failed_experiments`  
> **Topic:** Exhaustive Forensic Autopsy of 28+ Paradigms, 150+ Experimental Iterations, 24 Cloud Runs, and the Single-Core Wall-Clock Ceiling  
> **Primary Investigator:** Kushal Agarwal (Team 14 Lead, DAC-203 Project, IIT Roorkee)  
> **Source Repositories Audited:**  
> 1. `c:\Users\Kushal Agarwal\Desktop\DAC-203 PROJECT` (Production Milestones M0 through M5, Kissat Folders 20 to 52)  
> 2. `c:\Users\Kushal Agarwal\Desktop\DAC-203-P14-M3.5-LATENCY-LAB` (Continuous Autonomous Latency & Decision Minimization Laboratory)  

---

## 1. Executive Purpose: Why Document 150+ Failed Experiments?

The machine learning for SAT (ML4SAT) literature has long suffered from publication bias: researchers report minor decision-count reductions on small synthetic benchmarks while omitting wall-clock execution time, memory footprints, and fair comparisons against modern production CDCL solvers (such as Armin Biere's Kissat and CaDiCaL).

This repository documents an exhaustive empirical campaign. Over hundreds of experiments across two comprehensive laboratory environments, we systematically evaluated:
* **Deep Neural Networks in the CDCL Inner Loop** (Graph-Q-SAT, One-shot GNNs, Imitation Learning, RLAF, NeuroBack).
* **Cross-Disciplinary Physical & Mathematical Theories** (Survey Propagation cavity fields, PID control-theoretic restart regulators, Shannon information density, Spectral Fiedler cuts, Active Inference).
* **Structural Circuit Partitioning & Algebraic Methods** (Hypergraph cutset decomposition, Neuro-FRAIG simulation, Neural Extended Resolution / BVA, Algebraic XOR Gaussian elimination).
* **Hardware-Aware Micro-Architectures & Low-Latency Engines** (Flat L1-resident array solvers, AVX2 SIMD MLPs, compiler loop unrolling, DAgger lookahead trees).
* **Native Kissat In-Register Nano-Stacks** (Folders 20 through 52: Causal Implication Flow, Micro-Neural Causal Flow, Contextual MAB restarts, Tiered LBD reduction, Proof reservoirs).

Every single attempt to beat stock Kissat 4.0.1 on single-core industrial wall-clock time ultimately collided with a set of fundamental mathematical, information-theoretic, and micro-architectural boundaries. 

Documenting these failure mechanisms with exact telemetry, mathematical theorems, and raw code autopsies establishes the **true Pareto frontier of the field** and provides the community with an unassailable roadmap of what does not work and why.

---

## 2. Complete Taxonomy of Evaluated Solvers (30+ Engines)

```
                                      THE SOLVER TAXONOMY
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
 [CLASSICAL BASELINES]             [MINISAT / M3.5 LINEAGE]              [KISSAT / NANO LINEAGE]
 * MiniSat 2.2.0 Stock             * Graph-Q-SAT (M1)                   * Kissat MAB Restarts (F22)
 * Glucose 4.2.1 Industrial        * Fixed-K Handoff (M1.5)             * Causal Implication Flow (F27)
 * CaDiCaL 1.9.5 / 2.1.3 Stock     * One-Shot Prior (M2A)               * Micro-Neural Causal Flow (F28)
 * Kissat 3.1.1 / 4.0.1 Stock      * NeuroComb Prior (M2B)              * Path A: Clause Attention (F29)
                                   * RLAF In-Solver (M2C)               * Path B: Graph Adaptive (F30)
 [EXTERNAL REPLICATIONS]           * Native AVX2 SIMD (M3)              * Multi-Scale Simplex (F32)
 * RLAF (ICLR 2026)                * FlatL1Solver (Sprint 13)           * Distilled Student MAB (F37)
 * ImitSAT (SAT 2020)              * M3.5 Version 1 & 2                 * Precision Student (F39)
 * NeuroBack (NeurIPS 2021)        * Adaptive Mid-Refresh (M4)          * Cooperative Dual-Core (F40-F44)
                                   * ChronoBT + OTFS Unified (M5)       * Multi-Point Nano-Stack (F49)
                                                                        * Tiered LBD Stack (F50)
                                                                        * Decoupled Proof Reservoir (F51)
                                                                        * NeuroCDCL Unified (F52)
```

---

## 3. Master Scorecard of All Attempted Paradigms

| # | Paradigm / Technical Approach | Literature / Foundation | What Was Attempted | Aggregate Decision Impact | Aggregate Wall-Clock Impact | Primary Forensic Failure Mechanism |
|---|---|---|---|---|---|---|
| **1** | **Graph-Q-SAT (M1)** | Lederman et al. (*NeurIPS 2020*) | In-loop GNN + DQN branching at every search node | -48% on 50-var toys; 0% on industrial | **500x–1,000x SLOWER** (412s vs 2.6s on 163-suite) | **Fatal Latency Shock**: 10 ms GNN vs 30 ns CDCL inner loop. |
| **2** | **One-Shot GNN Policy (M2A)** | Selsam et al. (*ICLR 2019 / NeuroSAT*) | Root-only GNN scoring for static variable ranking | **+27% MORE decisions** on uf50 vs VSIDS | Neutral latency; negative decisions | **Heuristic Staleness**: Static ranking degrades after early backtracks. |
| **3** | **Imitation Learning (M2B)** | Yolcu & P\u00f3czos (*SAT 2020*) | Supervised MLP learning on MiniSat trace decisions | Identical to MiniSat VSIDS (30.59 vs 30.59) | ~5% inference overhead | **Imitation Ceiling**: Student cannot surpass the algorithmic teacher. |
| **4** | **RLAF Action Feedback (M2C)** | T\u00f6nshoff et al. (*ICLR 2026*) | In-solver conflict activity multipliers ($w_i$) | **+28.9% MORE decisions** on industrial | Severe degradation on structured instances | **Intermediate Cut-Point Paradox**: Dilutes 1-UIP conflict learning. |
| **5** | **Native AVX2 Structural MLP (M3)** | Hinton (*Distillation 2015*) | C++ AVX2 SIMD embedded neural forward pass | Negligible change in decision quality | 4.15 \u03bcs latency (fast for ML, but 100x > Kissat) | **Capacity Deficit**: Sub-kilobyte weights cannot model global hypergraphs. |
| **6** | **Dual Engine Solver (M3.5 V1/V2)** | Hybrid CDCL-ML | MiniSat 2.2 + Native AVX2 branching engine | -47% vs MiniSat on 163 suite (139k vs 262k) | 947 ms vs 2,600 ms (Beat MiniSat; Lost to Kissat) | **Industrial Scale Wall**: Solved only 22.3% of Industrial 130 instances. |
| **7** | **Adaptive Refresh Controller (M4)** | Cost-Aware RL / Bandits | Event-triggered neural refresh on conflict bursts | 146k vs 139k decisions (worse than M3.5) | 1,096 ms vs 947 ms | **Monitoring Overhead & Cache Thrashing**: Re-scoring evicts L1/L2 watchlists. |
| **8** | **Teacher Lookahead & DAgger (M4 Lab)** | Ross et al. (*AISTATS 2011*) | 2-sided lookahead distillation + MoE routing | 53.6k vs 60.0k decisions (**only 1.12x reduction**) | Sub-second on 163 suite; failed 5x–10x target | **Resolution Lower Bounds**: Hit the minimal proof-complexity floor. |
| **9** | **SOTA Engine Sprints (M5.0–M5.3)** | ChronoBT, OTFS, Probing | Advanced CDCL algorithmic mechanics in C++ | 143k on 163-suite; 6.5M on Industrial 130 | 86.6s on Industrial 130 (78.5% solve rate) | **Zero Neural Value**: All gains were purely from classical ChronoBT and OTFS. |
| **10**| **Neural Backbone Seeding (Dir A)** | Selsam & Bj\u00f8rner (*SAT 2019*) | Pre-search backbone classification + phase bias | **+73% MORE decisions** (103k vs 60k on 163) | 1.67s wall time; 6 instances violated soundness | **Phase Brittle-Point**: 25% error traps search in exponential dead subtrees. |
| **11**| **Neural Cube-and-Conquer (Dir B)** | Heule et al. (*HVC 2011 / March*) | Root GNN cutset selection into $2^k$ cubes | **+188% MORE serial decisions** (173k vs 60k) | Makespan decs 44k; wall time 2.3s vs 412s M1 | **Search Redundancy**: Total work tripled; sub-cubes remained hard. |
| **12**| **Input C&C + RLAF Guidance (Lab)** | Causal DAG + RLAF | Cubing on AC-GNN primary inputs + Kissat | -15.8% decisions on 1 problem; timeout on 3-SAT | 413 ms vs 13.7 ms (Process spawning + GNN cost) | **Classical Degree Dominance**: Classical max-degree cubing (-28.2%) beat neural. |
| **13**| **Tri-Paradigmatic Suite (D1/D2/D3)** | Spectral, Shannon Entropy, DAG | Information-theoretic and causal branching | 307k–2.08M decisions across 32 benchmarks | 47s–52s wall time (Kissat: 41s, CaDiCaL: 46s) | **Lower Solve Rate**: Solved only 15–17 vs Stock Kissat's 19 instances. |
| **14**| **Super D2 Fast Entropy Engine** | Friston (*Active Inference 2010*) | SIMD-accelerated entropy branching | 1.18M decs (87.5% solved on 32-suite) | 15.5s vs Kissat 41.5s on 32-suite; crashed on 501 | **Asymptotic Explosion**: $O(V \cdot C)$ calculation froze on 100k-variable CNFs. |
| **15**| **Neuro-FRAIG Preprocessing** | Mishchenko et al. (*ABC / DAC*) | AC-GNN input detection + 64-bit AIG simulation | **0.0% decision reduction** on industrial BMC | 18 ms preprocessing; Kissat still timed out @ 30s | **Domain Redundancy**: Industrial RTL formulas are pre-synthesized (0 duplicate wires). |
| **16**| **Neural Extended Resolution (ER)** | Tseitin 1968; BVA | BVA pair scoring: $z \leftrightarrow l_1 \vee l_2$ + 3 clauses | **+404% MORE decisions on 3-SAT; timeouts** | 11x slower; Kissat timed out on Crypto & Crafted | **Definition Pollution**: New variables dilute VSIDS scores and pollute L1 cache. |
| **17**| **Algebraic XOR & Multi-Core IPC** | Soos (*CryptoMiniSat*); Glucose | Gaussian elimination + IPC shared clause queue | 0 XORs found on non-crypto; IPC lock contention | Higher CPU usage, slower wall-clock time | **Structural Specificity**: Zero affine structure in general SAT; lock overhead. |
| **18**| **L1-Resident Flat CDCL (`FlatL1`)** | Hardware-Aware C++17 | Flat contiguous arrays in 32 KB L1 data cache | Identical decisions to MiniSat (28.09 vs 28.09) | **3.05 \u03bcs median latency** (30.5x faster than M0) | **L1 Cache Spilling**: Scales to $N \le 100$; explodes into DRAM at $N > 1,000$. |
| **19**| **Cross-Disciplinary Science Solvers** | Physics (SP), Control (PID), Info Theory | Survey propagation, PID restart controller, Info-Bottleneck | **5x–7x MORE decisions** than CaDiCaL (1.5M–2.1M vs 327k) | Slower wall time (60.4s–61.1s vs 53.5s) | **Bethe Lattice Assumption Violation**: Cavity fields fail on modular logic. |
| **20**| **Neural Macro Cutset Partitioning** | Hypergraph Bisection / Decomp | Hypergraph separator decomposition into sub-solvers | **+30% to +300% MORE decisions**; timeouts | Solved synthetic toys; timed out on all industrial | **Severed Implication Boundaries**: High-density functional constraints cut across. |
| **21**| **Global Solution Policy (Size-Shift)** | Global Model Prediction | End-to-end vector assignment prediction | Succeeded on synthetic size-shift (339 vs 57k decs) | 3.88 ms on synthetic 1000-var formulas | **UNSAT Incompatibility**: Predicts assignments that do not exist; 0% on UNSAT. |
| **22**| **Compiler & Micro-Architecture Sprints**| Compiler Optimization / SIMD | Loop unrolling, inlined binary watchers, BVE | MiniSat decisions reduced on small formulas | Loop unrolling blew ICache (7x tail latency); BVE won 2.2x | **ICache Thrashing**: Unrolling increased instruction footprint past L1I limits. |
| **23**| **Folder 22: Kissat MAB Restarts** | Multi-Armed Bandit / UCB | Contextual MAB dynamic Luby vs Geometric restart | Small decision variance reduction | Neutral wall-clock impact | **Kissat Target Phase Saturation**: Kissat's stable/focused modes already dominate. |
| **24**| **Folder 24: Neural CEGAR Lab** | Counterexample-Guided Abstraction | Bounding sub-formula abstraction via neural cores | +45% decisions on crafted instances | Severe timeout rate on industrial | **Refutation Amplification**: Spurious counterexamples multiplied proof depth. |
| **25**| **Folder 27/28: Causal Implication Flow**| Pearl (*Causality*); Implication DAG | Tracking causal forward flow on 2WL chains | -12% to -24% decisions on selected families | +6% to +12% runtime overhead | **Deduce-Loop Overhead**: Even 1 extra conditional branch in deduce slows BCP. |
| **26**| **Folder 29/30: Paths A & B (NCA/GAC)** | Clause Attention & Adaptive Control | In-solver clause attention weights and backprop | Divergent behavior across seeds | +15% runtime overhead | **Gradient Stagnation**: On-line backprop cannot converge within 100 conflicts. |
| **27**| **Folder 40–44: Cooperative Dual-Core** | Asymmetric Portfolio & Shared Lemmas | Dual-thread Kissat exchanging LBD $\le 2$ clauses | Solved 18/27 boundary instances | Thread synchronization and memory contention | **Memory Contention**: Single-core parity not achieved; dual-core required. |
| **28**| **Folder 49–51: 24-Run Cloud Fleet** | Multi-Point Nano-Stack & Proof Reservoir | Tiered LBD filtering, nano clause scoring, mode switching | Peak 15/27 on boundary suite (Run 10); 22/27 fleet union | Persistent seesaw: 1,440s to 1,600s PAR-2 | **The BCP vs Structural Lemma Seesaw**: Retention slows 2WL; pruning traps SAT. |
| **29**| **Folder 52: NeuroCDCL Unified** | 4-Point Embedded Nano-Stack | MNCF flow + Clause Scorer + MAB + Mode Switcher | **-442,797,312 decisions (-21.55%)** on 400 Grand Suite | 242 solved vs Kissat Stock's 243 (1084.07s vs 1078.41s) | **The 3.8% Wall-Clock Ceiling**: 21.5% decision gain canceled by 3.8% overhead. |

---

## 4. Structure of this Forensic Archive

This archive is partitioned into 9 exhaustive technical dossiers:

1. [`01_DEEP_LEARNING_AND_GNN_IN_LOOP.md`](01_DEEP_LEARNING_AND_GNN_IN_LOOP.md)  
   *Detailed autopsies of Graph-Q-SAT, One-Shot GNNs, Imitation Learning, RLAF, Backbone Seeding, and Neural Cube-and-Conquer.*
2. [`02_CROSS_DISCIPLINARY_AND_PHYSICAL_THEORIES.md`](02_CROSS_DISCIPLINARY_AND_PHYSICAL_THEORIES.md)  
   *Mathematical failures of Survey Propagation (Physics), PID Controllers (Control Theory), Shannon Entropy (Information Theory), and Spectral Graph Theory.*
3. [`03_STRUCTURAL_ALGEBRAIC_AND_PREPROCESSING.md`](03_STRUCTURAL_ALGEBRAIC_AND_PREPROCESSING.md)  
   *Failures of Macro Cutset Partitioning, Neuro-FRAIG Preprocessing, Neural Extended Resolution (BVA), and Global Solution Policies.*
4. [`04_LOW_LATENCY_SYSTEMS_AND_MICROARCH.md`](04_LOW_LATENCY_SYSTEMS_AND_MICROARCH.md)  
   *Hardware-level investigations: Flat L1 CDCL, AVX2 SIMD distillation, Compiler loop unrolling, DAgger lookahead trees, and the WalkSAT 200x UNSAT trap.*
5. [`05_KISSAT_NANO_NEURAL_AND_RUN1_TO_RUN24.md`](05_KISSAT_NANO_NEURAL_AND_RUN1_TO_RUN24.md)  
   *The complete Kissat neural integration campaign (Folders 20–52), the 24-run cloud VM fleet on the Boundary 27 suite, the BCP-vs-Lemma seesaw, and the 400-instance Grand Tournament.*
6. [`06_FORMAL_MATHEMATICAL_THEOREMS_OF_FAILURE.md`](06_FORMAL_MATHEMATICAL_THEOREMS_OF_FAILURE.md)  
   *Rigorous mathematical statements and proofs of the 8 fundamental failure theorems governing neural and heuristic intervention in CDCL.*
7. [`07_CODE_LEVEL_AUTOPSIES_AND_PATCHES.md`](07_CODE_LEVEL_AUTOPSIES_AND_PATCHES.md)  
   *Line-by-line C/C++ source code autopsies, patch diffs, and exact telemetry: Causal Implication Flow in `deduce.c`, Inlined Clause Attention in `patch_deduce.py`, The Conservation Simplex in `bump.c`, Gate Extraction in `structural_gate_cuber.cpp`, XOR Gaussian Reduction, Neural CEGAR loops, and Tiered LBD cache thrashing.*
8. [`08_POLYNOMIAL_CALCULUS_AND_ADVANCED_ALGEBRAIC.md`](08_POLYNOMIAL_CALCULUS_AND_ADVANCED_ALGEBRAIC.md)  
   *Polynomial Calculus with Resolution (BSPCR / Deep PCR / Neuro-PCR V2): Degree-1 Affine $\mathbb{F}_2$ equations, Degree-2 Bilinear At-Most-One (AMO) cliques, Nullstellensatz degree lower bounds, Gröbner basis monomial explosion, and bounded-degree completeness collapse.*
9. [`09_RL_DAGGER_AND_AUTONOMOUS_M4_MARATHON.md`](09_RL_DAGGER_AND_AUTONOMOUS_M4_MARATHON.md)  
   *The 26-hour M4 Autonomous Laboratory Marathon (Versions V1 through V7): Tree Teacher DAgger, Mixture of Experts (MoE) neural routers (`E_STRUCT`, `E_RANDOM_HARD`, `E_INDIST`), The UNSAT Training Data Desert, The 4-Instance Cap Trap (195k artificial decisions), and Reinforcement Learning credit assignment limits.*
