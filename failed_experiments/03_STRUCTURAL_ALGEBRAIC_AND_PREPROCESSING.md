# Technical Dossier 03: Structural Partitioning, Algebraic Methods & Preprocessing

> **Archive ID:** `FE-DOSSIER-03`  
> **Scope:** Neural Macro Cutset Partitioning (Lab 33), Neuro-FRAIG Circuit Preprocessing, Neural Extended Resolution / BVA, Algebraic XOR Gaussian Elimination, and Global Solution Polarity Policies (Lab 32).  
> **Primary Laboratory:** `DAC-203-P14-M3.5-LATENCY-LAB\19_ORACLE_TO_STUDENT_SOTA_DECISION_LAB\32_GLOBAL_SOLUTION_POLICY` & `33_NEURAL_MACRO_PARTITIONING`  
> **Primary Benchmark Platforms:** Industrial BMC Circuits, Cryptographic Miters, Synthetic Size-Shift Suites ($N=50 \to 1000$), SMT Bit-Vector Formulas.  

---

## 1. Executive Summary

A recurring theme in automated reasoning research is that raw CNF formulas lose the high-level semantic structure of the original domain (e.g., circuit gates, module boundaries, arithmetic XORs, and global solution manifolds).

To address this, we designed five distinct structural and algebraic intervention strategies:
1. **Hypergraph Decomposition:** Shattering CNFs into independent sub-problems via minimal cutset separators.
2. **Circuit Gate Simulation (Neuro-FRAIG):** Detecting primary inputs with GNNs and performing 64-bit functionally reduced simulation to merge equivalent wires.
3. **Neural Extended Resolution (BVA):** Introducing new Tseitin extension variables ($z \leftrightarrow l_1 \vee l_2$) to shorten resolution proof lengths.
4. **Affine XOR Extraction:** Detecting linear systems over $\text{GF}(2)$ and solving them via parallel Gaussian elimination.
5. **Global Solution Manifold Prediction:** Predicting complete assignment vectors end-to-end to seed search polarity.

**The Universal Result:** None of these high-level abstractions survived contact with real-world industrial benchmarks. They produced either **zero impact** (due to commercial synthesis tools already having eliminated structural redundancies), **severe decision explosions (up to +404%)** due to definition pollution, or **catastrophic collapse on unsatisfiable (UNSAT) instances**.

---

## 2. Forensic Autopsies by Methodology

### 2.1 Neural Macro Cutset Partitioning (`Lab 33: MACRO_CUTSET_CONQUER`)
* **Theoretical Formulation:**
  * Framed industrial SAT formulas as hypergraphs $\mathcal{H} = (V, \mathcal{E})$, where vertices are variables and hyperedges are clauses.
  * Executed spectral hypergraph bisection to identify a minimal vertex cutset $\mathcal{S} \subset V$ that partitions the residual graph into two disconnected components $V_1, V_2$ ($\mathcal{H} \setminus \mathcal{S} = \mathcal{H}_1 \cup \mathcal{H}_2$).
  * Assigned the cutset $\mathcal{S}$ and dispatched $\mathcal{H}_1$ and $\mathcal{H}_2$ to independent CDCL instances.
* **Empirical Results (24 Benchmarks — `HEAD_TO_HEAD_EVALUATION.csv`):**
  * *On Synthetic Formulas:*
    * `synthetic_n0500.cnf`: Baseline MiniSat took 615 decisions; Macro Cutset took **2,428 decisions (3.95x WORSE)**.
    * `synthetic_n1000.cnf`: Baseline MiniSat took 9,086 decisions; Macro Cutset took **11,874 decisions (+30.7% WORSE)**.
  * *On Real Industrial / SMT / Crypto Formulas:*
    * `12d79233...cnf` (Hardware Model Checking): Timed out after **502,174 decisions**.
    * `bv-term-small-rw...cnf` (SMT Bit-Vector): Timed out after **62,648 decisions**.
    * `combined-crypto1...cnf` (Cryptographic Miter): Timed out after **159,486 decisions**.
* **Why It Failed (The Non-Separability of Industrial Logic):**
  1. *Functional Coupling:* Industrial formulas encode functional transformations (e.g., ALU datapaths, memory buses, cryptographic state transitions). High-density constraints span globally across the entire hypergraph.
  2. *Artificial Cutset Explosion:* Narrow cutsets simply do not exist in real formulas. Forcing a cutset partition required $|\mathcal{S}|$ to encompass thousands of variables.
  3. *Unit Propagation Severing:* In CDCL, unit propagation across module boundaries is the primary engine of search pruning. By assigning $\mathcal{S}$ externally and severing inter-module implications, the solver lost the ability to generate cross-module conflict clauses, forcing redundant exponential search across all sub-solvers.

---

### 2.2 Neuro-FRAIG Preprocessing (Circuit Input Detection & AIG Simulation)
* **Theoretical Formulation:**
  * Literature foundation: Mishchenko et al., *"FRAIGs: A Heuristic Method for Combinational Equivalence Checking of Incomplete Logic"* (*DAC / IEEE TCAD*).
  * Used an Asymmetric Convolutional GNN (AC-GNN) to identify primary input registers from raw CNF gate encodings.
  * Extracted an And-Inverter Graph (AIG), performed 64-bit parallel random vector simulation, and merged functionally equivalent wire pairs to reduce formula size prior to CDCL search.
* **Empirical Results (SAT Competition Hardware BMC Suite):**
  * *Preprocessing Overhead:* Consumed 18.2 milliseconds per benchmark.
  * *Equivalent Wires Merged:* **0 wires merged (0.0% reduction in variables and clauses)**.
  * *Search Impact:* Total decisions and solving time were identical to stock Kissat. Kissat still timed out on all hard instances at 30 seconds.
* **Why It Failed (The Pre-Synthesis Redundancy Trap):**
  * In academic literature, FRAIGing is demonstrated on raw, unoptimized circuit descriptions.
  * Real-world industrial benchmarks submitted to the SAT Competition originate from commercial verification workflows (Synopsys Design Compiler, Cadence JasperGold, Yosys, ABC).
  * These synthesis pipelines **already apply complete structural hashing, constant folding, and Bounded Variable Elimination (BVE)** prior to CNF export.
  * The formula is already mathematically minimal with respect to combinational equivalence; spending CPU time running neural gate detection and simulation is 100% wasted effort.

---

### 2.3 Neural Extended Resolution & Bounded Variable Addition (`Neural_ER_BVA`)
* **Theoretical Formulation:**
  * Literature foundation: Tseitin (1968), Cook (1976), Manthey et al. (*Bounded Variable Addition, SAT 2012*).
  * Standard resolution operates in a fixed boolean space. Extended Resolution (ER) allows introducing new extension variables $z \leftrightarrow l_1 \vee l_2$, theoretically compressing exponential resolution proofs into polynomial size (e.g., for the Pigeonhole Principle).
  * A neural model evaluated candidate literal pairs $(l_1, l_2)$ to select extension variables that maximized resolution clause compression.
* **Empirical Results (`NEURAL_ER_MULTI_TIER_RESULTS.json`):**
  * *On Random 3-SAT:* Decisions exploded from 22,400 to **112,896 decisions (+404% MORE decisions)**.
  * *On Cryptographic & Crafted Benchmarks:* Solved 0 instances within 300 seconds; all runs timed out.
  * *Wall-Clock Overhead:* Execution was **11.4x slower** than stock Kissat.
* **Why It Failed (The Definition Pollution Theorem):**
  1. *Watched-Literal Cache Pollution:* Every extension variable $z \leftrightarrow l_1 \vee l_2$ introduces 3 new clausal constraints: $(\bar{z} \vee l_1 \vee l_2), (z \vee \bar{l}_1), (z \vee \bar{l}_2)$. Adding thousands of extension variables flooded the two-watched-literal (2WL) memory structures, destroying CPU L1/L2 cache locality.
  2. *VSIDS Activity Dilution:* In CDCL, variable bumping concentrates search pressure on the active conflict core. Extension variables, possessing high clausal connectivity, absorbed VSIDS activity without providing direct implication power, steering the decision engine away from the real asserting literals.

---

### 2.4 Algebraic XOR Gaussian Elimination & Multi-Core IPC (`XOR_Crypto_IPC`)
* **Theoretical Formulation:**
  * Literature foundation: Mate Soos, *"CryptoMiniSat"* (*SAT 2009*); Glucose parallel clause exchange.
  * Extracted parity constraints (linear equations over $\text{GF}(2)$) from the CNF:
    $$\bigoplus_{i=1}^k x_i = c$$
  * Solved the affine sub-system using bit-parallel Gaussian elimination. Coupled with an Inter-Process Communication (IPC) shared clause memory ring across 4 CPU cores.
* **Empirical Results:**
  * *On Cryptographic Benchmarks (MD5/SHA-1 inversion):* Successfully extracted XORs and reduced decision count by 32%.
  * *On General Industrial Benchmarks (400 Grand Suite):* **0 XOR constraints detected (0.0%)**.
  * *Wall-Clock Impact:* High IPC lock contention and memory synchronization overhead resulted in **15% slower wall-clock execution** than single-core stock Kissat.
* **Why It Failed (Affine Scarcity & Lock Contention):**
  * Outside of cryptographic hashing circuits, industrial SAT instances contain virtually no affine parity structure. They are dominated by Horn-like implications, mux trees, and state-machine transitions.
  * The complex Gauss-Jordan elimination matrix sat completely idle while consuming heap allocation.
  * Concurrently, multi-process clause sharing via POSIX/Windows shared memory buffers encountered severe cache-line bouncing and mutex contention, degrading the throughput of the underlying BCP engine.

---

### 2.5 Global Solution Manifold Policy (`Lab 32: GLOBAL_SOLUTION_POLICY`)
* **Theoretical Formulation:**
  * Trained an end-to-end graph neural network to predict the entire satisfying truth assignment vector $\mathbf{s}^* \in \{0, 1\}^{|V|}$ directly from formula bipartite embeddings.
  * The predicted vector was loaded into the solver's phase-saving array to guide polarity selection on all branches.
* **Empirical Results (Size-Shift Synthetic Evaluation):**
  * *On Planted Satisfiable Instances ($N=1,000$):* Achieved an extraordinary reduction—decisions plummeted from **57,053 down to 339 decisions (a 168.3x speedup)** in 3.88 ms.
  * *On Unsatisfiable (UNSAT) Instances:* **0% speedup. Total refutation failure.**
  * *On Real Industrial Formulas:* Prediction accuracy dropped to **50.2% (pure random noise)**, causing catastrophic performance degradation.
* **Why It Failed (The Satisfiability Assumption & Refutation Asymmetry):**
  1. *The UNSAT Vacuum:* In the International SAT Competition, roughly **50% of all industrial benchmarks are UNSATISFIABLE** (e.g., verification safety properties that hold, bug-free circuits). For an UNSAT formula, no satisfying assignment exists. Seeding a "solution vector" is mathematical nonsense: it forces the solver to search for a phantom assignment, leading it down an invalid path and multiplying the length of the required refutation resolution proof.
  2. *The Manifold Collapse:* The neural policy succeeded on synthetic benchmarks because the random circuit generator sampled instances from a low-dimensional manifold that the network memorized. Real-world industrial instances possess astronomical combinatorial complexity, causing the network's continuous embeddings to collapse to arbitrary noise.

---

## 3. Comparative Summary: Structural & Preprocessing Methods

| Methodology | Target Mechanism | Best Observed Case | Real Industrial Case | Root Failure Mechanism |
|---|---|---|---|---|
| **Macro Cutset Partitioning** | Graph Bisection | -15% on toy synthetic | Timed out on 100% of industrial | Severed inter-module implication chains |
| **Neuro-FRAIG Preprocessing** | Equivalence Merging | 0.0% reduction | 0.0% reduction (0 wires merged) | Circuits are already pre-synthesized |
| **Neural Extended Resolution** | Tseitin Extension ($z$) | 0 wins | +404% decisions; 11.4x slower | Definition pollution & cache thrashing |
| **Algebraic XOR / IPC** | $\text{GF}(2)$ Gaussian Elim | -32% on crypto SHA | 0 XORs found; slower wall-clock | Absence of affine structures in general SAT |
| **Global Solution Policy** | Vector Assignment | 168x gain on planted SAT | 0% on UNSAT; 50% accuracy | Cannot refutation-prove UNSAT instances |

---

## 4. Definitive Takeaway

High-level semantic abstractions (hypergraph cuts, circuit simulations, extension variables, affine equations) are powerful only when the problem domain strictly adheres to their underlying assumptions. 

In general industrial SAT solving, formulas are already pre-synthesized into compact, non-separable constraint graphs. Attempting to artificially impose high-level decompositions degrades the microscopic, hardware-optimized conflict learning loops that allow modern CDCL solvers to succeed.
