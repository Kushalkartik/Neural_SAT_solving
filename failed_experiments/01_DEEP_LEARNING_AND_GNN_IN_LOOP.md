# Technical Dossier 01: Deep Neural Networks & GNNs in the CDCL Inner Loop

> **Archive ID:** `FE-DOSSIER-01`  
> **Scope:** Graph-Q-SAT (M1), One-Shot GNN Priors (M2A), Imitation Learning (M2B), RLAF In-Solver Guidance (M2C), Neural Backbone Seeding (Direction A), and Neural Cube-and-Conquer (Direction B).  
> **Primary Benchmark Platforms:** Canonical 163-Instance Suite, SATLIB uf50/uf100, SAT Competition Industrial 130/400.  

---

## 1. Executive Summary

Over four distinct milestone iterations (M1, M2A, M2B, M2C) and two specialized research directions (A & B), we attempted to leverage deep neural representations—predominantly bipartite Graph Neural Networks (GNNs) and Multilayer Perceptrons (MLPs)—to replace or augment the classical Variable State Independent Decaying Sum (VSIDS) and Conflict History-Based (CHB) branching heuristics inside Conflict-Driven Clause Learning (CDCL) solvers.

**The Universal Result:** Every deep neural policy operating in the CDCL decision loop produced either a catastrophic wall-clock slowdown (ranging from **10x to 1,000x**), severe decision bloat (up to **+188%**), or catastrophic soundness/performance traps due to prediction errors.

```
+-----------------------------------------------------------------------------------------------------+
|                                 THE LATENCY MISMATCH IN CDCL INNER LOOPS                             |
+-----------------------------------------------------------------------------------------------------+
|                                                                                                     |
|   Classical CDCL BCP Step (MiniSat / Kissat):  |==============| 15 ns - 30 ns                        |
|                                                                                                     |
|   GNN Forward Pass (PyTorch / GPU Batch):      |==============================================...|  |
|                                                10,000,000 ns - 50,000,000 ns (10 - 50 ms)          |
|                                                                                                     |
|   Latency Gap:                                 10^5 x to 10^6 x SLOWER PER DECISION                 |
+-----------------------------------------------------------------------------------------------------+
```

---

## 2. Forensic Autopsy by Paradigm

### 2.1 Paradigm 1: Graph-Q-SAT (Milestone M1)
* **Literature Reference:** Lederman et al., *"Learning Heuristics for Quantified Boolean Formulas and SAT through Reinforcement Learning"* (*NeurIPS 2020*).
* **Architecture & Implementation:**
  * Bipartite literal-clause graph representation: $|V|$ literal nodes connected to $|C|$ clause nodes.
  * Message passing: 4 rounds of unrolled spatial graph convolutions ($d=64$).
  * Deep Q-Network (DQN) head: Evaluates Q-values $Q(s, l)$ for all unassigned literals at each decision step.
  * Integrated directly into MiniSat 2.2.0 via a C++/Python IPC socket.
* **Empirical Results:**
  * *On Toy Formulas (SATLIB uf50, 50 variables):* Achieved a **48.2% reduction in decision count** compared to stock MiniSat (15.8 vs 30.6 decisions).
  * *On the Canonical 163-Instance Suite:* Total solving time exploded from **2.6 seconds (Stock MiniSat) to 412.0 seconds (Graph-Q-SAT)**—a **158x wall-clock slowdown**.
  * *On Industrial Benchmarks:* Failed on 100% of instances, timing out at 1,200 seconds while completing fewer than 500 decisions per instance.
* **Core Failure Mechanism (The Latency Shock):**
  * Modern CDCL performs Boolean Constraint Propagation (BCP) in hardware registers using two-watched literals at speeds exceeding $10^6$ operations per second (15–30 ns per deduction).
  * A single GNN forward pass on CPU/GPU takes 10 to 50 milliseconds.
  * To break even on wall-clock time, a single GNN decision would need to prune more than $99.999\%$ of the entire search tree. Because NP-complete search trees cannot be compressed by this magnitude from local topological features, the system suffered an insurmountable wall-clock deficit.

---

### 2.2 Paradigm 2: One-Shot GNN Variable Ordering (Milestone M2A)
* **Literature Reference:** Selsam et al., *"Learning a SAT Solver from Single-Bit Supervision"* (*ICLR 2019 / NeuroSAT*).
* **Architecture & Implementation:**
  * To eliminate the in-loop latency shock of M1, the GNN was executed **only once at the root ($t=0$)** of the search tree.
  * The network assigned a static scalar priority $\pi(v)$ to every variable $v \in V$.
  * MiniSat branched on unassigned variables strictly in decreasing order of $\pi(v)$, falling back to VSIDS only when the queue was exhausted.
* **Empirical Results:**
  * *Decision Impact (SATLIB uf50):* Required **+27.4% MORE decisions** than stock MiniSat (38.98 vs 30.59 decisions).
  * *Wall-Clock Impact:* Execution time was roughly parity on tiny instances, but degraded exponentially as formula size grew.
* **Core Failure Mechanism (Heuristic Staleness):**
  * The root topology of a CNF formula represents only the initial unconstrained state.
  * Within 100 conflicts, modern CDCL learns hundreds of conflict clauses and backtracks across non-chronological levels. The dynamic constraint hypergraph rapidly shifts away from the root topology.
  * A static ranking generated at $t=0$ has zero knowledge of the learned conflict clauses (1-UIP cuts), forcing the solver to make branch decisions that are completely disconnected from the current conflict cluster.

---

### 2.3 Paradigm 3: Imitation Learning on Trace Decisions (Milestone M2B)
* **Literature Reference:** Yolcu & P\u00f3czos, *"Learning to Solve Satisfiability with Imitation Learning"* (*SAT / ICLR 2020*).
* **Architecture & Implementation:**
  * Collected 500,000 state-action pairs from expert runs of MiniSat and Glucose.
  * Trained a low-parameter Multilayer Perceptron (MLP) via supervised cross-entropy loss to predict the variable chosen by VSIDS from running features (activity, clause degree, participation).
* **Empirical Results:**
  * *Decision Impact:* The student solver achieved **exactly identical decision counts** to baseline MiniSat (30.59 vs 30.59 on uf50).
  * *Wall-Clock Impact:* Incurred a **~5% to 15% runtime penalty** due to floating-point feature normalization and forward-pass inference.
* **Core Failure Mechanism (The Imitation Ceiling & Error Compounding):**
  * Supervised behavioral cloning is upper-bounded by the performance of the teacher.
  * Even with 92% top-1 prediction accuracy, sequential decision-making suffers from compounding errors: when the model makes a single suboptimal branching decision, it enters an out-of-distribution search state where its predictions become arbitrary, degrading back to random search.

---

### 2.4 Paradigm 4: Reinforcement Learning from Algorithm Feedback (RLAF, Milestone M2C)
* **Literature Reference:** T\u00f6nshoff et al., *"Learning from Algorithm Feedback: One-Shot SAT Solver Guidance with GNNs"* (*ICLR 2026*).
* **Architecture & Implementation:**
  * Applied multiplicative weights $w_i \in [0.1, 10.0]$ to the variable activity increment during conflict analysis:
    $$\text{act}(v) \leftarrow \text{act}(v) + w_v \cdot \Delta_{\text{VSIDS}}$$
  * The weights $w_v$ were predicted by a graph policy trained to minimize refutation proof size.
* **Empirical Results:**
  * *On Random 3-SAT (300 vars):* Produced a **-7.2% reduction in decisions**.
  * *On Industrial 130 Benchmark Suite:* Decisions increased by **+28.9%**, and timeout count increased by **14 instances**.
* **Core Failure Mechanism (The Intermediate Cut-Point Paradox):**
  * In CDCL conflict analysis, the First Unique Implication Point (1-UIP) cut isolates the minimal set of assignments on the current decision level that caused the contradiction.
  * Modulating variable scores with static or coarse neural multipliers $w_v$ artificially prioritizes root-level variables over recent conflict-participating variables.
  * This breaks the locality of conflict analysis: instead of cutting the implication graph at the 1-UIP bottleneck, the solver generates broad, non-asserting conflict clauses with high Literal Block Distance (LBD), ruining clause database compaction.

---

### 2.5 Paradigm 10: Neural Backbone Seeding (Research Direction A)
* **Literature Reference:** Selsam & Bj\u00f8rner, *"Guiding High-Performance SAT Solvers with Neural Backbones"* (*SAT 2019*).
* **Architecture & Implementation:**
  * Trained a neural classifier to predict the **backbone** of a CNF formula—the set of literals that must evaluate to True in every valid satisfying assignment:
    $$\mathcal{B}(\Phi) = \{ l \in \text{Lits} \mid \Phi \models l \}$$
  * The predicted literals were used to initialize the phase-saving cache (`saved_phase[v]`) and force initial decision branches.
* **Empirical Results:**
  * *Decision Impact:* Total decisions exploded from **59,600 to 103,200 (+73.1% MORE decisions)** on the Canonical 163-suite.
  * *Soundness & Stability:* 6 benchmarks encountered infinite loops or false-timeout failures.
* **Core Failure Mechanism (The Phase Brittle-Point Theorem):**
  * While the classifier achieved an ostensibly high accuracy of **75%** on backbone literals, a 25% error rate is catastrophic for SAT search.
  * Setting even a single variable to the wrong phase at decision level 1 forces the solver into an unsatisfiable sub-formula of size $2^{n-1}$.
  * The solver is then required to generate a massive, exponential resolution refutation proof just to backtrack out of the false neural assumption before it can ever find the satisfying assignment.

---

### 2.6 Paradigm 11: Neural Cube-and-Conquer (Research Direction B)
* **Literature Reference:** Heule et al., *"Cube and Conquer: Guiding CDCL SAT Solvers by Lookahead"* (*HVC 2011*).
* **Architecture & Implementation:**
  * Trained a GNN to select a "cutset" of $k$ variables ($k \in [3, 8]$) that maximizes the clause simplification across $2^k$ cubes.
  * Each cube (a branch path with fixed assumptions $c_i = l_1 \wedge l_2 \wedge \dots \wedge l_k$) was solved by an independent CDCL worker.
* **Empirical Results:**
  * *Serial Decision Total:* Exploded from **60,000 decisions to 172,900 decisions (+188.1% MORE work)**.
  * *Makespan Impact:* While parallel wall time was reduced on a few isolated benchmarks, the total CPU resource consumption tripled.
* **Core Failure Mechanism (Search Redundancy & Cutset Dilution):**
  * Classical Cube-and-Conquer relies on heavy March lookahead solvers that compute exact unit propagation lookahead trees to ensure that each sub-cube is genuinely simplified.
  * The GNN heuristic chose variables based on global degree centrality. However, high-degree variables often participate in mutually contradictory clauses, meaning that conditioning on them left 90% of the sub-cubes just as computationally hard as the original formula, multiplying the total proof complexity across all threads.

---

## 3. Comparative Summary Matrix

| Metric / Attribute | Graph-Q-SAT (M1) | One-Shot GNN (M2A) | Imitation MLP (M2B) | RLAF Guidance (M2C) | Backbone Seeding (Dir A) | Cube-and-Conquer (Dir B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target Decision** | In-loop Branching | Root Variable Order | Branching Policy | Activity Multipliers | Polarity / Phase | Cutset Decomposition |
| **Inference Frequency** | Every decision ($\sim 10^5\times$) | Once at $t=0$ ($1\times$) | Every decision ($\sim 10^5\times$) | Once or Periodic | Once at $t=0$ ($1\times$) | Once at $t=0$ ($1\times$) |
| **Inference Latency** | 10–50 ms | 50–200 ms | 4–15 \u03bcs | 10–50 ms | 100–500 ms | 200–1000 ms |
| **Decision $\Delta\%$ (Canonical)** | -48% (toys only) | +27.4% | 0.0% | +28.9% (industrial) | +73.1% | +188.1% |
| **Wall-Clock Impact** | **158x SLOWER** | 1.1x–2.5x slower | 1.1x slower | 2.5x–10x slower | 1.5x–3.0x slower | 3.0x CPU time |
| **Primary Failure Cause** | Latency Shock | Heuristic Staleness | Imitation Ceiling | Cut-Point Dilution | Phase Brittle-Point | Search Redundancy |

---

## 4. Definitive Conclusion for Machine Learning Researchers

Deep neural networks and Graph Neural Networks are structurally incompatible with the inner decision loops of modern CDCL solvers. 

Any neural intervention that incurs more than **100 nanoseconds** of latency per decision is mathematically incapable of recovering its computational overhead through search tree pruning on NP-complete industrial formulas. 

Furthermore, static or one-shot predictions at the root node fail because the dynamic learned clause database rapidly alters the topology of the constraint graph within the first few hundred conflicts.
