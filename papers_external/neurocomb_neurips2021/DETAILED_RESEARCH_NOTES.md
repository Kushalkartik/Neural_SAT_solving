# Research Dossier: NeuroComb (ICLR 2024 / NeurIPS 2021)

**Title:** NeuroComb: Improving SAT Solving with Graph Neural Networks  
**Authors:** Wenxi Wang, Yang Hu, Mohit Tiwari, Sarfraz Khurshid, Kenneth L. McMillan, Risto Miikkulainen (UT Austin)  
**Venue:** International Conference on Learning Representations (ICLR 2024) / Preprinted NeurIPS 2021  
**Primary Artifact:** [`neurocomb_neurips2021.pdf`](neurocomb_neurips2021.pdf)

---

## 1. Executive Summary & Research Motivation

Prior to NeuroComb, learned heuristics either executed statically at the root (ignoring subsequent conflict dynamics) or attempted to replace the branching heuristic entirely.

**NeuroComb's Core Thesis:**  
Modern CDCL solvers already possess powerful adaptive heuristics (VSIDS for recency, LRB for learning rate, CHB for conflict history). Rather than replacing these heuristics, use a Graph Neural Network to provide **hybrid dual guidance**:
1. **Variable Selection Guidance:** Enhancing initial activity scores for critical variables.
2. **Clause Deletion Guidance:** Predicting which learned clauses are redundant versus indispensable to guide database reduction.
3. **Adaptive Bandit Arbitration:** Using a **Multi-Armed Bandit (MAB)** to dynamically arbitrate between neural guidance and classical heuristics based on real-time solver performance.

---

## 2. Neural Architecture & Dual-Prediction Heads

### 2.1 Bipartite GNN Backbone
NeuroComb employs an alternating GNN operating on the standard literal-clause bipartite graph:
$$\mathbf{c}_j^{(t+1)} = \text{ReLU}\left( \mathbf{W}_C \mathbf{c}_j^{(t)} + \frac{1}{|c_j|} \sum_{l \in c_j} \mathbf{W}_{LC} \mathbf{l}_i^{(t)} \right)$$
$$\mathbf{l}_i^{(t+1)} = \text{ReLU}\left( \mathbf{W}_L \mathbf{l}_i^{(t)} + \frac{1}{|C(l_i)|} \sum_{c \in C(l_i)} \mathbf{W}_{CL} \mathbf{c}_j^{(t+1)} \right)$$

### 2.2 Dual Prediction Tasks
1. **Variable Importance Head:** Predicts the probability that variable $x$ appears in the optimal resolution proof:
   $$\hat{y}_{\text{var}}(x) = \sigma\left( \mathbf{w}_v^T [\mathbf{l}_x^{(T)} \,\|\, \mathbf{l}_{\neg x}^{(T)}] + b_v \right)$$
2. **Clause Redundancy Head:** Predicts whether a clause $c$ can be deleted without increasing decision steps:
   $$\hat{y}_{\text{clause}}(c) = \sigma\left( \mathbf{w}_c^T \mathbf{c}_j^{(T)} + b_c \right)$$

Trained via supervised regression against decision traces recorded from successful Glucose runs on SAT Competition benchmarks.

---

## 3. Solver Integration & Multi-Armed Bandit (MAB) Controller

NeuroComb introduces a multi-armed bandit at solver restart boundaries:
- **Arm 1:** Pure Classical Heuristics (Glucose VSIDS / LRB + LBD clause deletion).
- **Arm 2:** Neural-Boosted Variable Selection (VSIDS activities scaled by $\hat{y}_{\text{var}}$).
- **Arm 3:** Neural-Guided Clause Reduction (pruning clauses based on $\hat{y}_{\text{clause}}$).

The MAB uses an Upper Confidence Bound (UCB1) or Exponential-weight algorithm for Exploration and Exploitation (EXP3), updating arm rewards based on conflict generation rate:
$$R_t = \frac{\Delta \text{conflicts}_t}{\Delta \text{time}_t}$$

---

## 4. Critical Audit: Successes & Methodological Flaws

### 4.1 Successes
- **First Successful Hybrid MAB Controller:** NeuroComb proved that an adaptive bandit can mitigate the risk of bad neural predictions by falling back to classical heuristics when neural guidance fails to produce conflicts.
- **Tackled Clause Reduction:** Recognized that learned clause management is just as critical to CDCL efficiency as variable branching.

### 4.2 Critical Bottlenecks & Limitations
1. **Supervised Imitation of Suboptimal Traces:**  
   NeuroComb trained its GNN on Glucose execution traces. However, classical solvers frequently take suboptimal detours before finding refutations. Training via supervised imitation forced the GNN to learn and reinforce those suboptimal search patterns.
2. **Heuristic Bandit without Distribution Awareness:**  
   NeuroComb's MAB only evaluated local conflict rate. It possessed **zero out-of-distribution (OOD) awareness**: when presented with an unseen formula family, the bandit wasted hundreds of restart cycles exploring ineffective neural arms before finally reverting to classical solving.

---

## 5. Direct Inspiration & Concepts Synthesized into NeuroCDCL

NeuroComb provided the direct conceptual bridge to our **Folder 52 (NeuroCDCL Kissat)** architecture:

1. **In-Register Bandit Restarts (Folder 22 / Folder 52):**  
   We adopted NeuroComb's insight that restart boundaries are the optimal point for heuristic arbitration. In Folder 52, we embedded a **non-stationary MAB restart controller** directly in C99 inside Kissat's `kissat_restart()`, achieving adaptive heuristic switching at **zero microsecond overhead**.
2. **Elimination of Supervised Proof Invariants in Favor of Causal Flow:**  
   Rather than training a static GNN to guess proof participation offline, Folder 52 tracks **actual empirical implication flows** during the live run via bitset causal arrays, eliminating offline dataset dependencies entirely.
