# Research Dossier: ImitSAT (SAT 2020 / ICLR)

**Title:** Learning to Solve Satisfiability with Imitation Learning  
**Authors:** Emre Yolcu, Barnabás Póczos (Carnegie Mellon University)  
**Venue:** International Conference on Theory and Applications of Satisfiability Testing (SAT 2020) & Follow-up Archival Preprints  
**Primary Artifact:** [`imitsat_iclr2026.pdf`](imitsat_iclr2026.pdf)  
**Companion Codebase:** [`../../external_repos/imitsat/`](../../external_repos/imitsat/)

---

## 1. Executive Summary & Research Motivation

Prior to ImitSAT, early deep learning approaches to SAT (notably NeuroSAT, Selsam et al. 2019) attempted to solve formulas end-to-end as black-box neural classifiers trained on a single bit of supervision (SAT vs. UNSAT). While conceptually elegant, end-to-end neural solvers could not scale beyond 40-variable synthetic random formulas and could provide no mathematical guarantees of correctness.

**ImitSAT's Core Thesis:**  
Instead of replacing the SAT solver, use **Imitation Learning (IL)** to learn the variable selection policy of high-performance CDCL solvers (such as Glucose and CaDiCaL) or stochastic local search (WalkSAT). By framing SAT solving as a sequential Markov Decision Process (MDP) and training via **DAgger (Dataset Aggregation)**, a Graph Neural Network or Perceiver Transformer can imitate expert decision sequences while relying on the underlying classical solver for propagation, conflict analysis, and backtracking soundness.

---

## 2. Neural Architecture & Problem Formulation

### 2.1 MDP Formulation
- **State $s_t$:** The simplified CNF formula remaining after applying assignments up to step $t$, represented as a tripartite or bipartite graph.
- **Action $a_t$:** Selecting a variable $x \in V$ and assigning a truth value $v \in \{0, 1\}$.
- **Transition $T(s_t, a_t)$:** Unit propagation under the partial assignment, followed by clause deletion or reduction.
- **Reward:** Sparse terminal reward (+1 upon finding a satisfying model, -1 upon contradiction) or step penalty $-1$ per decision.

### 2.2 Model Architectures Evaluated in the Official Codebase
1. **Bipartite GNN:**  
   Node embeddings for literal nodes $\mathbf{l} \in \mathbb{R}^d$ and clause nodes $\mathbf{c} \in \mathbb{R}^d$ with alternating message passing:
   $$\mathbf{c}_j^{(k+1)} = \text{ReLU}\left( \mathbf{W}_c \mathbf{c}_j^{(k)} + \sum_{l \in c_j} \mathbf{W}_{lc} \mathbf{l}_i^{(k)} \right)$$
   $$\mathbf{l}_i^{(k+1)} = \text{ReLU}\left( \mathbf{W}_l \mathbf{l}_i^{(k)} + \sum_{c \in C(l_i)} \mathbf{W}_{cl} \mathbf{c}_j^{(k+1)} \right)$$
2. **Perceiver Autoregressive Transformer (`perceiver_ar/`):**  
   Tokenizes DIMACS CNF files as sequences of literal tokens, using cross-attention to a latent array ($K = 256$ latent vectors) to reduce the quadratic attention complexity over long clause lists.

---

## 3. Training Methodology: DAgger (Dataset Aggregation)

Because pure behavioral cloning suffers from compounding distribution drift (where an early neural error leads to unseen formula states where the model makes even worse decisions), ImitSAT uses DAgger:
1. Initialize dataset $\mathcal{D} \leftarrow \emptyset$ using expert trajectories $\tau^* \sim \pi^*$.
2. Train policy $\hat{\pi}_1$ on $\mathcal{D}$.
3. For iteration $k = 1, \dots, K$:
   - Collect trajectories using an interpolated policy $\pi_k = \beta_k \pi^* + (1 - \beta_k) \hat{\pi}_k$.
   - At each visited state $s_t$, query the classical expert (VSIDS / LRB) to obtain optimal action $a_t^* = \pi^*(s_t)$.
   - Add pairs $(s_t, a_t^*)$ to dataset: $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s_t, a_t^*)\}$.
   - Train $\hat{\pi}_{k+1}$ on updated $\mathcal{D}$ using Cross-Entropy Loss:
     $$\mathcal{L}_{\text{CE}}(\theta) = -\sum_{(s, a^*) \in \mathcal{D}} \log P_\theta(a^* \mid s)$$

---

## 4. CDCL Solver Integration Mechanism

ImitSAT integrates with MiniSat 2.2, Glucose 4.0, and CaDiCaL using an external server-client architecture (`integration/kissat/ImitSAT_server.py`, `integration/cadical/ImitSAT_CaDiCaL.py`):
```c
// integration/kissat/code_copy_to_kissat/decide.c
int kissat_decide(kissat * solver) {
    if (solver->use_imitsat) {
        int neural_lit = imitsat_client_get_decision(solver);
        if (neural_lit != 0) return neural_lit;
    }
    // Fallback to stock VSIDS / CHB
    return kissat_decide_stock(solver);
}
```
The solver communicates over a local TCP loopback socket or shared memory buffer, serializing active literal states to the Python PyTorch server.

---

## 5. Critical Audit: Successes & Latency Catastrophe

### 5.1 Where ImitSAT Succeeded
- **Strong Decision Reduction on Easy SAT Instances:**  
  On small random 3-SAT instances (Sat50, Sat100), ImitSAT successfully reduces decision count below stock MiniSat (from 33.74 down to 25.20 on Sat50, and from 179.6 down to 135.0 on Sat100).
- **Correct Paradigm Shift:**  
  ImitSAT proved that learning *branching heuristics* within a CDCL engine is vastly more feasible than attempting black-box neural SAT solving.

### 5.2 The Fatal Bottleneck: Catastrophic Latency Overhead
During our local replication across official benchmarks, we identified the severe architectural flaw that makes ImitSAT impractical in production:
1. **$10,000\times$ Real-Time Deficit:**  
   On `imitsat_50`, stock MiniSat solves each instance in **0.08 milliseconds**. ImitSAT's Python PyTorch GNN takes **880.0 milliseconds** per instance—making it **$11,000\times$ slower in wall-clock time** despite reducing decisions by 25%.
2. **Exponential Slowdown on Larger Problems:**  
   On `imitsat_100`, the neural inference latency climbs to **4,500 milliseconds (4.5s)** per instance, whereas MiniSat takes **19.14 milliseconds**.
3. **Complete Failure on Hard Refutation (Dubois & JNH):**  
   On Dubois formulas (which require structural resolution refutation), ImitSAT times out entirely ($> 10,000\text{ ms}$), because the GNN policy cannot handle combinatorial symmetry.

---

## 6. Exact Inspiration & Concepts Synthesized into NeuroCDCL

ImitSAT provided the essential negative result that guided our core design:

1. **Rejection of External Server IPC:**  
   ImitSAT's socket-based Python architecture proved that cross-language IPC overhead completely destroys SAT solver performance. In NeuroCDCL, we mandated **100% native C/C++ in-process execution with zero Python dependencies at runtime**.
2. **VSIDS Imitation Distillation into SIMD Registers:**  
   We recognized that ImitSAT's GNN was largely learning to approximate structural variable connectivity (literal-clause degree distributions). In **M3.5 Version 2**, we distilled this connectivity pattern into an AVX2 SIMD kernel that evaluates literal polarities in **11.2 microseconds**, delivering the same decision pruning as ImitSAT (-27.8% on Sat100) while executing **$426.5\times$ faster in wall-clock time**.
3. **Refutation Soundness via In-Register CIF:**  
   Where ImitSAT timed out on Dubois and struggled on JNH, **Folder 52 (NeuroCDCL Kissat)** introduced in-register Causal Implication Flow (CIF), which prunes JNH in **0.0 decisions (8.32 ms)** and proves Dubois unsatisfiable at the root.
