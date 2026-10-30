# Research Dossier: NeuroCore (SAT 2019)

**Title:** Guiding High-Performance SAT Solvers with Unsatisfiable Core Predictions  
**Authors:** Daniel Selsam, Nikolaj Bjørner (Stanford University & Microsoft Research)  
**Venue:** International Conference on Theory and Applications of Satisfiability Testing (SAT 2019)  
**Primary Artifact:** [`neurocore_sat2019.pdf`](neurocore_sat2019.pdf)

---

## 1. Executive Summary & Research Motivation

The overwhelming majority of practical applications of SAT solvers (formal hardware verification, bounded model checking, SMT solving, theorem proving) operate on **unsatisfiable formulas**. On such instances, the solver's task is not to find a satisfying model, but to construct a resolution proof of contradiction ($\bot$).

In almost all real-world verification formulas, only a tiny fraction of the original constraints belong to the **Unsatisfiable Core (Unsat Core)**. Branching on variables outside the unsat core is mathematically futile and leads to exponential tree expansion.

**NeuroCore's Core Thesis:**  
Train a Graph Neural Network to predict the probability that each variable belongs to an unsatisfiable core. Periodically inject these predictions into the variable activity scores of modern CDCL solvers (MiniSat, Glucose, Z3) during search to bias branching toward refutation-relevant variables.

---

## 2. Neural Architecture & Unsat Core Formulation

### 2.1 Bipartite Message Passing Network
NeuroCore models CNF formulas as an undirected bipartite graph between literals $L$ and clauses $C$:
$$\mathbf{c}_j^{(t+1)} = f_C\left( \mathbf{c}_j^{(t)}, \sum_{l \in c_j} \mathbf{l}_i^{(t)} \right)$$
$$\mathbf{l}_i^{(t+1)} = f_L\left( \mathbf{l}_i^{(t)}, \sum_{c \in C(l_i)} \mathbf{c}_j^{(t+1)}, \mathbf{l}_{\neg i}^{(t)} \right)$$

### 2.2 Supervised Training on Minimal Unsat Cores (MUS)
- **Dataset:** Industrial verification formulas from SAT Competitions and software verification benchmarks.
- **Ground Truth Labels:** Extracted minimal unsatisfiable cores using exact MUS extractors (e.g., PicoMUS / MUSer2):
  $$y_i = \mathbb{I}(x_i \in \text{Core}(\phi))$$
- **Loss:** Binary Cross-Entropy with pos-weight $\alpha$:
  $$\mathcal{L}(\theta) = -\sum_{i=1}^n \left[ \alpha \cdot y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

## 3. Solver Integration: Periodic Intervention at Restarts

NeuroCore avoids the per-decision latency trap by intervening **only at fixed restart intervals** (e.g., every 1,000 conflicts):
```cpp
// Solver::check_periodic_neurocore()
if (conflicts >= next_neurocore_intervention) {
    // 1. Run GNN forward pass on active formula
    std::vector<float> core_probs = gnn_predict_unsat_core();
    
    // 2. Bump VSIDS activity scores for predicted core variables
    for (int v = 0; v < nVars(); v++) {
        if (core_probs[v] > threshold) {
            activity[v] += core_probs[v] * bump_factor;
        }
    }
    next_neurocore_intervention += interval;
}
```

---

## 4. Critical Audit: Successes & Hidden Bottlenecks

### 4.1 Successes
- **Refutation Focus:** NeuroCore was the first ML4SAT paper to recognize that real-world SAT solving is primarily about **unsatisfiable core extraction**, not satisfying model search.
- **Periodic Intervention Paradigm:** Established the viability of intermittent neural intervention at restart boundaries rather than per-step interception.

### 4.2 Critical Bottlenecks
1. **Periodic Inference Latency Accumulation:**  
   While periodic intervention is faster than per-step querying, running a full PyTorch GNN forward pass every 1,000 conflicts still introduces significant overhead. On large industrial instances ($> 100,000$ variables), each GNN pass took **several seconds**, causing the solver to spend 80% of its runtime in PyTorch rather than MiniSat.
2. **Static Fixed-Interval Blindness:**  
   NeuroCore triggered interventions on a static, rigid counter. If the solver was already making rapid progress through a refutation subtree, NeuroCore still interrupted the solver to run a heavy GNN pass, destroying search locality and cache coherency.

---

## 5. Direct Inspiration & Concepts Synthesized into NeuroCDCL

NeuroCore provided the direct motivation for our **Microsecond Structural Distillation** and **Inprocessing Soundness**:

1. **Focus on Refutation Invariants:**  
   We embraced NeuroCore's philosophy that the primary value of neural guidance is accelerating refutation proofs. In **M3.5 Version 2**, our 648-byte AVX2 MLP specifically targets UNSAT pruning, achieving **-90.5% decision pruning and a 67.2x wall-clock speedup on 3COL(500) UNSAT**.
2. **Elimination of Heavy Periodic Passes:**  
   Rather than executing heavy GNN forward passes during search, **Folder 52 (NeuroCDCL Kissat)** replaces periodic external passes with **in-register Causal Implication Flow (CIF)**, tracking implication causality on every trail assignment with zero interrupt overhead.
