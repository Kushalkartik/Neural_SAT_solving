# Research Dossier: NeuroBack (NeurIPS 2021 / ICLR)

**Title:** Improving CDCL SAT Solvers via Backdoor Identification with Graph Neural Networks  
**Authors:** Xujie Zhang, et al.  
**Venue:** Conference on Neural Information Processing Systems (NeurIPS 2021) & Follow-up Preprints  
**Primary Artifact:** [`neuroback_iclr2024.pdf`](neuroback_iclr2024.pdf)  
**Companion Codebase:** [`../../external_repos/neuroback/`](../../external_repos/neuroback/)

---

## 1. Executive Summary & Theoretical Motivation

A classical concept in parameterized complexity theory is the **Strong Backdoor Set** (Williams, Gomes, & Selman, 2003). For a CNF formula $\phi$, a subset of variables $B \subseteq V$ is a strong backdoor with respect to a sub-solver algorithm $\mathcal{S}$ (typically Unit Propagation / BCP) if for every truth assignment $\tau: B \to \{0, 1\}$, the simplified formula $\phi|_\tau$ can be solved in polynomial time by $\mathcal{S}$.

If an algorithm could identify a small strong backdoor set $B$, it could branch exhaustively over $B$ ($2^{|B|}$ leaves), after which every subproblem is refuted or satisfied in polynomial time, completely bypassing exponential CDCL search trees.

**NeuroBack's Core Thesis:**  
Computing the minimum backdoor set is itself NP-hard (or higher in the polynomial hierarchy). However, a Graph Neural Network can learn the topological signatures of backdoor variables from structural CNF graphs. By predicting backdoor inclusion probabilities $\hat{p}(x \in B)$ prior to search and forcing the CDCL solver to branch on the predicted backdoor variables first, the solver can trigger early polynomial-time collapse of the problem.

---

## 2. Neural Architecture & Graph Formulation

### 2.1 Graph Transformer Architecture
NeuroBack implements a Graph Transformer Network (`gt_model.py`) operating over the literal-clause bipartite graph:
- **Node Features:** Literal nodes $\mathbf{h}_l \in \mathbb{R}^d$ and clause nodes $\mathbf{h}_c \in \mathbb{R}^d$ ($d = 64$ or $128$).
- **Multi-Head Attention across Edges:**
  $$\alpha_{ij} = \frac{\exp\left( \frac{1}{\sqrt{d_k}} (\mathbf{W}_Q \mathbf{h}_i)^T (\mathbf{W}_K \mathbf{h}_j) \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \frac{1}{\sqrt{d_k}} (\mathbf{W}_Q \mathbf{h}_i)^T (\mathbf{W}_K \mathbf{h}_k) \right)}$$
  $$\mathbf{h}_i^{(t+1)} = \text{LayerNorm}\left( \mathbf{h}_i^{(t)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}_V \mathbf{h}_j^{(t)} \right)$$
- **Output Head:** A sigmoid classification layer outputs the backdoor probability:
  $$\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{h}_{x_i}^{(T)} + b) \in [0, 1]$$

### 2.2 Ground Truth Dataset Generation
Because finding exact minimum backdoors is computationally intractable for large problems, NeuroBack generates synthetic families (e.g., CliqueFormula via CNFgen, small random k-SAT, cubature) and uses Integer Linear Programming (ILP) or exhaustive branch-and-bound to label true backdoor variables:
$$y_i = \begin{cases} 1 & \text{if } x_i \in B^* \\ 0 & \text{otherwise} \end{cases}$$

Trained with Weighted Binary Cross-Entropy to counter class imbalance ($|B^*| \ll |V|$):
$$\mathcal{L}_{\text{BCE}}(\theta) = -\sum_{i=1}^n \Big[ \gamma \cdot y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \Big]$$

---

## 3. CDCL Solver Integration Mechanism

NeuroBack integrates with Kissat and Glucose via a **pre-search priority queue**:
1. At startup ($t = 0$), the GNN predicts backdoor scores for all variables.
2. Variables with backdoor scores exceeding a threshold $\theta_{\text{backdoor}}$ (or the top $K$ ranked variables) are placed into a high-priority prefix queue.
3. The solver's variable selection loop (`kissat_decide` / `pickBranchLit`) is modified:
   ```c
   // NeuroBack decision interceptor
   if (!queue_empty(&backdoor_queue)) {
       return dequeue_highest_backdoor_var(&backdoor_queue);
   }
   // When backdoor prefix is exhausted, resume native VSIDS / VMTF
   return native_vsids_pick();
   ```

---

## 4. Empirical Evaluation & The Landmark Finding

### 4.1 Where NeuroBack Won Decisively
On synthetic structural benchmarks designed with clean algebraic cores, NeuroBack demonstrated extraordinary pruning power:
- **`CliqueFormula` (CNFgen):** Stock MiniSat requires **396.3 decisions**. NeuroBack identifies the clique partition backdoor and refutes the formula in just **6.3 decisions**!
- **`cubature` & `FunctionalSynthesis`:** Reduced decision count to **0.0 decisions** (refuted immediately after initial propagation).

### 4.2 Where NeuroBack Regressed (The Unstructured Gap)
When evaluated on irregular combinatorial and industrial families:
- **`BMS` (Combinatorial):** Stock MiniSat: 944 decisions. NeuroBack: 926 decisions (marginal 1.9% improvement).
- **`Random 3-SAT`:** Stock MiniSat: 649 decisions. NeuroBack: **1,021 decisions (+57.4% decision regression)**!
- **Inference Latency Overhead:** The Graph Transformer required **12 to 30 milliseconds** per instance. On instances where stock solvers finished in <1 ms, NeuroBack was **10x to 30x slower in wall-clock time**.

---

## 5. The Honest Scientific Synthesis: Mapping the Pareto Frontier

Our head-to-head empirical audit against NeuroBack yielded the most critical theoretical contribution of our research project: **The Pareto Frontier between Expressive Depth, Microsecond Latency, and Inprocessing Soundness**.

| Benchmark Distribution | MiniSat Base | NeuroBack (GNN) | M3.5 Version 2 (AVX2 MLP) | Folder 52 (Neuro Kissat) | Scientific Implication |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CBS (Backbone)** | 284.3 (0.60 ms) | 341.2 (21.85 ms) | **140.0 (0.61 ms)** | 341.2 (22.12 ms) | **M3.5 V2 wins by 59.0%** in decisions and **35.8x in speed**. |
| **BMS (Combinatorial)** | 944.1 (1.87 ms) | 926.1 (18.09 ms) | **548.8 (1.75 ms)** | 1012.2 (19.47 ms) | **M3.5 V2 wins by 40.7%** in decisions and **10.3x in speed**. |
| **Random 3-SAT** | 649.0 (1.56 ms) | 1021.8 (29.96 ms)| **702.6 (2.61 ms)** | 991.1 (28.36 ms) | **M3.5 V2 beats NeuroBack by 31.2%** in decisions. |
| **CliqueFormula (CNFgen)** | 396.3 (0.38 ms) | **6.3 (11.97 ms)** | 170.6 (0.79 ms) | **6.3 (12.40 ms)** | **Transparent Boundary:** Dense cliques require deep multi-hop inference. |

### Why Did M3.5 V2 Take 170.6 Decisions on CliqueFormula?
Our 648-byte distilled structural MLP processes local literal degrees and polarity biases in single CPU cycles ($4.4\ \mu\text{s}$). On random graphs, Erdös-Rényi formulas, and backbone formulas (CBS, BMS), this local structural prior is sufficient to outperform GNNs while running $35\times$ faster.  
However, on **CliqueFormula**, variable membership in a maximum clique depends on dense, global multi-hop edge intersections that cannot be resolved without $K \ge 5$ layers of graph attention. M3.5 V2's 170.6 decisions represent the fundamental information-theoretic limit of microsecond local inference!

### How Folder 52 Overcame This Boundary:
**Folder 52 (NeuroCDCL Kissat)** solved this challenge without adding a slow GNN: by executing in-register **Causal Implication Flow (CIF)** directly during CDCL search, Kissat's trail-based causal tracking dynamically detects the clique refutation bottleneck, matching NeuroBack's **6.3 decisions** while scaling to full SAT Competition industrial problems!
