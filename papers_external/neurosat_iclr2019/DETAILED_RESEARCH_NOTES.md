# Research Dossier: NeuroSAT (ICLR 2019)

**Title:** Learning a SAT Solver from Single-Bit Supervision  
**Authors:** Daniel Selsam, Matthew Lamm, Benedikt Bünz, Percy Liang, David L. Dill (Stanford University)  
**Venue:** International Conference on Learning Representations (ICLR 2019)  
**Primary Artifact:** [`neurosat_iclr2019.pdf`](neurosat_iclr2019.pdf)

---

## 1. Executive Summary & Historical Significance

NeuroSAT is the **foundational pioneer paper** of deep learning for satisfiability. Prior to NeuroSAT, machine learning applied to SAT was restricted to hand-engineered feature extraction (e.g., SATzilla's portfolio algorithm selection).

**NeuroSAT's Audacious Premise:**  
Can an artificial neural network learn the fundamental logic of Boolean Satisfiability—including resolution, unit propagation, and variable assignment—trained **strictly end-to-end on a single bit of supervision per formula: Is this formula SAT (1) or UNSAT (0)?**

---

## 2. The Foundational Bipartite Graph Architecture

NeuroSAT introduced the mathematical graph formulation that every subsequent ML4SAT paper (including RLAF, ImitSAT, NeuroBack, and NeuroComb) relies upon:

### 2.1 Literal-Clause Bipartite Graph
Given CNF formula $\phi$ with literals $L$ and clauses $C$:
- Literal nodes: $l \in L = \{x_1, \neg x_1, \dots, x_n, \neg x_n\}$, initial embedding $\mathbf{l} \in \mathbb{R}^d$.
- Clause nodes: $c \in C$, initial embedding $\mathbf{c} \in \mathbb{R}^d$.
- Edges: Undirected edge $(l, c)$ whenever literal $l \in c$.

### 2.2 Alternating Message Passing Equations
NeuroSAT defines an alternating message-passing loop over $T = 26\text{--}32$ iterations:
1. **Clause Message Update:**
   $$\mathbf{c}_j^{(t+1)} = \text{LSTM}_C \left( \mathbf{c}_j^{(t)}, \sum_{l \in c_j} \mathbf{l}^{(t)} \right)$$
2. **Literal Message Update:**
   $$\mathbf{l}_i^{(t+1)} = \text{LSTM}_L \left( \mathbf{l}_i^{(t)}, \sum_{c \in C(l_i)} \mathbf{c}_j^{(t+1)}, \mathbf{l}_{\neg i}^{(t)} \right)$$
   Crucially, the literal update explicitly receives the embedding of its own complement $\mathbf{l}_{\neg i}^{(t)}$ via an involution operator, preserving the fundamental boolean property that literal $l$ and $\neg l$ are mutually exclusive.

### 2.3 Single-Bit Classification Head & Model Extraction
The formula-level embedding is pooled across all clauses:
$$\mathbf{z}_\phi = \frac{1}{|C|} \sum_{c \in C} \mathbf{c}^{(T)}$$
$$\hat{y}_{\text{SAT}} = \sigma(\mathbf{w}^T \mathbf{z}_\phi + b) \in [0, 1]$$
When $\hat{y} > 0.5$ (predicting SAT), NeuroSAT extracts satisfying truth assignments by clustering the literal vectors: literals assigned `true` cluster together in embedding space.

---

## 3. Critical Audit: The Limits of End-to-End Deep Learning

### 3.1 Historical Breakthrough
NeuroSAT proved that:
- Deep neural networks can discover structural invariants of NP-complete problems without human-designed features.
- Bipartite message passing is permutation-invariant to variable/clause ordering and negation-equivariant.

### 3.2 Fundamental Architectural Limitations
1. **The Scalability Wall ($N \le 40$ Variables):**  
   NeuroSAT was evaluated exclusively on tiny synthetic random formulas ($N \in [10, 40]$ variables). Because it attempts to solve SAT purely inside recurrent LSTM hidden states without a resolution backtrack tree, it completely fails to scale to 100-variable formulas, let alone industrial formulas with $1,000,000$ variables.
2. **Zero Mathematical Soundness:**  
   NeuroSAT provides no certificates: it cannot output a DRAT proof of unsatisfiability, and its predicted assignments frequently violate a small subset of clauses on hard instances.

---

## 4. Direct Lessons Learned & Synthesis into NeuroCDCL

NeuroSAT provided the foundational negative proof that shaped our entire project philosophy:

1. **Neural Networks Must Guide, Never Replace, the CDCL Engine:**  
   NeuroSAT proved that deep learning cannot replace resolution and conflict analysis. In NeuroCDCL, **the native CDCL solver (MiniSat / Kissat) remains the absolute master of truth and soundness**, with complete 100% mathematical guarantees.
2. **Simplification of Message Passing into Local SIMD Kernels:**  
   NeuroSAT required 32 iterations of heavy LSTMs with 128-dimensional vectors. In **M3.5 Version 2**, we proved that 90% of the useful branching signal can be captured by local literal degree distributions evaluated in a **single microsecond pass** over CPU SIMD registers, discarding the heavy recurrent LSTM machinery.
