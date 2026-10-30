# Authoritative Neural Weights Inventory & Model Zoo

This directory hosts the complete collection of trained neural weights, teacher-student distillation checkpoints, embedded C99 headers, and architecture definitions across the **M3.5 Version 2** and **Folder 52 (NeuroCDCL Kissat)** platforms.

---

## 1. Directory Layout

```
models/
├── architectures/                     # PyTorch neural model & distillation definitions
│   ├── rlaf_policy_network.py         # 5-layer Relational GCN policy head
│   ├── distill_m3_compact.py          # Knowledge distillation trainer (GNN -> MLP)
│   ├── graph_q_sat_network.py         # Bipartite MPNN Deep Q-Network
│   ├── neurodag_gnn.py                # Resolution DAG message passing network
│   ├── train_micro_neural_policy.py   # Causal flow student trainer (MNCF 4->8->4->1)
│   ├── train_1d_micro_neural.py       # 1D Micro-Neural Flow Policy trainer (AdamW)
│   ├── train_and_distill_precision.py # Precision restart student distillation harness
│   └── train_clause_teacher_and_distill.py # Folder 49 Clause Quality teacher/student trainer
│
└── weights/                           # Binary, C header, and PyTorch weight artifacts
    ├── m3_5_v2/                       # M3.5 Version 2 (MiniSat Neural-Guided CDCL)
    │   ├── structural_mlp_weights.bin # [PRIMARY] 648-byte AVX2 SIMD weights (4.4 us)
    │   ├── compact_d16_weights.bin    # 16-dim compact distilled weights (27.7 KB)
    │   ├── compact_d32_weights.bin    # 32-dim compact distilled weights (100.9 KB)
    │   ├── compact_d16_T1.pt          # PyTorch checkpoint for d16 student
    │   ├── compact_d32_T1.pt          # PyTorch checkpoint for d32 student
    │   ├── distilled_student_model.pt # PyTorch student distillation checkpoint
    │   ├── rlaf_weights.bin           # Binary serialization of RLAF GNN policy
    │   └── rlaf_teacher_seed42.pt     # Full PyTorch RLAF GRPO teacher checkpoint
    │
    ├── folder52_kissat/               # Folder 52 (NeuroCDCL Kissat Flagship)
    │   ├── structural_mlp_weights.h   # Embedded C99 topological weights header
    │   ├── micro_neural_net.h         # [MNCF] Bounded Micro-Neural Causal Flow header
    │   ├── micro_neural_policy.pt     # PyTorch Micro-Neural Causal Flow (MNCF)
    │   ├── micro_neural_policy_bounded.pt # Bounded polarity MNCF model
    │   ├── nano_clause_net.h          # [FOLDER 49] Clause usefulness Nano-MLP (65 params)
    │   ├── nano_mlp_distilled.h       # [MAB RESTART] Precision restart Nano-MLP (832 params)
    │   ├── nano_mode_net.h            # [FOLDER 49] Focused/Stable mode switching net
    │   ├── nano_restart_net.h         # [FOLDER 49] Dynamic restart interval net
    │   ├── best_precision_student.pt  # Precision causal student model (6.8 KB)
    │   ├── best_empirical_student.pt  # Empirical telemetry student model (3.4 KB)
    │   ├── best_empirical_teacher.pt  # Deep offline telemetry teacher (317 KB)
    │   └── best_causal_teacher.pt     # Full causal implication teacher (1.68 MB)
    │
    └── foundational/                  # Baselines & comparative neural milestones
        ├── graph_q_sat_dqn.pt         # M1 Graph-Q-SAT Deep Q-Network (267 KB)
        ├── neurodag_gnn.pt            # NeuroDAG resolution graph network (1.98 MB)
        ├── neuro_decomp_gnn.pt        # Formula modular decomposition GNN (32.5 KB)
        └── neural_er_scorer.pt        # Extended resolution clause scorer (12.6 KB)
```

---

## 2. System 1: M3.5 Version 2 Weight Specifications

### 2.1 Primary Production Model: `structural_mlp_weights.bin` (648 Bytes)
- **Target Solver:** `solvers/m3_5_v2/m3_5_v2.exe`
- **Execution Mechanism:** Zero-copy native binary array loaded into AVX2 256-bit SIMD registers (`structural_mlp_engine.cc`).
- **Latency:** **$4.4\ \mu\text{s}$** per 1,000 variables.
- **Architecture:** Feed-forward network with SIMD LayerNorm:
  - Input: 6 graph-topological features (positive occurrences, negative occurrences, normalized degree, literal polarity balance, inverse clause length centrality, short clause count).
  - Layers: `Linear(6, 16) -> ReLU -> AVX2_LayerNorm(16) -> Linear(16, 1)`.
  - Output: Exponential activity priority $\exp(\mu)$ and sign prior $\text{sign}(\text{bal})$.
- **Exact Byte Layout (648 Bytes):**
  - Layer 1 Weights: $16 \times 6 \times 4\text{ bytes} = 384\text{ bytes}$
  - Layer 1 Biases: $16 \times 4\text{ bytes} = 64\text{ bytes}$
  - LayerNorm Scale & Bias: $16 \times 4 + 16 \times 4 = 128\text{ bytes}$
  - Layer 2 Weights & Bias: $16 \times 4 + 1 \times 4 = 68\text{ bytes}$
  - Magic (`SMLP`): $4\text{ bytes}$. Total = $648\text{ bytes}$.

### 2.2 Distillation Lineage & Teacher Checkpoints
1. **Teacher Network (`rlaf_teacher_seed42.pt` / `rlaf_weights.bin`):**  
   3-layer Relational GCN trained on 60,000 instances (Random 3-SAT, 3-Coloring, Cryptographic) using Group Relative Policy Optimization (GRPO).
2. **Compact Intermediate Students (`compact_d16_weights.bin`, `compact_d32_weights.bin`):**  
   Distilled using multi-task loss:
   $$\mathcal{L} = \text{MSE}(\mu_{\text{student}}, \mu_{\text{teacher}}) + 2.0 \cdot \text{BCEWithLogits}(\rho_{\text{student}}, \sigma(\rho_{\text{teacher}}))$$

---

## 3. System 2: Folder 52 (NeuroCDCL Kissat) Multi-Point Neural Weights

Folder 52 combines multiple specialized in-engine neural mechanisms compiled directly into C99 headers for sub-nanosecond to nanosecond execution:

### 3.1 Point 1: Micro-Neural Causal Flow (MNCF)
* **Artifacts:**
  * Embedded Header: [`models/weights/folder52_kissat/micro_neural_net.h`](file:///c:/Users/Kushal%20Agarwal/Desktop/neuro-cdcl/models/weights/folder52_kissat/micro_neural_net.h)
  * Checkpoints: `micro_neural_policy.pt` (3.4 KB) & `micro_neural_policy_bounded.pt` (3.6 KB)
  * Training Script: `models/architectures/train_1d_micro_neural.py`
* **Target Engine:** `solvers/neuro_cdcl_f52/src/deduce.c` and `bump.c`.
* **Latency:** **$< 0.8\text{ nanoseconds}$** per variable bump.
* **Mechanism:**
  - Backpropagates causal refutation credit along the 1-UIP implication DAG during conflict analysis.
  - Variable VSIDS bumps are scaled by the micro-neural transfer policy:
    $$\phi = 1.0 + 0.35 \cdot \frac{\log(1 + w_{\text{cif}})}{1 + 0.35 \cdot \log(1 + w_{\text{cif}})}, \quad \text{score} \leftarrow \text{score} + \text{inc} \cdot \phi$$

### 3.2 Point 2: Folder 49 Clause Usefulness Scorer
* **Artifacts:**
  * Embedded Header: [`models/weights/folder52_kissat/nano_clause_net.h`](file:///c:/Users/Kushal%20Agarwal/Desktop/neuro-cdcl/models/weights/folder52_kissat/nano_clause_net.h)
  * Multi-Point Complements: `nano_mode_net.h`, `nano_restart_net.h`
  * Training Script: `models/architectures/train_clause_teacher_and_distill.py`
* **Target Engine:** `solvers/neuro_cdcl_f52/src/neural_clause_scorer.c` and `reduce.c`.
* **Latency:** **$< 5\text{ nanoseconds}$** per clause evaluated.
* **Parameters:** 65 floating-point parameters ($6 \to 8 \to 1$ Nano-MLP).
* **Mechanism & Integration:**
  - Evaluates 6 clause features (normalized LBD/glue, clause length, usage count, activity proxy, tier context).
  - Outputs a 32-bit usefulness rank score `kissat_score_clause_quality(solver, c, max_act)`.
  - In `reduce.c`:
    ```c
    const uint32_t neural_usefulness = kissat_score_clause_quality (solver, c, max_act);
    red.rank = ((uint64_t) neural_usefulness << 32) | (uint64_t) negative_size;
    RADIX_STACK (reducible, uint64_t, *reds, USEFULNESS);
    ```
    Replaces static heuristics with a 64-bit radix-sorted neural clause retention policy.

### 3.3 Point 3: Precision Contextual MAB Restart Controller
* **Artifacts:**
  * Embedded Header: [`models/weights/folder52_kissat/nano_mlp_distilled.h`](file:///c:/Users/Kushal%20Agarwal/Desktop/neuro-cdcl/models/weights/folder52_kissat/nano_mlp_distilled.h)
  * Checkpoints: `best_causal_teacher.pt` (1.68 MB), `best_precision_student.pt` (6.8 KB)
  * Training Script: `models/architectures/train_and_distill_precision.py`
* **Target Engine:** `solvers/neuro_cdcl_f52/src/mab_restart.c` and `restart.c`.
* **Latency:** **$\sim 30\text{ nanoseconds}$** (inlined C SIMD).
* **Parameters:** 832 floating-point parameters ($8 \to 32 \to 16 \to 4$).
* **Mechanism:**
  - Evaluates 8 causal zero-lookahead search features to predict contextual priors over 4 restart regimes (Agile, Moderate, Deep, Frozen), preventing thrashing on complex industrial instances.

---

## 4. Foundational & Comparative Checkpoints

1. **`graph_q_sat_dqn.pt` (267.6 KB):**  
   Reproduction checkpoint of Kurin et al.'s Deep Q-Network from NeurIPS 2020.
2. **`neurodag_gnn.pt` (1.98 MB):**  
   NeuroDAG resolution proof graph message passing network.
3. **`neuro_decomp_gnn.pt` (32.5 KB):**  
   Hypergraph modular decomposition neural partitioner.
4. **`neural_er_scorer.pt` (12.6 KB):**  
   Extended Resolution auxiliary variable pair scorer.
