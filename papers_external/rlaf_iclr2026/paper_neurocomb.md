# Paper Notes: NeuroComb (Wang et al., ICLR 2024)

### Bibliographic Identity
- **Title**: NeuroComb: Improving SAT Solving with Graph Neural Networks
- **Authors**: Wenxi Wang, Yang Hu, Mohit Tiwari, Sarfraz Khurshid, Kenneth L. McMillan, Risto Miikkulainen
- **Venue**: International Conference on Learning Representations (ICLR 2024)
- **arXiv**: 2110.14053 (Initial arXiv submission 2021; accepted at ICLR 2024)
- **Local PDF**: `03_EXTERNAL_REFERENCES/PAPERS/NEUROCOMB/neurocomb_neurips2021.pdf`

### 1. Research Problem
How to integrate GNN predictions into CDCL solvers without incurring prohibitive per-step inference overhead that negates decision step savings.

### 2. Method
- One-shot GNN inference at the root to predict variable importance and clause redundancy.
- Embeds static neural predictions into dynamic CDCL heuristics (VSIDS / LRB).
- Multi-armed bandit adaptively selects between neural-guided branching and pure VSIDS.

### 3. Novelty Demarcation for Project 14
- **Established by NeuroComb**: One-shot root inference combined with dynamic CDCL heuristics and bandit switching between static and dynamic modes.
- **Project 14 Distinction**: NeuroComb does NOT perform principled out-of-distribution (OOD) detection via representation covariance, nor does it evaluate predictive uncertainty (action entropy & margin), nor does it learn an explicit refresh-benefit model balancing measured GNN latency against downstream conflict reduction within bounded continuation horizons ($H \in [200, 500]$).
