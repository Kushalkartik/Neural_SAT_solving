# Paper Notes: NeuroCore (Selsam & Bjørner, SAT 2019)

### 1. Research Problem
Can deep neural networks predict variables that belong to unsatisfiable cores, and can these predictions effectively guide modern CDCL solvers (MiniSat, Glucose, Z3)?

### 2. Method
- GNN predicts probability of variable occurrence in unsatisfiable cores.
- Periodically resets or bumps solver VSIDS scores using neural predictions.
- Interventions occur at fixed restart intervals.

### 3. Novelty Demarcation for Project 14
- NeuroCore proves that periodic neural intervention is established prior art.
- Project 14 does NOT claim novelty for periodic guidance. Project 14 novelty lies in *adaptive cost-aware gating* (Trust -> Refresh -> Branch), Mahalanobis OOD detection on graph states, and learned conflict-benefit estimation.
