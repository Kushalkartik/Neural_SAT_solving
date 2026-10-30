# Paper Notes: Graph-Q-SAT (Kurin et al., NeurIPS 2020)

### 1. Research Problem
Can graph neural networks paired with reinforcement learning learn a branching heuristic for CDCL SAT solving that generalizes to problems larger than those seen during training?

### 2. Method & Architecture
- **Representation**: Bipartite clause-variable factor graph (literal nodes connected to clause nodes).
- **Network**: Message-passing neural network (MPNN) propagating embeddings between clauses and literals.
- **RL Framework**: Deep Q-Networks (DQN) with discount factor $\gamma = 0.99$, reward -1 per step, $\epsilon$-greedy exploration.
- **Solver Integration**: Intercepts branching decisions in MiniSat 2.2.0; queries GNN on every decision.

### 3. Key Findings & Claims
- Reduces number of decision steps by 2x to 3x on synthetic random 3-SAT.
- Generalizes to formulas up to 4x larger than training size.

### 4. Overlap and Boundaries with Project 14
- **Established by Paper**: GNN MDP formulation and step reduction on random 3-SAT.
- **Observed in Code**: GNN inference runs synchronously on CPU/GPU at every single step; incurs massive wall-clock overhead.
- **Project Interpretation**: Proves that decision step reduction alone is insufficient; motivates Project 14's adaptive Trust-Refresh-Branch controller.
