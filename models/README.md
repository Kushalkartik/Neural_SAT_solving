# Neural Models, Weights, and Architectures

This directory contains the neural network weights, architectures, and distillation artifacts powering **NeuroCDCL**:

- **Detailed Technical Specifications:** See [`WEIGHTS_INVENTORY.md`](WEIGHTS_INVENTORY.md) for full layer-by-layer topologies, parameter counts, memory profiles, and training lineages.
- **Architectures:** See [`architectures/`](architectures/) for the underlying PyTorch model definitions.
- **Weights:**
  - [`weights/m3_5_v2/`](weights/m3_5_v2/): Production 648-byte AVX2 SIMD weights, compact d16/d32 student binaries, and RLAF GRPO teacher checkpoints.
  - [`weights/folder52_kissat/`](weights/folder52_kissat/): Embedded C99 static weights header (`structural_mlp_weights.h`), Micro-Neural Causal Flow policies, and deep causal teacher checkpoints.
  - [`weights/foundational/`](weights/foundational/): Comparative checkpoints including Graph-Q-SAT DQN, NeuroDAG GNN, and NeuroDecomp GNN.
