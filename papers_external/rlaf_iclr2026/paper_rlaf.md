# Paper Notes: RLAF (ICLR 2026)

### Bibliographic Identity
- **Title**: Learning from Algorithm Feedback: One-Shot SAT Solver Guidance with GNNs
- **Authors**: Anonymous / Under Review (ICLR 2026)
- **Venue**: International Conference on Learning Representations (ICLR 2026)
- **Local PDF**: `03_EXTERNAL_REFERENCES/PAPERS/RLAF/rlaf_iclr2026.pdf` (SHA256: `129b6c38...`)

---

### 1. Research Problem & Core Insight
- **The Dilemma**: Traditional learned branching heuristics (e.g. Graph-Q-SAT) invoke a GNN at every single branching step, incurring 80×+ wall-clock overhead that negates any algorithmic decision savings.
- **The RLAF Insight**: Run GNN inference strictly **once at $t=0$** to output variable weights $w(x) \in \mathbb{R}_{>0}$ and polarities $p(x) \in \{0, 1\}$.
- **The Core Distinction vs Supervised Methods (NeuroComb)**: Instead of training via supervised regression against offline solver activities (which may be suboptimal or stale), RLAF trains the GNN using **Reinforcement Learning from Algorithm Feedback (RLAF)**, directly optimizing the downstream solver's runtime / conflict count via **Group Relative Policy Optimization (GRPO)**.

---

### 2. Mathematical Formulation

#### 2.1 Guided Branching Heuristic
For a formula $\phi$, the solver selects the next literal using external guidance weights $W = (w, p)$:
$$\hat{x} \leftarrow \arg\max_{x} w(x) \cdot \text{SCORE}(x)$$
and branches on $\hat{x}$ if $p(\hat{x}) = 1$ else $\neg \hat{x}$.

In CDCL solvers (MiniSat / Glucose):
- Initial VSIDS activities are scaled/initialized by $w(x)$.
- Variable polarities (phase saving) are initialized to $p(x)$.
- Subsequent CDCL conflict bumps dynamically adjust the score during search.

#### 2.2 Policy Parameterization $\pi_\theta(W \mid \phi)$
The GNN outputs two latent values per variable $x$: $y(x) = (\mu(x), \rho(x)) \in \mathbb{R}^2$:
- Variable weight: $w(x) \sim \text{LogNormal}(\mu(x), \sigma^2)$
- Variable polarity: $p(x) \sim \text{Bernoulli}(\sigma(\rho(x)))$
- Zero-Initialization: The final layer is initialized to zeros, guaranteeing that the untrained policy $\pi_0$ is exact unweighted classical solving ($w(x) = 1.0, p(x) = 0.5$).
- Deterministic Evaluation: At test time, use the mode: $\hat{w}(x) = \exp(\mu(x))$ and $\hat{p}(x) = \mathbb{I}(\rho(x) > 0)$.

#### 2.3 Training Algorithm: GRPO (Group Relative Policy Optimization)
RLAF frames one-shot guidance as a single-step contextual bandit / MDP:
1. For each training instance $\phi_i$, sample a group of $M$ actions: $W_{i, 1}, \dots, W_{i, M} \sim \pi_{\theta_{\text{old}}}(\phi_i)$.
2. Run the SAT solver to completion for each $W_{i, j}$, obtaining costs $C_{i, j} = \text{decisions}(W_{i, j})$.
3. Set rewards $R_{i, j} = -C_{i, j}$.
4. Compute Group-Relative Advantage (no critic network needed!):
   $$\hat{A}_{i, j} = \frac{R_{i, j} - \text{mean}(R_i)}{\text{std}(R_i) + \epsilon}$$
5. Optimize the PPO-clipped objective:
   $$\mathcal{L}_{\text{PPO}}(\theta \mid \phi_i) = \frac{1}{M} \sum_{j=1}^M \min\left(r_{i, j}(\theta) \hat{A}_{i, j}, \; \text{clip}(r_{i, j}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i, j}\right)$$
   where $r_{i, j}(\theta) = \frac{\pi_\theta(W_{i, j} \mid \phi_i)}{\pi_{\theta_{\text{old}}}(W_{i, j} \mid \phi_i)}$.
6. Penalize policy drift via KL divergence:
   $$\mathcal{L}(\theta \mid \phi_i) = \mathcal{L}_{\text{PPO}}(\theta \mid \phi_i) - \beta \cdot \text{KL}(\pi_\theta(\phi_i), \pi_{\theta_{\text{old}}}(\phi_i))$$

---

### 3. Novelty Demarcation for Project 14
- **Established by RLAF (ICLR 2026)**: One-shot root guidance optimized via GRPO on downstream CDCL solver feedback.
- **Project 14 Novelty (Milestone M5)**: RLAF only runs *one-shot at the root* ($t=0$). If the solver enters a hard combinatorial subtree, the one-shot prior becomes stale. Project 14 builds the **V2 Adaptive Cost-Aware Controller (Trust → Refresh → Branch)**, using OOD Mahalanobis distance, uncertainty proxies, and a learned refresh-benefit model to decide *when* to pay the cost of subsequent GNN refreshes during search.

