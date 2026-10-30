# Research Dossier: Mahalanobis OOD Detection (NeurIPS 2018)

**Title:** A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks  
**Authors:** Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin (KAIST & University of Michigan)  
**Venue:** Conference on Neural Information Processing Systems (NeurIPS 2018)  
**Primary Artifact:** [`mahalanobis_ood_neurips2018.pdf`](mahalanobis_ood_neurips2018.pdf)

---

## 1. Executive Summary & Problem Formulation

In machine learning, models trained on an empirical distribution $P_{\text{train}}$ often produce high-confidence, catastrophically wrong predictions when presented with out-of-distribution (OOD) test samples $x \sim P_{\text{OOD}}$.

In SAT solving, this issue is acute: a neural heuristic trained on synthetic random 3-SAT or graph coloring will encounter real-world industrial benchmarks (e.g., cryptographic SHA-256 pre-image inversion or microprocessor pipelining) whose clause-variable topological structure is completely alien. Applying uncalibrated neural priors to such formulas results in severe solver stalling and decision explosion.

**Lee et al.'s Core Thesis:**  
Deep neural network feature representations can be modeled as a class-conditional Gaussian distribution. By computing the empirical mean vectors and tied covariance matrix across training representations, test samples can be rigorously audited using the **Mahalanobis Distance** in latent feature space without retraining the neural network.

---

## 2. Mathematical Formulation

### 2.1 Feature Representation & Gaussian Modeling
Let $f(x) \in \mathbb{R}^d$ denote the latent representation extracted from a deep neural network layer. The training representations are assumed to follow a mixture of Gaussians:
$$P(f(x) \mid y = c) = \mathcal{N}\left( f(x) \mid \hat{\mu}_c, \hat{\Sigma} \right)$$
where $\hat{\mu}_c$ is the empirical class mean for class $c$, and $\hat{\Sigma}$ is the tied empirical covariance matrix:
$$\hat{\mu}_c = \frac{1}{N_c} \sum_{i: y_i = c} f(x_i)$$
$$\hat{\Sigma} = \frac{1}{N} \sum_{c} \sum_{i: y_i = c} (f(x_i) - \hat{\mu}_c)(f(x_i) - \hat{\mu}_c)^T$$

### 2.2 Mahalanobis Distance Metric
For an unseen test input $x$, compute the minimum Mahalanobis distance to the closest training centroid:
$$M(x) = \min_c (f(x) - \hat{\mu}_c)^T \hat{\Sigma}^{-1} (f(x) - \hat{\mu}_c)$$
A threshold $\tau$ separates in-distribution inputs ($M(x) \le \tau$) from out-of-distribution inputs ($M(x) > \tau$).

---

## 3. Direct Adaptation and Synthesis into NeuroCDCL

In our research, Lee et al.'s Mahalanobis framework forms the **mathematical bedrock of our Trust & Gating Architecture**:

### 3.1 Graph-Level OOD Auditing for CNF Formulas
We adapt the framework to the pooled whole-graph embedding $\mathbf{z}_\phi \in \mathbb{R}^d$ of a CNF formula $\phi$:
1. **Regularized Covariance Inversion:**  
   Because CNF feature matrices may have near-collinear dimensions, we compute a ridge-regularized precision matrix:
   $$\Lambda = \left( \hat{\Sigma} + 10^{-4} \cdot \mathbf{I} \right)^{-1}$$
2. **Formula OOD Score:**
   $$\mathcal{D}_{\text{Mahalanobis}}(\phi) = (\mathbf{z}_\phi - \hat{\mu}_{\text{train}})^T \Lambda (\mathbf{z}_\phi - \hat{\mu}_{\text{train}})$$
3. **The Gating Contract (Zero Overfitting / Zero Ad-Hoc Band-Aids):**  
   If $\mathcal{D}_{\text{Mahalanobis}}(\phi) > \tau_{\text{OOD}}$, the system classifies the formula as structurally out-of-distribution. Instead of letting corrupted neural weights derail the search, the solver **smoothly bypasses the neural prior and runs unguided stock CDCL**.
4. **Result:**  
   Guarantees that neural guidance can never trigger catastrophic regressions on unseen industrial domains, transforming catastrophic losses into safe, bounded execution.
