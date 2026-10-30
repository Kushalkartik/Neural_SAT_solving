# Technical Dossier 09: Reinforcement Learning, DAgger & The M4 Autonomous Marathon

> **Archive ID:** `FE-DOSSIER-09`  
> **Scope:** The 26-Hour M4 Autonomous Laboratory Marathon (Versions V1 through V7), Tree Teacher Distillation (DAgger), Mixture-of-Experts (MoE) Routing, Offline/Online RL, The 4-Instance Cap Trap, and The UNSAT Training Data Desert.  
> **Primary Laboratory:** `DAC-203-P14-M3.5-LATENCY-LAB\19_ORACLE_TO_STUDENT_SOTA_DECISION_LAB`  
> **Primary Source Files Audited:**  
> - `26_FINAL/01_SCIENTIFIC_GAP_REPORT.md`  
> - `21_FAILURE_ANALYSIS/M4_FAILURE_CASEBOOK.md`  
> - `26_FINAL/DECISION_MASS_MAP.md`  
> - `08_DAGGER/` & `05_TREE_TEACHER/`  
> - `11_MOE/` & `avx2_neural_router.h`  
> - `12_RL_ENVIRONMENT/`, `13_OFFLINE_RL/`, `14_ONLINE_RL/`  

---

## 1. Executive Summary

During Milestone M4, our autonomous research laboratory executed a continuous **26-hour experimental marathon**, progressing through seven sequential champion architectures (Versions V1 through V7). 

The mission was to minimize the total number of search decisions on the Canonical 163-Instance Suite below a target of **$\le 12,009$ decisions (a 5.0x reduction)** using:
1. **Tree Teacher Lookahead & DAgger:** Training student neural policies to imitate a multi-step sequence lookahead oracle.
2. **Counterfactual Dataset Factories:** Harvesting 44,000 state transitions across 321 counterfactual search tree states.
3. **Mixture-of-Experts (MoE) Neural Routing:** Specializing branching policies for structured, random, and industrial instances (`E_STRUCT`, `E_RANDOM_HARD`, `E_INDIST`) via AVX2 SIMD in-register routers.
4. **Reinforcement Learning (Offline CQL & Online PPO):** Reward shaping based on learned clause Literal Block Distance (LBD) and conflict frequency.

**The Catastrophic Finding:** Rather than dropping to 12,000 decisions, total decisions exploded from **60,047 (M1 baseline) up to 310,661 decisions (5.17x MORE decisions)** in the student models.

This dossier documents the root causes of this explosion: **The UNSAT Training Data Desert**, **The 4-Instance Cap Trap**, and **The Credit Assignment Horizon in Reinforcement Learning**.

---

## 2. The 7 Sequential Champion Iterations (V1 to V7)

```
+----------------------------------------------------------------------------------------------------+
|                             THE M4 AUTONOMOUS CHAMPION PROGRESSION                                 |
+----------------------------------------------------------------------------------------------------+
| Version | Core Algorithmic Mechanism Injected            | Decisions (163 Suite) | Wall Time (s)   |
+---------+------------------------------------------------+-----------------------+-----------------+
| **M1**  | Baseline Graph-Q-SAT (NeurIPS 2020)            |        60,047         |    412.13s      |
| **V1**  | Supervised Behavioral Cloning on MiniSat Trace |       328,450         |     98.40s      |
| **V2**  | Lookahead DAgger Student (Depth <= 12 gating)  |       318,120         |     84.10s      |
| **V3**  | Two-Sided Jeroslow-Wang (JW-2) Polarity Head   |       324,590         |     82.50s      |
| **V4**  | Conflict-Burst Safety Valve (VSIDS handoff)    |       315,900         |     78.20s      |
| **V5**  | Mixture-of-Experts Router (3 Domain Experts)   |       312,400         |     74.60s      |
| **V6**  | Counterfactual Hard-Negative Mining            |       311,850         |     72.10s      |
| **V7**  | Native AVX2 Register-Pinned In-Search Refresh   |       310,661         |     70.86s      |
+----------------------------------------------------------------------------------------------------+
```

While wall-clock time was successfully reduced from **412.13s down to 70.86s (a 5.82x speedup)** by replacing Python GNNs with C++ AVX2 kernels, the total decision count remained stubbornly pinned at **~310,000 decisions**, failing to achieve the $\le 12,009$ target.

---

## 3. The 3 Fatal Root Causes of the Decision Explosion

### Cause 1: The UNSAT Training Data Desert
* **The Forensic Discovery (`01_SCIENTIFIC_GAP_REPORT.md`):**
  * When auditing `counterfactual_dataset_v1.npz`, we discovered that all 44,000 state transitions had been harvested exclusively from satisfiable SATLIB `uf50-218` benchmarks.
  * **The Structural Asymmetry between SAT and UNSAT:**
    * In **Satisfiable (SAT) formulas**, the optimal search policy prioritizes variables and polarities that navigate toward an existing truth assignment. The objective is *solution discovery*.
    * In **Unsatisfiable (UNSAT) formulas**, no solution exists. The optimal search policy must drive the solver directly toward *contradictions and failed literals* to derive short 1-UIP asserting clauses and minimize resolution refutation proof size.
  * Because the student model had zero exposure to UNSAT refutation trees, its learned polarity head was heavily biased toward `True` assignments that preserved open subtrees.
  * When deployed on unsatisfiable formulas, this "satisfaction bias" actively steered the solver **away from conflicts**, preventing BCP from detecting contradictions and forcing the solver down massive, unpruned exponential branches.

---

### Cause 2: The 4-Instance Cap Trap (Artificial Baseline Metrics)
* **The Forensic Discovery (`01_SCIENTIFIC_GAP_REPORT.md` & `DECISION_MASS_MAP.md`):**
  * Why did the 163-instance decision count explode from 60,047 to 310,661?
  * We broke down decisions instance by instance. The 20 hard random 3-SAT instances (`gen_rand3sat_hard`) accounted for **305,572 out of 310,661 decisions (98.36% of all decisions in the entire benchmark suite)**!
  * Even more revealing: **just 4 instances** accounted for the vast majority of the difference:

| Benchmark Instance | M1 Baseline Decisions | SOTA Student Decisions | The Hidden Baseline Reality |
|---|:---:|:---:|---|
| `hard_rand3sat_n200_00` | **10,000 (Capped)** | **51,990 (Solved)** | M1 halted at artificial 10,000 decision cap! |
| `hard_rand3sat_n200_01` | **10,000 (Capped)** | **57,567 (Solved)** | M1 halted at artificial 10,000 decision cap! |
| `hard_rand3sat_n200_03` | **5,868 (Timed Out)** | **54,683 (Solved)** | M1 timed out at 30 seconds after only 5.8k decs! |
| `hard_rand3sat_n200_04` | **3,972 (Timed Out)** | **31,484 (Solved)** | M1 timed out at 30 seconds after only 3.9k decs! |
| **Sum of These 4 Instances**| **29,840 decisions** | **195,724 decisions** | **165,884 Artificial "Decision Explosion"** |

* **The Reality Check:**
  * In the M1 baseline, instances were terminated when they exceeded an arbitrary cutoff of 10,000 decisions (`CAP_EXCEEDED`) or timed out at 30 seconds.
  * The fast C++ SOTA solver (with its 1.3 $\mu$s cycle time) did not time out; it ran all the way to completion, taking 51,000 to 57,000 decisions to legitimately solve the problems!
  * **Over 63% of the "decision explosion" was an artifact of comparing a capped/timed-out baseline against a solver that ran to verified completion.**

---

### Cause 3: The Credit Assignment Horizon in Reinforcement Learning
* **The Formulation (`12_RL_ENVIRONMENT` & `14_ONLINE_RL`):**
  * We modeled CDCL as a Markov Decision Process (MDP):
    * State $s_t$: Current residual clause graph, assignment trail, and LBD history.
    * Action $a_t$: Choice of branching literal $(v, \text{sign})$.
    * Reward $r_t$: Positive reward (+1.0) if a conflict derives an asserting clause with $\text{LBD} \le 3$; negative step penalty ($-0.01$) per decision.
* **Why Reinforcement Learning Failed to Learn Better Branching:**
  1. *Extreme Temporal Delay:* In industrial SAT, a branching decision made at decision level 5 might only participate in a conflict analysis **10,000 to 50,000 steps later** at decision level 40.
  2. *The Discount Factor Horizon:* Standard temporal difference learning ($\text{TD}(\lambda)$) decays rewards by $\gamma^t$. For any discount factor $\gamma < 1$ (e.g., $\gamma = 0.99$), after $t = 1,000$ steps:
     $$\gamma^{1000} \approx (0.99)^{1000} \approx 4.3 \times 10^{-5} \approx 0$$
     The credit signal for the root decision completely vanishes.
  3. *Credit Smearing:* Assigning a shared reward to all 10,000 intermediate decisions rewards thousands of irrelevant variables that happened to be assigned while search was stuck in a barren subtree, corrupting the policy gradient with pure variance.

---

## 4. Mixture of Experts (MoE) Routing Failures

In `11_MOE` and `avx2_neural_router.h`, we implemented an AVX2 SIMD in-register neural router to dynamically route instances to one of three specialized policy networks:
* **`E_STRUCT`:** Specialized for structured verification circuits (prioritizes high clause-density bridge variables).
* **`E_RANDOM_HARD`:** Specialized for random 3-SAT near the phase transition (prioritizes balanced 2-sided Jeroslow-Wang scores).
* **`E_INDIST`:** Specialized for symmetric and expander formulas.

### Why MoE Routing Degraded Performance:
1. **Router Misclassification on Hybrid Instances:** Real industrial benchmarks are not pure; a hardware model-checking instance contains both structured datapath modules and chaotic control-logic state machines. Misclassifying an instance and deploying `E_RANDOM_HARD` on circuit logic multiplied decisions by 4x.
2. **Phase Saving Disruption during Mid-Search Expert Switching:** When the router dynamically switched experts mid-search (based on conflict burst signals), the new policy abruptly flipped the preferred polarities of variables, invalidating Kissat's phase-saving cache (`saved_phase[v]`) and triggering massive redundant search re-explorations.

---

## 5. Definitive Takeaways from the M4 Marathon

1. **You Cannot Train on SAT Alone:** Machine learning models trained exclusively on satisfiable formulas develop an inherent "satisfaction bias" that destroys refutation performance on unsatisfiable subtrees.
2. **Beware of Capped Baselines:** Capping difficult benchmark instances in early experiments creates a false baseline that makes complete, uncapped solvers appear to suffer from "decision explosions."
3. **Reinforcement Learning Cannot Bridge the CDCL Credit Horizon:** The temporal gap between a branching decision and its resulting 1-UIP conflict lemma ($10^4$ steps) exceeds the mathematical credit assignment horizon of discounted reinforcement learning algorithms.
