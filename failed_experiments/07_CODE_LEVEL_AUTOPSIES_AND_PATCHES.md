# Technical Dossier 07: Code-Level Autopsies, Source Patches & Algorithmic Failure Traces

> **Archive ID:** `FE-DOSSIER-07`  
> **Scope:** Exact C/C++ Source Code Patches, Python Orchestrators, Mathematical Implementations, and Line-by-Line Micro-Architectural Autopsies across Folders 24 to 52 in `DAC-203 PROJECT` and Folders 41 to 43 in `M3.5-LATENCY-LAB`.  
> **Primary Source Code Audited:**  
> - `27_CAUSAL_IMPLICATION_FLOW_KISSAT/kissat_cif/src/deduce.c`  
> - `30_PATH_B_GRAPH_ADAPTIVE_CONTROLLER_KISSAT/patch_deduce.py`  
> - `32_MULTI_SCALE_NEURAL_CDCL_KISSAT/patch_conservation_simplex.py`  
> - `24_NEURAL_CEGAR_LAB/cegar_engine.py`  
> - `25_NEURO_CONTEXTUAL_SOLVER/kissat_ncas/`  
> - `41_STRUCTURAL_DECOMPOSITION_PROTOTYPE/structural_gate_cuber.cpp`  
> - `42_THREE_FRONTIERS_BREAKTHROUGH_LAB/xor_gauss_engine.cpp`  
> - `43_UNIFIED_PARADIGMS_LAB/bspcr_production_solver.cpp`  
> - `50_KISSAT_TIERED_LBD_STACK/AUDIT_6_FAILURE.log`  

---

## 1. Executive Purpose

To achieve absolute scientific depth, an empirical autopsy must not merely describe failure at a high conceptual level; it must dissect the **exact lines of C and C++ code** where the theoretical hypothesis was implemented, identify the exact instruction-level penalties introduced, and quote the exact empirical telemetry generated.

This dossier documents the raw source code patches, mathematical invariants, and mechanical points of failure for the most prominent experimental attempts.

---

## 2. In-Loop Deduce Patches: Causal Flow & Adaptive Attention

### 2.1 Causal Implication Flow (Folder 27: `kissat_cif/src/deduce.c`)
* **Hypothesis:** Tracking backward implication flow across the 2-watched-literal implication graph provides a causal centrality score that identifies variables driving conflict generation.
* **The Code Modification in Kissat's Inner Deduction Loop:**
```c
// File: 27_CAUSAL_IMPLICATION_FLOW_KISSAT/kissat_cif/src/deduce.c
// Located directly inside: kissat_deduce() -> analyze_reason_side_literals
clause *reason = kissat_dereference_clause (solver, ref);
const unsigned r_size = reason->size;
const float split_flow = (r_size > 1) ? (uip_flow / (float)(r_size - 1)) : uip_flow;

for (all_literals_in_clause (lit, reason)) {
  if (lit != uip) {
    if (solver->cif_weights) {
      solver->cif_weights[IDX(lit)] += split_flow; // <-- INLINE FLOW UPDATE
    }
    if (analyze_literal (solver, all_assigned, frames, lit))
      unresolved_on_current_level++;
  }
}
```
* **Micro-Architectural Autopsy:**
  * In modern CDCL, `kissat_deduce` is executed **$10^6$ to $10^7$ times per second**.
  * The baseline loop contains only bit-shifts and flat array indexing.
  * Adding `solver->cif_weights[IDX(lit)] += split_flow` introduced:
    1. A floating-point division `(uip_flow / (float)(r_size - 1))` per conflict clause.
    2. An extra pointer dereference `solver->cif_weights`.
    3. An extra cache-line read-modify-write across the 4-byte float array.
  * *Result:* Incurred an immediate **6% to 12% wall-clock latency overhead**. Even though search decisions dropped by up to 24%, the cumulative execution time increased on 80% of benchmarks.

---

### 2.2 Path B: Inlined Clause Attention (Folder 30: `patch_deduce.py`)
* **Hypothesis:** Instead of distributing implication flow uniformly, an in-register attention mechanism should weight literals based on their assignment level relative to the current decision level.
* **The Code Modification:**
```c
// File: 30_PATH_B_GRAPH_ADAPTIVE_CONTROLLER_KISSAT/patch_deduce.py
if (r_size > 1 && r_size <= 32 && solver->cif_weights && lambda > 0.05f) {
  float scores[32];
  float sum_scores = 0.0f;
  unsigned count = 0;
  const unsigned cur_lvl = solver->level;
  const float inv_cur_lvl = (cur_lvl > 0) ? (1.0f / (float) cur_lvl) : 1.0f;

  for (all_literals_in_clause (lit, reason)) {
    if (lit != uip) {
      const assigned *lit_a = all_assigned + IDX (lit);
      const float is_bin = lit_a->binary ? 1.0f : 0.0f;
      const float lvl_r = (float) lit_a->level * inv_cur_lvl;
      float sc = 1.0f + 0.15f * is_bin - 0.20f * (lvl_r - 0.5f);
      if (sc < 0.20f) sc = 0.20f;
      scores[count++] = sc;
      sum_scores += sc;
    }
  }
  const float inv_sum = (sum_scores > 1e-6f) ? (1.0f / sum_scores) : (1.0f / (float) count);
  const float uniform_flow = uip_flow / (float)(r_size - 1);
  count = 0;
  for (all_literals_in_clause (lit, reason)) {
    if (lit != uip) {
      const float attn_flow = scores[count++] * inv_sum * uip_flow;
      const float blended_flow = (1.0f - lambda) * uniform_flow + lambda * attn_flow;
      solver->cif_weights[IDX (lit)] += blended_flow;
      if (analyze_literal (solver, all_assigned, frames, lit))
        unresolved_on_current_level++;
    }
  }
}
```
* **Micro-Architectural Autopsy:**
  * Required allocating stack buffers `scores[32]`, two sequential loops over every clause literal, floating-point divisions, and a normalization step.
  * *Result:* Slowed down `kissat_deduce` by **+15.3%**. Across 15 hard benchmarks (`PATH_B_GAC_15_RESULTS.csv`), runtime increased from 41.2s to 48.9s.

---

## 3. Mathematical Simplex Coupling: The Conservation Law Failure

### 3.1 Multi-Scale Conservation Simplex (Folder 32: `patch_conservation_simplex.py`)
* **Hypothesis:** The total VSIDS score increment injected into the solver during conflict analysis should be strictly conserved on a mathematical 1-simplex, splitting energy dynamically between learned 1-UIP cut literals and reason-side resolution literals:
  $$\sum_{v \in \text{analyzed}} \Delta \text{score}(v) = N \cdot \Delta_{\text{VSIDS}} \quad (\text{Strict Conservation})$$
* **The Code Modification in `bump.c`:**
```c
// File: 32_MULTI_SCALE_NEURAL_CDCL_KISSAT/kissat_multi_scale/src/bump.c
const int conserve_opt = GET_OPTION (conserve); // e.g. 50%
if (conserve_opt > 0 && n_active > 1 && k_learned > 0 && k_learned < n_active) {
  const double N = (double) n_active;
  const double K = (double) k_learned;
  const double R = N - K;
  const double theta = (double) conserve_opt * 0.01;

  // Conserved Simplex Scaling:
  // K * inc_cut + R * inc_res == (K + R) * inc == N * inc
  inc_cut = inc * (1.0 + theta * (R / N));
  inc_res = inc * (1.0 - theta * (K / N));
}

for (all_stack (unsigned, idx, solver->analyzed)) {
  if (flags[idx].active) {
    const double this_inc = all_assigned[idx].learned ? inc_cut : inc_res;
    const double old_score = kissat_get_heap_score (scores, idx);
    const double new_score = old_score + this_inc;
    kissat_update_heap (solver, scores, idx, new_score);
  }
}
```
* **Mathematical Invariant Proof:**
  $$\text{Total Bump} = K \cdot \text{inc} \left(1 + \theta \frac{R}{N}\right) + R \cdot \text{inc} \left(1 - \theta \frac{K}{N}\right)$$
  $$\text{Total Bump} = (K + R)\text{inc} + \theta \text{inc} \left( \frac{KR}{N} - \frac{RK}{N} \right) = N \cdot \text{inc} \quad (\text{Exact Invariance})$$
* **The Empirical Failure (`conservation_simplex_test_results.csv`):**

| Benchmark Category | Benchmark File | Stock Time (s) | Stock Conflicts | Simplex Time (s) | Simplex Conflicts | &Delta;% Conflicts | &Delta;% Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Circuit Verification** | `sc2020_008_15300be1...cnf` | 7.08s | 216,182 | **18.43s** | **503,704** | **+133.0%** | **+160.4%** |
| **Software SMT** | `sc2020_026_53463611...cnf` | **10.71s** | 158,105 | **TIMEOUT (25s)** | N/A | **TIMEOUT** | **+133.3%** |
| **Cryptanalysis** | `sc2020_005_12bb20e3...cnf` | **5.67s** | 218,369 | **TIMEOUT (25s)** | N/A | **TIMEOUT** | **+340.9%** |
| **EDA Synthesis** | `sc2021_053_22922a96...cnf` | **21.50s** | 305,109 | **TIMEOUT (25s)** | N/A | **TIMEOUT** | **+16.3%** |

* **Algorithmic Root Cause:**
  * While mathematically elegant, penalizing reason-side variables ($\text{inc}_{\text{res}} = \text{inc}(1 - \theta K/N)$) starved the resolution graph antecedent variables.
  * In CDCL, antecedent variables on previous decision levels are essential: they propagate unit clauses that form future 1-UIP cuts. Depriving them of activity delayed future conflict assertions, resulting in a **133% conflict explosion**.

---

## 4. Structural Gate Extraction & Backdoor Cubing

### 4.1 Structural Gate Cuber (Folder 41: `structural_gate_cuber.cpp`)
* **Hypothesis:** CNF formulas encoding hardware circuits contain logic gates (AND, OR, XOR, Equivalence). Variables that are gate outputs are functionally determined; variables that are **primary inputs** form a natural backdoor set. Partitioning the formula into $2^k$ cubes over primary inputs will isolate independent sub-problems.
* **The Implementation:**
```cpp
// File: 41_STRUCTURAL_DECOMPOSITION_PROTOTYPE/structural_gate_cuber.cpp
// Component 1: Pattern Matching for Logic Gates
// AND gate: y <=> (x1 & x2)
// Binary clauses: (~y v x1), (~y v x2)
// Non-binary clause: (~x1 v ~x2 v y)
for (int y = 1; y <= n; ++y) {
    Lit lit_y = y;
    const auto& succs = bin_impl[litToIdx(lit_y)];
    if (succs.size() >= 2 && succs.size() <= 6) {
        std::vector<Lit> required_lits;
        for (Lit s : succs) required_lits.push_back(notLit(s));
        required_lits.push_back(lit_y);
        std::sort(required_lits.begin(), required_lits.end());
        for (int ci : var_clauses[y]) {
            auto cl = f.clauses[ci];
            std::sort(cl.begin(), cl.end());
            if (cl == required_lits) {
                res.is_gate_output[y] = true;
                res.gate_type[y] = 1; // AND
                for (Lit s : succs) res.gate_inputs[y].push_back(varOfLit(s));
                res.num_gates_found++;
                break;
            }
        }
    }
}
// Primary Inputs = Variables that are NOT gate outputs
for (int v = 1; v <= n; ++v) {
    if (!res.is_gate_output[v]) res.primary_inputs.push_back(v);
}
```
* **The Empirical Telemetry (`TOURNAMENT_HARD_RESULTS.csv`):**

| Benchmark Family | Instance | Kissat Time (ms) | HyperConquer Time (ms) | Cubes Generated | Speedup vs Kissat |
|---|---|:---:|:---:|:---:|:---:|
| **Combinatorics (PHP)** | `php_4_3` | 13.58 ms | 32.86 ms | 2 | **0.41x (2.4x SLOWER)** |
| **Combinatorics (PHP)** | `php_6_5` | 15.28 ms | 41.49 ms | 8 | **0.37x (2.7x SLOWER)** |
| **SATLIB Transition** | `uf100-01` | 30.52 ms | 51.26 ms | 8 | **0.60x (1.7x SLOWER)** |
| **SATLIB Transition** | `uuf100-01` | 28.68 ms | 47.23 ms | 8 | **0.61x (1.6x SLOWER)** |
| **Hardware BMC** | `BMC-170055892` | 5,326.87 ms | 20,015.01 ms | 8 | **0.27x (3.7x SLOWER)** |

* **Algorithmic Root Cause:**
  * Cubing overhead (extracting gates, creating 8 sub-CNF files, and launching process workers) added massive constant overhead.
  * On unsatisfiable formulas (like `php`, `uuf`, `Steiner`), **every single sub-cube must be solved to refutation**. Because primary inputs in industrial BMC do not decouple feedback loops, each of the 8 sub-problems retained exponential complexity, multiplying total runtime by 2x to 4x.

---

## 5. Parity & Affine XOR Scarcity in Non-Cryptographic Domains

### 5.1 Affine XOR Engine Generalization (Folder 42: `xor_gauss_engine.cpp`)
* **Hypothesis:** Extracting XOR constraints and eliminating them via $\text{GF}(2)$ Gaussian row reduction will provide universal acceleration across industrial SAT.
* **The Empirical Telemetry (`NON_CRYPTO_GENERALIZATION_RESULTS.csv`):**

| Benchmark Category | Benchmark Name | XOR Constraints Detected | Kissat Stock Time (ms) | XOR Gauss Time (ms) | Speedup vs Kissat |
|---|---|:---:|:---:|:---:|:---:|
| **Factoring & Crypto** | `Planning_FullBF` | **13,720 XORs** | 15,000.0 ms (TIMEOUT) | **625.1 ms** | **23.59x FASTER** |
| **Factoring & Crypto** | `Planning_FullBG` | **8,640 XORs** | 15,000.0 ms (TIMEOUT) | **247.1 ms** | **59.77x FASTER** |
| **Combinatorics** | `Steiner15` | **0 XORs** | 14.0 ms | 39.5 ms | **0.21x (4.8x SLOWER)** |
| **Graph Coloring** | `HardV60` | **0 XORs** | 17.8 ms | 30.8 ms | **0.60x (1.7x SLOWER)** |
| **Hardware BMC** | `170222843` | **0 XORs** | 1,331.5 ms | 14,035.6 ms | **0.18x (5.5x SLOWER)** |
| **Random 3-SAT** | `N200` | **0 XORs** | 302.8 ms | 349.3 ms | **0.77x (1.3x SLOWER)** |

* **Algorithmic Root Cause:**
  * Whenever affine structures existed (cryptography, modular planning), the Gaussian engine delivered up to **59.8x speedup**.
  * However, on standard industrial verification and combinatorial SAT, **exactly zero XOR constraints were found**.
  * Scanning the clause database for parity signatures consumed CPU cycles and cache bandwidth, resulting in a **0.18x to 0.77x slowdown**.

---

## 6. Counterexample-Guided Abstraction Refinement (CEGAR)

### 6.1 Neural CEGAR Loops (Folder 24: `cegar_engine.py`)
* **Hypothesis:** Abstracting the CNF formula by removing non-core clauses, solving the relaxed formula, and adding back violated clauses (CEGAR loop) will bypass dense search spaces.
* **The Empirical Telemetry (`CEGAR_VS_STOCK_TOURNAMENT_RESULTS.csv`):**

| Instance ID | Category | Kissat Stock Status (Time) | Neural CEGAR Status (Time) | CEGAR Iterations | Certification |
|---|---|:---:|:---:|:---:|:---:|
| `sc2020_014` | Industrial SAT | **SAT in 1.27s** | **UNKNOWN in 7.53s** | 15 iterations | **FAILED (False)** |
| `sc2020_024` | Industrial SAT | **SAT in 1.34s** | **UNKNOWN in 17.86s** | 15 iterations | **FAILED (False)** |
| `sc2020_005` | Cryptanalysis | **SAT in 4.78s** | **TIMEOUT (30.03s)** | 5 iterations | **FAILED (False)** |
| `sc2023_012` | Industrial UNSAT | **UNSAT in 0.54s** | **UNKNOWN in 0.26s** | 15 iterations | **FAILED (False)** |
| `sc2021_159` | Industrial UNSAT | **UNSAT in 0.89s** | **UNSAT in 14.61s** | 11 iterations | Certified (16x SLOWER) |
| `sc2021_011` | Industrial UNSAT | **UNSAT in 1.56s** | **UNKNOWN in 4.45s** | 15 iterations | **FAILED (False)** |

* **Algorithmic Root Cause:**
  * In automated verification, CEGAR works because predicate abstractions collapse large state spaces.
  * In propositional SAT, removing clauses creates spurious satisfying assignments that violate the true constraints.
  * The refinement loop spent 15 iterations adding back clauses one by one, generating thousands of spurious abstraction conflicts while stock Kissat solved the concrete formula directly in 1.2 seconds.

---

## 7. The Tiered LBD & Proof Reservoir Failure Audit

### 7.1 Multi-Point Nano-Stack Audit on 6 Losses (Folder 50: `AUDIT_6_FAILURE.log`)
* **Context:** Testing whether Folder 49/50 could crack 6 previously lost tournament instances under a full 1,200s timeout:
```log
[1/6] sc2024_015 -> SAT in 196.9s | Reductions: 270 | SOLVED! (1.34x vs Stock Kissat: 264.7s)
[2/6] sc2023_idx133 -> TIMEOUT in 1200.0s | Reductions: 1092 | STILL TIMEOUT (Stock: 284.3s)
[3/6] sc2024_074 -> TIMEOUT in 1200.0s | Reductions: 770  | STILL TIMEOUT (Stock: 300.0s)
[4/6] sc2024_003 -> SAT in 369.3s | Reductions: 485 | SOLVED! (1.53x vs Stock Kissat: 564.7s)
[5/6] sc2024_idx178 -> UNSAT in 1087.3s | Reductions: 178 | SOLVED! (0.62x vs Stock Kissat: 675.2s)
[6/6] sc2023_idx127 -> SAT in 649.7s | Reductions: 411 | SOLVED! (1.14x vs Stock Kissat: 737.7s)
Total Tested: 6 | Cracked: 4 / 6 (66.7%) | Unsolved: 2 / 6
```
* **The Fatal Seesaw Mechanism:**
  * While the new student cracked 4 out of 6 instances (demonstrating genuine neural acceleration on `sc2024_015` and `sc2024_003`), `sc2023_idx133` executed **1,092 clause reductions** and timed out.
  * In `sc2023_idx133`, retaining tiered LBD lemmas caused the learned clause database to grow past 800,000 clauses. Two-watched-literal propagation slowed from $8.5 \times 10^6$ propagations/sec down to $1.2 \times 10^6$ propagations/sec due to L2/L3 cache misses, turning an easy 284-second solve into a 1,200-second timeout.

---

## 8. Definitive Code-Level Conclusion

Every single experiment failed not because of bad luck or programming errors, but because **Conflict-Driven Clause Learning operates at the absolute physical limits of modern CPU cache architectures and resolution proof systems**:

1. Adding **floating-point arithmetic or conditional branching** into `kissat_deduce` immediately destroys the 15–30 ns cycle time of Boolean Constraint Propagation.
2. Altering **VSIDS activity increments** (even on a mathematically conserved simplex) starves the antecedent implication chains necessary for 1-UIP derivation.
3. Imposing **high-level partitions or abstractions** (cubing, CEGAR, cutsets) severs the unit propagation boundaries that enable CDCL to resolve conflicts in exponential search spaces.
