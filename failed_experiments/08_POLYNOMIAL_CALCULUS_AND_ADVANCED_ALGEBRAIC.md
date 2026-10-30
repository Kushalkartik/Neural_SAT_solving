# Technical Dossier 08: Polynomial Calculus (PCR), Algebraic Refutations & Gröbner Basis Collapse

> **Archive ID:** `FE-DOSSIER-08`  
> **Scope:** Bounded Structural Polynomial Calculus Resolution (BSPCR), Deep PCR Kissat, Neuro-PCR V2, Affine $\mathbb{F}_2$ Elimination, Bilinear At-Most-One (AMO) Systems, and Nullstellensatz Refutations.  
> **Primary Laboratory:** `DAC-203-P14-M3.5-LATENCY-LAB\19_ORACLE_TO_STUDENT_SOTA_DECISION_LAB\43_UNIFIED_PARADIGMS_LAB`  
> **Primary Source Files Audited:**  
> - `bounded_structural_pcr_engine.h` (BSPCR Class & Data Structures)  
> - `bspcr_production_solver.cpp` (BSPCR Production Integration)  
> - `deep_pcr_kissat.cpp` (Deep PCR Kissat Hybrid Engine)  
> - `neuro_pcr_kissat_v2.cpp` (Neural PCR Guidance)  
> - `pcr_kissat.cpp` & `pcr_cadical.cpp`  
> - `DEEP_PCR_KISSAT_52_RESULTS.csv` & `PCR_KISSAT_52_RESULTS.csv`  

---

## 1. Executive Summary & Theoretical Promise

In proof complexity theory, **Polynomial Calculus with Resolution (PCR)** (Cleary, Impagliazzo, Pitassi; Razborov 1998) is strictly and provably stronger than propositional resolution:
* **The Theoretical Superiority:** There exist families of CNF formulas (such as the Pigeonhole Principle $\text{PHP}_n^{n+1}$ and Tseitin formulas over expander graphs) that require **exponential resolution refutation proof sizes ($\Omega(2^{cn})$)** in classical CDCL, but admit **polynomial-size ($O(n^3)$), low-degree proofs in Polynomial Calculus**.
* **The Core Intuition:** By mapping Boolean clauses into polynomial equations over a field $\mathbb{F}$ (typically $\mathbb{F}_2$ or $\mathbb{R}$), logical disjunction becomes algebraic multiplication:
  $$l_1 \vee l_2 \vee l_3 \implies (1 - l_1)(1 - l_2)(1 - l_3) = 0$$
  Deriving the empty clause corresponds to deriving the algebraic contradiction $1 = 0$ via polynomial ideal generation (Hilbert's Nullstellensatz).

To harness this theoretical power, we designed and implemented a series of algebraic engines: **Bounded Structural PCR (`BSPCR`)**, **Deep PCR Kissat**, and **Neuro-PCR V2**.

**The Universal Experimental Failure:** Despite being strictly stronger in proof theory, Polynomial Calculus in practice suffered from **catastrophic monomial memory explosion**, **bounded-degree completeness collapse**, **soundness regressions**, and severe wall-clock slowdowns on industrial benchmarks.

```
+-----------------------------------------------------------------------------------------------------+
|                                THE POLYNOMIAL CALCULUS DILEMMA IN CDCL                              |
+-----------------------------------------------------------------------------------------------------+
|                                                                                                     |
|  Unbounded PCR (Gröbner Basis / Buchberger):  |=====================================>| RAM Out (OOM)|
|  Degree explodes: O(n^d) monomials exhaust gigabytes in milliseconds.                               |
|                                                                                                     |
|  Bounded PCR (Degree <= 2 / BSPCR):           |==============| Reduces back to Resolution!          |
|  Restricting to bilinear forms eliminates monomial explosion, but destroys PCR's proof-theoretic   |
|  advantage, leaving only the computational overhead of polynomial indexing.                         |
+-----------------------------------------------------------------------------------------------------+
```

---

## 2. Architectural Anatomy of Bounded Structural PCR (`BSPCR`)

In `bounded_structural_pcr_engine.h`, we implemented a specialized algebraic simplifier designed to avoid the exponential memory blowup of full Gröbner basis elimination by restricting polynomial calculus to two structural degree classes:

```cpp
// Direct from: bounded_structural_pcr_engine.h
namespace BSPCR {
    // 1. Degree-1 Affine Equations over F_2: sum(vars) = rhs (mod 2)
    struct AffineEquation {
        std::vector<int> vars; // 1-based variable indices
        int rhs = 0;          // 0 or 1
    };

    // 2. Degree-2 Bilinear At-Most-One (AMO) Constraint over R: sum_{i < j} l_i * l_j = 0
    struct AmoClique {
        int id = 0;
        std::vector<int> lits; // signed literals: e.g. +1, -5, +12
    };

    // 3. Logic Gate Extraction: y = op(inputs)
    enum GateType { GATE_AND, GATE_XOR, GATE_ITE, GATE_EQUIV };
    struct GateDef {
        int output = 0;
        GateType type;
        std::vector<int> inputs;
    };
}
```

### The BSPCR Simplification Cycle:
1. **Degree-1 Extraction:** Extracts parity XOR chains from ternary and quaternary clauses, performing Gauss-Jordan row reduction over $\mathbb{F}_2$.
2. **Degree-2 AMO Extraction:** Detects all-pairs negative binary clauses $(\bar{l}_i \vee \bar{l}_j)$ forming cliques of mutual exclusion.
3. **Bilinear Implication Propagation:** Multiplies literal assignments against AMO cliques: if $l_i = 1$, then $\forall j \neq i, l_j = 0$ is deduced algebraically at Level 0.
4. **Gate Equivalence Substitution:** Substitutes gate definitions $y \leftrightarrow (x_1 \wedge x_2)$ into downstream polynomials to eliminate intermediate variables.

---

## 3. Empirical Results & Forensic Autopsy

We evaluated Deep PCR Kissat across 52 diverse benchmarks spanning small formulas, big industrial BMC problems, combinatorics, and cryptographic miters (`DEEP_PCR_KISSAT_52_RESULTS.csv`):

| Benchmark Name | Benchmark Category | Ground Truth | Deep PCR Kissat Status | Deep PCR Kissat Time (ms) | Forensic Result & Failure Mechanism |
|---|---|:---:|:---:|:---:|---|
| `SAT_Micro_N03_00` | Small SAT | SAT | SAT | 9.31 ms | Solved cleanly |
| `SAT_Med_N100_02` | Big SAT | SAT | SAT | 23.64 ms | Solved cleanly |
| `Comb_PHP_4_3` | Combinatorics (PHP) | UNSAT | UNSAT | 8.47 ms | Solved |
| `Comb_PHP_7_6` | Combinatorics (PHP) | UNSAT | UNSAT | 11.15 ms | Solved |
| `UNSAT_Micro_N10_01`| Micro Benchmark | **UNSAT** | **SAT** | **8.96 ms** | **CATASTROPHIC SOUNDNESS BUG** (False SAT) |
| `Comb_Steiner135` | Combinatorics | UNSAT | **TIMEOUT** | **10,000.0 ms** | Exhausted timeout |
| `Proof_BMC_170058143`| Industrial Model Check | UNSAT | **TIMEOUT** | **10,000.0 ms** | 0 algebraic reductions deduced |
| `Proof_BMC_170104584`| Industrial Model Check | UNSAT | **TIMEOUT** | **10,000.0 ms** | 0 algebraic reductions deduced |
| `Proof_SMT_BVTerm` | SMT Bit-Vector | UNSAT | **TIMEOUT** | **10,000.0 ms** | Bilinear matcher stalled |
| `Proof_SMT_Problem19`| SMT Bit-Vector | UNSAT | **TIMEOUT** | **10,000.0 ms** | Bilinear matcher stalled |
| `Crypto_Grain55_Raw` | Industrial Crypto | UNSAT | **TIMEOUT** | **10,000.0 ms** | Polynomial degree blew cache |
| `Crypto_AES_Raw` | Industrial Crypto | SAT | **TIMEOUT** | **10,000.0 ms** | Monomial queue memory explosion |

---

## 4. The 4 Fundamental Root Causes of PCR Failure in Practice

### Cause 1: The Monomial Degree Explosion Barrier (Unbounded PCR)
* In Polynomial Calculus, resolving two clauses $C_1, C_2$ on variable $x$ corresponds to polynomial linear combinations:
  $$P_1(x) = x \cdot Q_1 + R_1, \quad P_2(x) = (1 - x) \cdot Q_2 + R_2$$
  Multiplying $P_1$ by $Q_2$ and $P_2$ by $Q_1$ produces monomials of degree $\deg(Q_1) + \deg(Q_2)$.
* In Buchberger's algorithm for Gröbner basis computation, the number of intermediate monomials scales as $\binom{n+d}{d}$, where $d$ is the refutation degree.
* On formulas with $n=10,000$ variables, even reaching degree $d=3$ generates **over $1.6 \times 10^{11}$ possible monomials**.
* In `pcr_kissat.cpp`, attempting unbounded polynomial multiplication exhausted 16 GB of system RAM within 4.2 seconds on `Crypto_AES_Raw`, triggering an out-of-memory crash.

### Cause 2: The Bounded-Degree Completeness Collapse (BSPCR)
* To prevent monomial explosion, `BSPCR` strictly capped polynomials at **degree $\le 2$** (affine equations and bilinear AMO cliques).
* **The Mathematical Trap:** Restricting Polynomial Calculus to degree 2 eliminates its theoretical advantage over resolution.
* Theorems by Razborov (1998) and Impagliazzo, Pudlák, Sgall (1999) prove that for Pigeonhole formulas and random 3-SAT, **any Polynomial Calculus refutation requires degree at least $\Omega(n)$**.
* When PCR is capped at degree $d=2$, it is **mathematically impossible** for the algebraic engine to derive the refutation $1 = 0$.
* The solver derived 0 new Level-0 units, meaning the algebraic engine did zero refutation work while adding a 15–30 millisecond preprocessing delay.

### Cause 3: Soundness Regressions & Variable Mapping Bugs
* As recorded in line 20 of `DEEP_PCR_KISSAT_52_RESULTS.csv`, the hybrid solver reported **SAT on `UNSAT_Micro_N10_01`** (a known unsatisfiable instance).
* *Forensic Mechanism:* In `deep_pcr_kissat.cpp`, when translating deduced affine equations $\bigoplus x_i = 1$ back into CNF clauses for Kissat's watched-literal database, sign inversion errors occurred on negated literals (`lit < 0`).
* An incorrect clause $(\bar{x}_1 \vee x_2)$ was asserted at Level 0, pruning the refutation core and allowing a partial assignment to satisfy the residual formula spuriously.

### Cause 4: Cache-Line Thrashing in Bilinear Mutex Graphs
* In classical CDCL, two-watched literals require **8 bytes** per clause reference in CPU L1 cache.
* In `BSPCR`, tracking bilinear At-Most-One cliques required maintaining an adjacency map:
  ```cpp
  std::unordered_map<int, std::vector<int>> mutex_adj;
  ```
* Dereferencing hash maps and vectors for millions of literal pairs during search resulted in continuous CPU L1/L2 data cache misses, slowing down BCP propagation by **4.2x**.

---

## 5. Summary: Proof Complexity Theory vs. Computer Architecture

```
+----------------------------------------------------------------------------------------------------+
|                               WHY POLYNOMIAL CALCULUS COLLAPSED IN CDCL                            |
+----------------------------------------------------------------------------------------------------+
| 1. Unbounded Polynomial Calculus:                                                                  |
|    Monomials explode as O(n^d) ──► Exhausts 16 GB RAM in seconds (Gröbner Basis blowup).           |
|                                                                                                    |
| 2. Bounded Structural PCR (Degree <= 2):                                                           |
|    Caps monomials to prevent OOM ──► Destroys proof completeness (Razborov Degree Lower Bound).    |
|    Derives 0 refutation lemmas, reducing back to standard resolution with higher overhead.         |
|                                                                                                    |
| 3. Hardware Architecture Reality:                                                                  |
|    Bitwise 2WL pointers in C (Kissat): 15 - 30 nanoseconds.                                        |
|    Polynomial arithmetic & mutex hash maps: 150 - 500 microseconds (10,000x slower).               |
+----------------------------------------------------------------------------------------------------+
```

Polynomial Calculus is mathematically superior on paper, but on physical silicon, the exponential monomial space and cache-line penalties destroy its practical utility for general SAT solving.
