# ⚠️ V2.0 AUTHORITATIVE UPGRADE NOTICE — READ BEFORE USING THIS DOCUMENT

**This document has been upgraded to the Project 14 V2.0 Trust → Refresh → Branch architecture.**  
The V2.0 controller is the system of record for the final research candidate. Earlier formulations remain only as historical/baseline material unless explicitly re-used as an ablation. In particular, the former hand-weighted seven-parameter refresh gate is **not** the primary method and must not be described as a genuine value-of-computation estimator.

**Canonical final controller:**  
`Trust (OOD + uncertainty + search-turbulence) → Refresh (empirically learned future-benefit estimate vs measured refresh cost) → Branch (neural / VSIDS; optional blending only as a secondary ablation).`

**Scientific rule:** No expected winner, robustness claim, novelty claim, or positive performance outcome is assumed. All such statements must be established by the declared experiments.

**DAC 203 \| Project 14**

Literature & Novelty Audit

**Neural Branching Heuristics for SAT**

Document 14 \| v1.0 \| Research-control specification \| 15 September 2026

**Purpose.** This document freezes the literature-positioning and novelty-check procedure for Project 14. It distinguishes established prior art, project reproductions, extensions, and novelty candidates so that the team and any autonomous implementation agent do not accidentally claim an idea that is already present in prior work.

Audit basis: the project PRD plus targeted literature verification performed for this document. Literature status is time-sensitive; this audit should be refreshed immediately before a final paper/report is submitted. The audit does not constitute an exhaustive patent or prior-art legal search.

# 1. Executive Audit Verdict

| **Area**                                              | **Audit status**                      | **Implication for Project 14**                                                                                                                             |
|-------------------------------------------------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NeuroSAT / neural SAT reasoning                       | Established prior art                 | Do not claim GNN-based SAT reasoning or clause-variable message passing as novel.                                                                          |
| Neural guidance of classical SAT solvers              | Established prior art                 | The project is an integration/extension study, not a first neural-CDCL hybrid.                                                                             |
| Graph-Q-SAT                                           | Direct baseline / reproduction target | M1 is a reproduction, not a novelty contribution.                                                                                                          |
| NeuroCore-style periodic neural guidance              | Established prior art                 | Periodic external neural updates are explicitly prior art.                                                                                                 |
| NeuroComb-style one-shot + periodic hybrid guidance   | Established prior art                 | M2 and several M3 design ingredients overlap prior work; novelty must not be attached to these components individually.                                    |
| 2026 RLAF one-shot SAT guidance                       | Directly relevant current prior art   | One-shot GNN-generated weights + polarity + solver-cost RL is now clearly prior art; M2 must be framed as a controlled reproduction/ablation, not novelty. |
| Trust-aware empirical cost-calibrated refresh/reuse of learned guidance | Novelty candidate only                | Potentially differentiating, but requires an explicit claim-to-prior-art comparison and controlled evidence before any novelty claim.                      |
| Importance + polarity + uncertainty controller        | Candidate extension                   | Treat as an intermediate experimental extension unless the literature audit proves a genuinely new mechanism.                                              |

# 2. Project Contribution Taxonomy

| **Label** | **Definition**                         | **Project examples**                                                                                                     |
|-----------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| R         | Reproduction / verification            | M0 VSIDS baseline; M1 Graph-Q-SAT-faithful baseline.                                                                     |
| P         | Prior-art implementation               | One-shot neural prior; periodic neural guidance; basic learned/classical handoff where directly described by prior work. |
| E         | Engineering / systems contribution     | Stable solver hook, graph caching, logging, deterministic testing, overhead accounting.                                  |
| X         | Experimental extension                 | Importance + polarity + uncertainty; controller features; matched neural-call budgets.                                   |
| N?        | Novelty candidate, not yet established | Adaptive, cost-aware neural refresh/reuse governed by value-of-computation.                                              |
| F         | Finding / empirical result             | Which instances benefit, when overhead dominates, size/structure generalization, or negative results.                    |

# 3. Core Literature Map

| **Work**                                             | **What it establishes**                                                                                                                                                                 | **Overlap with Project 14**                                                                   | **Project treatment**                                                                                          |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| NeuroSAT (Selsam et al., ICLR 2019)                  | MPNN-based SAT reasoning; satisfiability prediction and emergent search behavior; cross-distribution generalization.                                                                    | Graph representation and neural SAT reasoning background.                                     | Background only; not a direct solver-heuristic baseline.                                                       |
| NeuroCore / Selsam & Bjørner (SAT 2019)              | GNN predicts unsat-core variables; predictions periodically replace solver variable activity scores; applied to MiniSat/Glucose/Z3.                                                     | Directly overlaps neural guidance, CDCL integration, and periodic intervention.               | Cite as prior art; use to define the boundary around M3/M5.                                                    |
| Graph-Q-SAT (Kurin et al., NeurIPS 2020)             | DQN + GNN branching heuristic integrated with MiniSat; learned from solver interaction; generalization across size/domain.                                                              | Direct architecture, RL framing, graph interface, MiniSat integration, baseline objective.    | M1 faithful reproduction target; no novelty claim.                                                             |
| NeuroComb (Wang et al., 2021/2022)                   | Predicts important variables/clauses once; combines static neural information with dynamic VSIDS; periodically applies guidance; multi-armed bandit chooses static-vs-dynamic behavior. | Strong overlap with one-shot guidance, hybrid branching, periodic use, resource practicality. | Prior art that sharply constrains novelty claims around M2-M4.                                                 |
| RLAF one-shot guidance (Tönshoff & Grohe, ICLR 2026) | Single GNN pass produces variable weights and polarities; injected into existing branching heuristic; trained with RL using solver computational cost.                                  | Extremely close to M2 and some M4 ingredients.                                                | Must be explicitly discussed in final report; one-shot + polarity + cost-aware RL are not novel by themselves. |

## 3.1 What each source proves about the novelty boundary

- Graph-Q-SAT establishes that RL-trained GNNs can directly act as a learned branching heuristic inside MiniSat-style search. Therefore, “using a GNN to choose the next SAT variable” is not novel for this project.

- NeuroCore establishes periodic neural intervention in CDCL by replacing or resetting dynamic activity information with learned predictions. Therefore, “query the network periodically during solving” is not novel by itself.

- NeuroComb establishes a practical one-shot neural inference design combined with a dynamic VSIDS heuristic and repeated downstream use of static neural information. Therefore, one-shot inference plus periodic hybrid use cannot be claimed as a new idea.

- The ICLR 2026 RLAF paper establishes one-shot GNN guidance that outputs both variable weights and polarities and uses solver computational cost directly in reinforcement learning. Therefore, “one-shot GNN + learned weights + polarity + solver-cost reward” is current direct prior art.

- The remaining defensible novelty candidate is narrower: whether the project introduces a materially new mechanism for deciding when to pay for a fresh neural computation during an evolving CDCL search, where the decision explicitly trades expected branching benefit against measured/estimated neural computation cost. Even here, the claim remains provisional until a broader search is performed.

# 4. Detailed Novelty Risk Register

| **Potential claim**                                               | **Prior-art status**                       | **Risk**    | **Allowed wording**                                                                                                    |
|-------------------------------------------------------------------|--------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------|
| “We are the first to use GNNs for SAT branching.”                 | False / contradicted                       | Critical    | Do not use.                                                                                                            |
| “We reproduce Graph-Q-SAT and measure its true runtime overhead.” | Supported as project contribution          | Low         | Use as an evaluation/reproduction contribution if implementation is faithful and validated.                            |
| “We introduce one-shot neural SAT guidance.”                      | Not supportable as novel                   | Critical    | Describe as controlled reproduction/benchmarking of a now-established idea.                                            |
| “We combine importance, polarity, and uncertainty.”               | Not established as novel                   | High        | Call it an experimental feature combination or project extension unless a formal literature audit establishes novelty. |
| “We adapt neural refresh frequency to solver state.”              | Potentially differentiating but incomplete | High        | Call it a hypothesis / novelty candidate until prior-art sweep is complete.                                            |
| “We optimize the value of computation of neural refreshes.”       | Potential novelty candidate                | Medium-High | Use cautious language: “we investigate whether a cost-aware refresh policy can…”.                                      |
| “Our method is faster because it uses fewer neural calls.”        | Empirical claim only                       | Medium      | Must report end-to-end wall time and neural-call count separately; fewer calls alone is insufficient.                  |
| “Our method generalizes better.”                                  | Empirical claim only                       | Medium      | Require predefined size-shift and structural-shift tests plus statistical analysis.                                    |

# 5. Required Novelty Definition for M5

**M5 should be defined by mechanism, not by outcome.** A faster result on a particular benchmark does not itself establish methodological novelty. The proposed differentiator must be the policy that decides whether the solver should refresh neural guidance, reuse cached guidance, or fall back to classical guidance, together with an explicit cost model and an experimentally testable value-of-computation criterion.

## 5.1 Minimum mechanism that must exist before M5 can be called the novel component

- A persistent cache of neural guidance attached to the current solver/variable state.

- A measurable notion of guidance staleness or expected loss from reuse, derived from solver state and/or disagreement signals.

- A measurable or calibrated estimate of the cost of a fresh neural refresh, including graph extraction/build cost where applicable and model inference time.

- A decision rule that compares expected benefit from refreshing against that estimated cost.

- A fallback policy that can use classical VSIDS when refresh is judged not worthwhile.

- Ablations separating refresh timing from the neural scoring rule itself.

## 5.2 What would not count as sufficient novelty

- Adding a fixed refresh interval K without learning or cost reasoning.

- Calling the network less often merely because inference is expensive.

- Adding a confidence threshold without evidence that it estimates refresh value.

- Combining known signals such as conflicts, restarts, VSIDS concentration, and model confidence without a new decision rule or evaluation rationale.

- Replacing one hyperparameter K with another adaptive threshold selected on the test set.

# 6. Research Questions Reclassified by Literature Status

| **PRD RQ**                                                         | **Literature-aware status**       | **Interpretation**                                                                   |
|--------------------------------------------------------------------|-----------------------------------|--------------------------------------------------------------------------------------|
| RQ1: Can learned branching outperform classical baselines?         | Established research direction    | Valid research question; not itself novel.                                           |
| RQ2: Where does neural guidance help or hurt?                      | Open empirical analysis           | Strong project finding opportunity, especially with overhead decomposition.          |
| RQ3: What is the real neural overhead?                             | Open empirical / systems question | Potentially valuable because wall-clock cost can differ from decision-count benefit. |
| RQ4: Does a one-shot neural prior work?                            | Now heavily prior-art constrained | Use as controlled comparison and bridge to M5, not as novelty.                       |
| RQ5: Can a simple controller select neural vs classical?           | Prior-art adjacent                | Useful experiment; novelty requires a new controller mechanism.                      |
| RQ6: Does importance + polarity + uncertainty help?                | Feature-level extension           | Potentially useful, but no automatic novelty.                                        |
| RQ7: Can cost-aware refresh reduce unnecessary neural computation? | Primary novelty candidate         | Most defensible place for methodological contribution, still provisional.            |
| RQ8: Does the approach generalize under size/structure shift?      | Empirical validation              | Important for scientific value and claim strength, not novelty alone.                |

# 7. Literature-Grounded Positioning Paragraph for Final Report

**Recommended wording (subject to final audit):** “Prior work has established several ways of integrating neural information into SAT solving, including neural reasoning over SAT graphs, learned branching with reinforcement learning, periodic neural intervention in CDCL, and one-shot neural parameterization of classical branching heuristics. Project 14 therefore does not claim novelty for GNN-based SAT branching, one-shot neural guidance, polarity prediction, or neural/classical hybridization individually. Instead, the project investigates whether the computational cost of obtaining fresh neural guidance can be treated as part of the control problem itself: cached neural guidance is reused when its expected benefit remains high, while a fresh neural computation is triggered when its estimated value exceeds its measured or predicted cost. The resulting contribution is evaluated primarily as a solver-systems question, with explicit overhead accounting, paired ablations, and out-of-distribution tests.”

# 8. Anti-Overlap Rules for the Implementation Agent

- Never introduce a method and label it “novel” merely because it was not named in the PRD. Search the literature matrix in this document first.

- Never silently convert an implementation convenience into a research contribution. Engineering changes belong in the implementation section unless experimentally justified.

- Keep M1 as a reproduction lane. Do not optimize, simplify, or alter Graph-Q-SAT while calling it a faithful reproduction.

- Keep M2 visibly labeled as a prior-art/control condition. The existence of ICLR 2026 one-shot guidance makes this mandatory.

- Do not claim the M5 controller is unique until the research decision log records the final search keywords, databases, relevant papers found, and the exact differences between M5 and each closest prior method.

- Any literature-discovered overlap after implementation starts must trigger a Research Decision Log entry and, when material, a revision of the claim matrix before further experiments are interpreted.

# 9. Search Protocol Before a Final Novelty Claim

1.  Run a final search across Google Scholar, Semantic Scholar, arXiv, major SAT/constraint venues, and general web search for combinations of: SAT + neural branching; SAT + GNN + branching heuristic; CDCL + learned heuristic; SAT + adaptive neural heuristic; SAT + cost-aware neural inference; SAT + online neural guidance; SAT + periodic neural guidance; and SAT + refresh/reuse of learned heuristic.

2.  Search by mechanism as well as terminology. A prior paper may implement the same idea under a different phrase such as static guidance, periodically updated scores, adaptive solver selection, learned restart control, or neural hyper-heuristics.

3.  Check papers and repositories for implementation details, not only abstracts. The final claim must distinguish conceptual novelty from an engineering variant of an existing implementation.

4.  Record every close paper in the Research Decision Log with: citation, mechanism, solver type, neural-query frequency, controller logic, cost model, benchmark family, and exact overlap/difference with M5.

5.  Freeze the final novelty wording only after the search log is signed off. Do not tune the claim after seeing the final results.

# 10. Evidence Required to Support the M5 Claim

| **Evidence**                        | **Required result**                                                                      | **Why it matters**                                                   |
|-------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| Fixed-K vs adaptive refresh         | Adaptive policy compared against multiple fixed refresh intervals under matched budgets. | Shows benefit comes from adaptation, not a lucky K.                  |
| No-cost-term ablation               | Controller without neural-cost term.                                                     | Tests whether explicit cost-awareness matters.                       |
| No-refresh ablation                 | Guidance computed once and reused.                                                       | Separates refresh value from one-shot value.                         |
| Always-refresh upper-cost condition | Refresh at every eligible decision/checkpoint.                                           | Measures the maximum neural-computation burden.                      |
| Classical fallback                  | VSIDS-only baseline.                                                                     | Establishes whether neural computation is worthwhile at all.         |
| Matched-call-budget comparison      | Methods compared with approximately equal neural-call budgets.                           | Prevents “fewer calls” from being mistaken for “better policy.”      |
| Overhead decomposition              | Graph build, inference, controller, solver runtime all separately measured.              | Shows where the claimed systems benefit originates.                  |
| OOD evaluation                      | Size shift and structural shift.                                                         | Tests whether the controller is robust beyond training distribution. |

# 11. Claim Ladder

| **Level** | **Claim strength**                           | **Evidence bar**                                                                                                                    |
|-----------|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| L0        | Implementation exists                        | Pass correctness tests.                                                                                                             |
| L1        | Reproduces a known baseline                  | Independent M1 validation against the specified Graph-Q-SAT protocol.                                                               |
| L2        | Improves a project metric                    | Paired evaluation + confidence interval + transparent overhead accounting.                                                          |
| L3        | Improves end-to-end solver behavior robustly | Multiple seeds, benchmark families, paired statistics, size/structure shift, ablations.                                             |
| L4        | Introduces a new mechanism                   | L0-L3 plus literature separation demonstrating that the mechanism itself is not already established in substantially the same form. |
| L5        | Generalizable methodological contribution    | L4 plus replication across solver settings/problem families and a clean explanation of why the mechanism works.                     |

# 12. Closest-Prior-Art Comparison Matrix

| **Dimension**               | **Graph-Q-SAT**            | **NeuroCore**                           | **NeuroComb**                                                      | **RLAF 2026**                                      | **Project M5**                                                         |
|-----------------------------|----------------------------|-----------------------------------------|--------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------------|
| Neural query                | During search              | Periodic                                | One-shot before solve; reuse periodically                          | One-shot before solve                              | Adaptive refresh during search                                         |
| Learned output              | Q-values/actions           | Unsat-core likelihood/activity guidance | Static variable/clause importance                                  | Variable weights + polarity                        | Learned branching guidance; exact output frozen in Docs 05-07          |
| Classical heuristic         | MiniSat integration        | VSIDS activity guidance                 | VSIDS hybrid                                                       | Existing score function                            | VSIDS + controller                                                     |
| RL                          | DQN/value learning         | Supervised prediction                   | GNN classification + bandit selection                              | Policy-gradient RL using solver cost               | RL for policy components as specified; controller cost term is central |
| Adaptation during solve     | Per-decision neural action | Periodic reset                          | Periodic use with bandit                                           | No fresh neural pass during solve                  | Yes: decide refresh/reuse/fallback                                     |
| Explicit refresh cost model | No                         | Not the central contribution            | Resource practicality, but not same cost-aware refresh formulation | Solver computational cost used for training reward | Yes: refresh benefit explicitly compared with computation cost         |
| Main project status         | Reproduce                  | Prior art                               | Prior art / control                                                | Current direct prior art                           | Novelty candidate: narrower trust-aware empirical cost-calibrated refresh mechanism      |

# 13. Final Go / No-Go Criteria for Novelty

- GO: the final literature sweep finds no substantially equivalent adaptive refresh/reuse policy with an explicit benefit-vs-neural-cost decision mechanism in the target SAT/CDCL setting, or the project can clearly demonstrate a materially different mechanism.

- GO WITH CAUTION: related adaptive solver-selection or neural scheduling ideas exist, but M5 has a clearly different SAT-specific mechanism and the claim is narrowed accordingly.

- NO-GO: a paper or implementation is found that already performs materially the same adaptive neural refresh/reuse decision in SAT/CDCL. In this case M5 must be reclassified as an empirical extension/replication, and the final report should emphasize benchmarking and findings rather than methodological novelty.

# 14. Version-Controlled Literature Checklist

| **Checkpoint**                    | **Owner**             | **Status at v1.0**    | **Required before final report**                 |
|-----------------------------------|-----------------------|-----------------------|--------------------------------------------------|
| Core prior-art matrix             | Research lead         | Complete              | Refresh once before final submission             |
| Graph-Q-SAT reproduction boundary | Baseline lead         | Complete              | Confirm exact code/version and deviations        |
| One-shot prior-art boundary       | Research lead         | Updated for ICLR 2026 | Mandatory                                        |
| M5 novelty mechanism definition   | Project lead          | Candidate only        | Freeze in Research Decision Log                  |
| Closest-paper overlap table       | Project lead          | Initial               | Expand with final search results                 |
| Claim wording audit               | Project lead          | Pending final results | Must be completed before final report            |
| Patent/legal prior-art search     | Outside project scope | Not performed         | Do not describe this document as legal clearance |

# 15. References Used for This Audit

**\[1\] Selsam, D., Lamm, M., Bünz, B., Liang, P., de Moura, L., & Dill, D. “Learning a SAT Solver from Single-Bit Supervision.”.** ICLR 2019. https://arxiv.org/abs/1802.03685

**\[2\] Selsam, D. & Bjørner, N. “Guiding High-Performance SAT Solvers with Unsat-Core Predictions.”.** SAT 2019, pp. 336–353. https://arxiv.org/abs/1903.04671

**\[3\] Kurin, V., Godil, S., Whiteson, S., & Catanzaro, B. “Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?”.** NeurIPS 2020. https://proceedings.nips.cc/paper_files/paper/2020/hash/6d70cb65d15211726dcce4c0e971e21c-Abstract.html

**\[4\] Wang, W., Hu, Y., Tiwari, M., Khurshid, S., McMillan, K. L., & Miikkulainen, R. “NeuroComb: Improving SAT Solving with Graph Neural Networks.”.** arXiv:2110.14053. https://dblp.org/rec/journals/corr/abs-2110-14053.html

**\[5\] Tönshoff, J. & Grohe, M. “Learning from Algorithm Feedback: One-Shot SAT Solver Guidance with GNNs.”.** ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/a04c50c0bd6937760a3b89ee10010c96-Abstract-Conference.html

Source interpretation note: the audit uses the cited papers to determine what is already established. Where this document labels M5 as a “novelty candidate,” that label is intentionally provisional; it is not a claim that no prior paper exists.

---

# V2.0 AUTHORITATIVE SYSTEM UPGRADE — TRUST → REFRESH → BRANCH

> **Status:** Authoritative amendment for the upgraded Project 14 system.  
> **Purpose:** Synchronize this document with the final controller architecture while preserving all valid content above.  
> **Precedence:** This amendment supersedes any earlier statement in this document that conflicts with the definitions below. It does **not** erase historical/reproduction material; such material remains valid when explicitly labeled as historical, baseline, or superseded.

## V2.0 design objective

The project is no longer based on an arbitrary seven-parameter weighted-sum "value-of-computation" gate.

The final research candidate is:

> **An uncertainty- and OOD-aware, empirically calibrated, cost-aware neural refresh controller for CDCL SAT solving.**

The controller separates three questions that were previously conflated:

1. **Trust:** Is the neural guidance sufficiently trustworthy in the present instance/search state?
2. **Refresh:** If neural guidance is trustworthy, is obtaining fresh neural guidance expected to repay its computational cost?
3. **Branch:** Given the current trusted guidance, should the solver branch using the neural policy or the classical VSIDS policy?

This separation is mandatory. Do not collapse all signals into one hand-weighted score.

## V2.0 canonical decision pipeline

```text
CDCL state
    |
    +--> graph embedding --------------------> OOD detector
    |
    +--> current neural scores --------------> uncertainty diagnostics
    |
    +--> solver telemetry + cache metadata --> refresh-benefit features
    |
    +-----------------------------------------+
                                              |
                                        TRUST GATE
                                     /             \
                                  FAIL              PASS
                                   |                 |
                                 VSIDS        REFRESH ESTIMATOR
                                                   |
                                       +-----------+-----------+
                                       |                       |
                                  predicted VOI > 0       predicted VOI <= 0
                                       |                       |
                                    REFRESH                  REUSE
                                       |                       |
                                       +-----------+-----------+
                                                   |
                                            TRUSTED GNN GUIDANCE
                                                   |
                                         NEURAL / VSIDS / optional
                                            secondary BLEND
```

## V2.0 scientific rule

Do **not** describe a hand-tuned weighted sum of confidence, disagreement, staleness, progress, and regime change as a true value-of-computation model.

A genuine cost-aware component in the upgraded system must be trained or calibrated against **observed counterfactual outcomes** and must estimate expected downstream benefit in comparable units to refresh cost.

The phrase "value-of-computation" is reserved for that empirically grounded estimator.


# V2.0 Novelty Boundary Amendment

The novelty candidate must now be worded precisely:

> **A trust-gated adaptive neural refresh policy that combines representation-level OOD rejection and decision-level uncertainty diagnostics with an empirically calibrated estimate of the future benefit of refreshing neural branching guidance relative to its measured computation cost.**

This remains a **novelty candidate**, not an established novelty claim.

The audit must distinguish:

- learned branching itself — prior art;
- Graph-Q-SAT — prior art/reproduction;
- one-shot neural guidance — prior art;
- periodic neural guidance / hybrid neural-static guidance — prior art;
- uncertainty-based fallback — potentially overlapping literature component;
- embedding-based OOD rejection — general ML technique, not by itself novel;
- learned refresh-benefit estimation — the project-specific mechanism to audit;
- the **specific integration and control policy** — the principal candidate to compare against prior work.

The audit must search specifically for combinations of:

```text
SAT + neural branching + adaptive refresh
SAT + neural guidance + uncertainty gating
SAT + OOD detection + branching heuristic
SAT + learned refresh scheduling
SAT solver + value of computation + neural heuristic
CDCL + neural inference budget
adaptive neural heuristic + solver
```

Do not call the combined system novel merely because no exact title appears in the first search results.

## Required final novelty language

Until the audit and experiments pass:

> "We investigate a candidate adaptive controller combining OOD/uncertainty safeguards with empirically calibrated neural-refresh decisions."

Only after explicit evidence may the wording become:

> "To the best of our literature audit, we found no prior work matching the exact combination ..."

Even then, qualify the scope and date of the search.


## V2.0 Cross-Document Consistency Check

This amendment is synchronized with Documents 01–20. The implementation agent must treat the newest V2.0 definitions as authoritative within this document and record any residual contradiction as a Decision Log item rather than silently choosing a meaning.
