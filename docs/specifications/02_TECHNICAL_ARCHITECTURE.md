# ⚠️ V2.0 AUTHORITATIVE UPGRADE NOTICE — READ BEFORE USING THIS DOCUMENT

**This document has been upgraded to the Project 14 V2.0 Trust → Refresh → Branch architecture.**  
The V2.0 controller is the system of record for the final research candidate. Earlier formulations remain only as historical/baseline material unless explicitly re-used as an ablation. In particular, the former hand-weighted seven-parameter refresh gate is **not** the primary method and must not be described as a genuine value-of-computation estimator.

**Canonical final controller:**  
`Trust (OOD + uncertainty + search-turbulence) → Refresh (empirically learned future-benefit estimate vs measured refresh cost) → Branch (neural / VSIDS; optional blending only as a secondary ablation).`

**Scientific rule:** No expected winner, robustness claim, novelty claim, or positive performance outcome is assumed. All such statements must be established by the declared experiments.

**DAC 203 \| ARTIFICIAL INTELLIGENCE**

**PROJECT 14**

Technical Architecture & Codebase Specification

*Neural Branching Heuristics for SAT and Constraint Satisfaction*

| **Field**        | **Value**                                                                                 |
|------------------|-------------------------------------------------------------------------------------------|
| Document ID      | P14-D02                                                                                   |
| Version          | 2.0                                                                                       |
| Status           | Implementation specification — derived from Master PRD v2.0                               |
| Primary audience | AI coding agent, project developers, experiment engineers, reviewers                      |
| Authority        | Subordinate to Master PRD; implementation authority for software structure and interfaces |

**Purpose:** Translate the Master PRD from a research plan into a concrete software architecture, module boundary specification, interface contract, lifecycle, repository layout, and implementation order. This document does not freeze scientific choices that belong to later dedicated specifications.

**Important:** Where the Master PRD intentionally leaves an implementation choice open, this document labels the choice as OPEN rather than silently deciding it. The later method-specific specifications must resolve those items before implementation is considered frozen.

# 1. Scope and Architectural Principles

This document specifies the software system required to execute the Project 14 method ladder: classical VSIDS CDCL, Graph-Q-SAT-style learned branching, one-shot neural guidance, a basic neural/classical controller, richer importance-polarity-confidence guidance, and adaptive cost-aware neural refresh. The Master PRD remains the source of truth for research questions, hypotheses, evaluation philosophy, and novelty positioning.

**Architectural objective:** Preserve a complete symbolic CDCL solver while allowing the branching decision mechanism to be replaced, wrapped, or augmented through a stable interface.

| **Principle**                      | **Required behavior**                                                                                                                               |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Correctness first                  | No neural component may bypass unit propagation, conflict analysis, clause learning, backjumping, restarts, or termination.                         |
| Stable solver seam                 | All learned policies enter through one branch-policy interface; solver internals remain unchanged except for explicitly isolated integration hooks. |
| Reproduce before innovate          | M0 and M1 must be independently runnable before adaptive mechanisms are integrated.                                                                 |
| Measurement separation             | Graph construction, neural inference, solver work, controller work, and end-to-end wall time are measured separately.                               |
| Deterministic experiment structure | Manifest, seed, solver configuration, model checkpoint, code revision, and environment are recorded for every run.                                  |
| No hidden research choices         | Scientific choices left open by the PRD become explicit decision records before they affect headline results.                                       |
| Replaceability                     | GNN, DQN, controller, graph encoder, and benchmark modules must be independently testable and replaceable.                                          |

# 2. System Context

DIMACS CNF  
│  
▼  
Benchmark Manifest / Experiment Runner  
│  
├──────── Classical CDCL / VSIDS  
│  
├──────── Graph Extraction ──► GNN ──► Neural Guidance  
│ │  
│ └──► confidence / polarity / importance  
│  
└──────── Solver Telemetry ───────► Controller / Adaptive Refresh  
│  
▼  
BranchAction  
│  
▼  
CDCL Solver  
│  
trace + metrics + result  
│  
┌────────────┴────────────┐  
▼ ▼  
Structured Logs Evaluation

The architecture follows the golden loop in the Master PRD: extract state, obtain classical guidance, optionally obtain cached or fresh neural guidance, let the controller choose the action, branch, propagate, learn from conflicts, backjump/restart, and log. The branch-policy interface is the central decoupling seam.

# 3. Top-Level Components

| **Component**      | **Responsibility**                                                         | **Must expose**                                        |
|--------------------|----------------------------------------------------------------------------|--------------------------------------------------------|
| Benchmark Manifest | Defines exact instances, split, metadata, timeout and experiment identity. | Manifest loader + immutable run plan                   |
| Experiment Runner  | Orchestrates runs, seeds, timeouts, logging and artifact collection.       | Run command, config validation, run directory creation |
| Solver Adapter     | Builds and controls the MiniSat-style CDCL process.                        | solve(), telemetry(), branch-policy registration       |
| Branching Policy   | Produces a legal branch action.                                            | choose_action(SolverState) -\> BranchAction            |
| State Extractor    | Converts solver state into graph/neural/controller inputs.                 | extract_graph(), extract_telemetry_features()          |
| Graph Builder      | Constructs variable-clause graph and encodes literal polarity.             | build(GraphInput) -\> GraphBatch / GraphState          |
| Neural Model       | Maps graph input to learned guidance outputs.                              | predict(GraphState) -\> GuidanceOutput                 |
| RL Environment     | Presents solver interactions as state/action/reward transitions.           | reset(), step(action), terminal(), info()              |
| RL Learner         | Trains DQN/value policy with replay and target network.                    | train(), evaluate(), save_checkpoint()                 |
| Controller         | Selects, blends, reuses, or refreshes neural guidance.                     | decide(ControllerState) -\> ControllerAction           |
| Instrumentation    | Captures per-decision and aggregate metrics.                               | event logger + run summary                             |
| Evaluation         | Aggregates results, computes paired comparisons and plots.                 | evaluate(manifest, results) -\> report artifacts       |
| Visualization      | Shows solver trajectories, race plots, overhead and controller behavior.   | plot scripts / dashboard artifacts                     |

# 4. Canonical Repository Layout

project14/  
├── README.md  
├── LICENSE  
├── pyproject.toml  
├── CMakeLists.txt  
├── environment/  
│ ├── environment.yml  
│ ├── requirements.txt  
│ ├── Dockerfile  
│ └── VERSIONS.md  
├── docs/  
│ ├── 01_Master_PRD.md  
│ ├── 02_Technical_Architecture.md  
│ ├── 03_GraphQSat_Reproduction.md  
│ ├── 04_MiniSat_Integration.md  
│ ├── 05_GNN_Model.md  
│ ├── 06_RL_Training.md  
│ ├── 07_Adaptive_Controller.md  
│ ├── 08_Datasets_Benchmarks.md  
│ ├── 09_Experiment_Protocol.md  
│ ├── 10_Reproducibility.md  
│ ├── 11_Logging_Schema.md  
│ ├── 12_Testing.md  
│ ├── 13_Statistics.md  
│ ├── 14_Literature_Audit.md  
│ ├── 15_Final_Report.md  
│ ├── 16_Decision_Log.md  
│ └── 17_Definitions_of_Done.md  
├── third_party/  
│ └── minisat/  
├── src/  
│ ├── solver/  
│ ├── graph/  
│ ├── neural/  
│ ├── rl/  
│ ├── controller/  
│ ├── experiments/  
│ ├── evaluation/  
│ └── common/  
├── tests/  
│ ├── unit/  
│ ├── integration/  
│ ├── property/  
│ └── golden/  
├── configs/  
│ ├── solver/  
│ ├── model/  
│ ├── training/  
│ ├── controller/  
│ └── experiments/  
├── data/  
│ ├── toy/  
│ ├── random_3sat/  
│ ├── structured/  
│ ├── public_subset/  
│ └── manifests/  
├── scripts/  
│ ├── build_solver.\*  
│ ├── generate_data.\*  
│ ├── train.\*  
│ ├── evaluate.\*  
│ └── reproduce\_\*.\*  
├── runs/  
└── artifacts/  
├── checkpoints/  
├── logs/  
├── tables/  
├── figures/  
└── reports/

**Repository rule:** The repository tree is a canonical proposal for the project. Exact package/module names may be adjusted during implementation, but responsibilities and interfaces must remain stable unless a Decision Log entry records the change.

# 5. Core Data Contracts

| **Object**       | **Minimum fields**                                                                                                                           | **Invariant**                                                                          |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| SolverState      | decision_level, assigned_variables, conflicts, propagations, restarts, legal_variables, elapsed_time                                         | Represents one branch decision point; state must be internally consistent with solver. |
| BranchAction     | variable_id, polarity, source, score metadata                                                                                                | variable_id must be legal/unassigned; polarity must be valid.                          |
| GraphState       | node_features, edge_index, edge_features, variable_mask, clause_mask, mapping metadata                                                       | Variable/clause IDs must map reversibly to solver IDs.                                 |
| GuidanceOutput   | variable_scores, optional polarity_scores, optional confidence, model_id                                                                     | Assigned variables must never become legal actions.                                    |
| ControllerState  | neural guidance summary + solver telemetry + cost features                                                                                   | Only features approved by controller spec are allowed.                                 |
| ControllerAction | mode, selected_action or score blend, refresh_flag, reason_code                                                                              | Action must resolve to a legal BranchAction before solver call.                        |
| RunConfig        | solver, method, dataset manifest, seed, timeout, hardware tag, checkpoint                                                                    | Immutable after run start.                                                             |
| RunResult        | SAT/UNSAT/UNKNOWN/timeout, runtime, decisions, conflicts, propagations, restarts, memory, neural calls                                       | Must be traceable to run_id and manifest entry.                                        |
| PerDecisionEvent | run_id, instance_id, decision_id, chosen_var, polarity, neural score/confidence, controller action, solver telemetry, graph/inference timing | One record corresponds to a branch decision event.                                     |

# 6. Interface Contracts

## 6.1 Branching Policy Interface

class BranchingPolicy:  
def reset(self, solver_context): ...  
def choose_action(self, state: SolverState) -\> BranchAction: ...  
def observe(self, event: DecisionEvent): ...  
def close(self): ...

**Requirement:** The policy interface must support M0-M5 without solver rewrites. VSIDS may be an adapter over the native solver heuristic; neural methods implement the same external contract.

## 6.2 Solver Adapter Interface

class SolverAdapter:  
def load_instance(self, cnf_path: str): ...  
def set_branch_policy(self, policy: BranchingPolicy): ...  
def solve(self) -\> RunResult: ...  
def get_state(self) -\> SolverState: ...  
def get_telemetry(self) -\> SolverTelemetry: ...  
def validate_action(self, action: BranchAction) -\> bool: ...  
def close(self): ...

## 6.3 Graph Builder Interface

class GraphBuilder:  
def build(self, state: SolverState) -\> GraphState: ...  
def validate(self, graph: GraphState) -\> ValidationReport: ...

## 6.4 Neural Model Interface

class NeuralGuidanceModel:  
def predict(self, graph: GraphState) -\> GuidanceOutput: ...  
def load_checkpoint(self, path: str): ...  
def save_checkpoint(self, path: str): ...

## 6.5 Controller Interface

class GuidanceController:  
def reset(self, run_context): ...  
def decide(self, state: ControllerState) -\> ControllerAction: ...  
def record_outcome(self, outcome): ...

# 7. Golden Runtime Lifecycle

1.  Experiment Runner loads immutable run configuration and benchmark manifest entry.

2.  Solver Adapter loads the CNF instance and installs the selected BranchingPolicy.

3.  Solver performs propagation and reaches a decision point.

4.  State Extractor creates the approved SolverState / GraphState / ControllerState views.

5.  Classical guidance is computed or read from the solver.

6.  Neural guidance is computed, reused, or skipped according to the selected method/controller.

7.  Controller resolves the final BranchAction or score blend.

8.  Solver validates the action, branches, propagates, and continues CDCL.

9.  On conflict, solver performs conflict analysis, learns a clause, and backjumps/restarts according to locked solver settings.

10. Instrumentation writes the decision event and timing breakdown.

11. At termination, RunResult and immutable run metadata are persisted.

12. Evaluation consumes only persisted artifacts; it must not alter the run.

Decision point  
│  
├── legal vars / solver telemetry  
│  
├── VSIDS signal  
│  
├── \[optional\] graph build  
│ │  
│ └── \[optional\] GNN inference  
│  
├── Controller / fixed policy  
│  
└── BranchAction  
│  
▼  
validate  
│  
▼  
branch → propagate → conflict?  
│  
yes ─────┘  
▼  
learn/backjump/restart

# 8. Method-to-Component Mapping

| **Method**                  | **Required modules**                                       | **Neural call behavior**                                                                            | **Research role**              |
|-----------------------------|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------|
| M0 VSIDS-only               | Solver Adapter + native/classical policy + instrumentation | 0                                                                                                   | Gold reference                 |
| M1 Graph-Q-SAT reproduction | M0 + Graph Builder + GNN + DQN + RL Environment            | Paper-faithful schedule; exact call policy resolved in reproduction spec                            | Primary reproduction baseline  |
| M2 One-shot neural prior    | M0 + Graph Builder + GNN + prior adapter                   | One expensive evaluation at a defined checkpoint; remainder reuses prior                            | Overhead-amortization baseline |
| M3 Basic controller         | M0 + neural brancher + simple Controller                   | Rule-based selection by fixed budget/conflict count/etc.; exact variant resolved in controller spec | Professor-advised baseline     |
| M4 Rich guidance            | M3 + importance/polarity/confidence outputs                | Same base call budget as comparison mode                                                            | Ablation of output signals     |
| M5 Adaptive refresh         | M4 + cost/benefit estimator + adaptive Controller          | Selective refresh triggered by state/empirically calibrated refresh-benefit logic                                     | Primary candidate contribution |

# 9. Configuration Architecture

configs/  
├── solver/  
│ ├── minisat_reference.yaml  
│ └── integration.yaml  
├── model/  
│ ├── gnn_base.yaml  
│ └── decoder.yaml  
├── training/  
│ ├── dqn_default.yaml  
│ └── seeds.yaml  
├── controller/  
│ ├── fixed_budget.yaml  
│ ├── confidence_gate.yaml  
│ └── adaptive_refresh.yaml  
└── experiments/  
├── m0_vsids.yaml  
├── m1_graph_qsat.yaml  
├── m2_one_shot.yaml  
├── m3_controller.yaml  
├── m4_rich_guidance.yaml  
└── m5_adaptive.yaml

| **Configuration rule**         | **Requirement**                                                                                                     |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Single source of configuration | All experiment-defining parameters must come from versioned config/manifest files rather than hard-coded constants. |
| Inheritance                    | Shared defaults may be inherited, but the resolved configuration must be exported into each run directory.          |
| Immutability                   | Resolved config is copied at run start and never edited in-place.                                                   |
| Validation                     | Unknown keys, wrong types, illegal ranges, and missing required fields must fail before a solver run begins.        |
| Provenance                     | Run metadata records config hash, checkpoint hash, code revision and environment identifier.                        |

# 10. Run Directory Contract

runs/\<timestamp\>\_\<experiment_id\>\_\<instance_id\>\_\<seed\>/  
├── config_resolved.yaml  
├── manifest_entry.json  
├── environment.json  
├── git_revision.txt  
├── solver_config.json  
├── model_checkpoint.txt  
├── run_meta.json  
├── decision_events.jsonl  
├── summary.json  
├── stdout.log  
├── stderr.log  
└── status.json

| **Artifact**          | **Purpose**                                                   |
|-----------------------|---------------------------------------------------------------|
| config_resolved.yaml  | Exact parameter values used by the run.                       |
| manifest_entry.json   | Exact CNF instance and metadata identity.                     |
| environment.json      | Python/CUDA/compiler/OS/hardware/version capture.             |
| run_meta.json         | run_id, start time, method, seed, timeout, hardware tag.      |
| decision_events.jsonl | Per-decision instrumentation and neural/controller telemetry. |
| summary.json          | Final solver result and aggregate metrics.                    |
| status.json           | Lifecycle state: started/completed/timeout/error.             |

# 11. Error Handling and Safety Rules

| **Failure**                 | **Required response**                                                                                                                                        |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Illegal branch action       | Reject before solver mutation; log policy error; fail closed to a documented safe fallback only if the experiment specification explicitly permits fallback. |
| Graph construction failure  | Abort the affected neural run; do not silently fabricate graph tensors.                                                                                      |
| NaN/Inf neural output       | Mark inference invalid; record the event; follow method-specific fallback rule. Never treat NaN as a score.                                                  |
| Checkpoint incompatibility  | Fail before evaluation; report model/config schema mismatch.                                                                                                 |
| Solver correctness mismatch | Stop headline evaluation for that method until the mismatch is diagnosed.                                                                                    |
| Timeout                     | Record timeout distinctly from UNSAT/SAT; do not convert timeout into failure/success.                                                                       |
| Instrumentation failure     | If mandatory timing/identity fields cannot be logged, mark run invalid for final analysis rather than silently continuing.                                   |
| Test-set tuning attempt     | Block by protocol: controller thresholds and model-selection criteria must be frozen before test evaluation.                                                 |

# 12. Testing Architecture

| **Test layer**        | **Examples**                                                                         | **Minimum gate**                                                   |
|-----------------------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Unit                  | graph indexing, literal polarity, masking, config validation, cost calculation       | All unit tests pass                                                |
| Property / invariance | variable renaming, clause renaming, action legality, permutation-preserving rankings | No invariant violation on defined randomized suite                 |
| Golden solver         | known SAT/UNSAT CNFs, baseline result match                                          | 100% expected-status agreement for golden set                      |
| Integration           | dummy policy, VSIDS mirror adapter, neural policy, controller                        | End-to-end solve completes without CDCL corruption                 |
| Reproducibility       | same seed/config repeated                                                            | Identical or explicitly documented deterministic-close outputs     |
| Performance sanity    | graph build and inference timings on fixed fixtures                                  | No unexpected order-of-magnitude regression without recorded cause |

# 13. Implementation Order

1\. Freeze repository and environment metadata; add build and test scaffolding.

2\. Import/lock the exact MiniSat source/version required by the solver specification.

3\. Build untouched classical baseline and golden correctness suite.

4\. Implement solver branching-policy hook with a dummy first-legal-variable policy.

5\. Implement a VSIDS-mirror adapter and verify behavioral equivalence on fixed fixtures.

6\. Implement state extraction and graph construction independently of neural inference.

7\. Implement permutation, legality, terminal-state, and shape tests for graph/state conversion.

8\. Implement the GNN inference API and standalone graph-to-guidance tests.

9\. Implement the RL environment and DQN training loop using the reproduction specification.

10\. Reproduce M1 under a locked configuration before adding M2-M5.

11\. Implement M2, then M3, then M4, each under the common Experiment Runner.

12\. Implement M5 only after baseline overhead and controller telemetry are available.

13\. Freeze final test manifest and thresholds; run headline matrix and ablations.

# 14. Open Decisions That Must Be Resolved Before Headline Implementation

| **ID**  | **Open item**                                                                            | **Why it matters**                         | **Owner document**                                   |
|---------|------------------------------------------------------------------------------------------|--------------------------------------------|------------------------------------------------------|
| ARCH-01 | Exact MiniSat version/commit and any allowed source modifications                        | Behavior and API determine reproducibility | 04 MiniSat/CDCL Integration                          |
| ARCH-02 | Exact graph tensor format, edge directionality and polarity encoding                     | Changes model input semantics              | 03 Graph-Q-SAT Reproduction + 05 GNN Model           |
| ARCH-03 | Exact GNN architecture and dimensions                                                    | Determines model capacity and timing       | 05 GNN Model                                         |
| ARCH-04 | Exact action encoding: variable-only plus polarity vs separate variable/polarity actions | Changes DQN action space and masking       | 03 Graph-Q-SAT Reproduction + 06 RL                  |
| ARCH-05 | Exact reward and any later cost penalty                                                  | Changes learned behavior                   | 03 Graph-Q-SAT Reproduction + 06 RL                  |
| ARCH-06 | Exact one-shot checkpoint definition and prior injection mechanism                       | Determines M2 scientific comparison        | 03 Graph-Q-SAT Reproduction + 09 Experiment Protocol |
| ARCH-07 | Exact controller equations/threshold selection process                                   | Defines M3-M5 behavior                     | 07 Adaptive Controller                               |
| ARCH-08 | Exact benchmark sizes, SAT/UNSAT proportions and public instances                        | Defines generalization and fairness        | 08 Dataset & Benchmark                               |
| ARCH-09 | Exact statistical tests and primary endpoints                                            | Prevents cherry-picking                    | 13 Statistical Analysis                              |
| ARCH-10 | Final literature audit boundary and novelty statement                                    | Prevents invalid contribution claim        | 14 Literature Audit                                  |

**Rule:** No open decision may be silently resolved inside source code. A temporary implementation may exist for plumbing tests, but it must be tagged as provisional and must not be used for headline scientific claims.

# 15. AI Coding Agent Contract

The implementation agent must treat this architecture document as a software contract, not as permission to invent research methodology.

| **Agent behavior**       | **Requirement**                                                                                                                                                             |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Read before coding       | Read the Master PRD and all currently available method-specific specifications before changing architecture.                                                                |
| Ask through Decision Log | When a scientific or architectural choice is genuinely unresolved, record the question, alternatives, and provisional implementation rather than silently choosing a value. |
| Build incrementally      | Implement one milestone at a time and run its tests before advancing.                                                                                                       |
| No hidden fallbacks      | Fallback to VSIDS, score blending, or cached guidance must be explicit in config and logs.                                                                                  |
| Preserve provenance      | Every result must retain config, checkpoint, instance, seed and code revision identity.                                                                                     |
| Scientific isolation     | Do not alter evaluation scripts merely to accommodate a preferred result; evaluation code is versioned and read-only during headline runs.                                  |
| Document deviations      | Any deviation from cited paper behavior must be captured in the reproduction matrix and reported in the final report.                                                       |
| Reproducible commands    | Every milestone must expose a command/script that reproduces its key result on the same environment.                                                                        |

# 16. Acceptance Criteria for This Architecture Document

- The complete method ladder can be represented without changing the core CDCL solver contract.

- Every major component has an explicit responsibility and interface.

- Run identity, configuration, logging, and artifact provenance are defined.

- Testing layers cover solver correctness, neural legality, invariance, integration, and reproducibility.

- Open scientific choices are explicitly tracked rather than hidden.

- The repository layout can support all 15 planned project documents plus code/data/artifacts.

- An AI coding agent can implement the plumbing in milestone order without deciding the research methodology on its own.

# 17. Traceability to Master PRD

| **Master PRD area**                       | **Architecture implementation**                                                             |
|-------------------------------------------|---------------------------------------------------------------------------------------------|
| §5 Core Design Principles                 | Sections 1, 11, 15                                                                          |
| §6 System Architecture                    | Sections 2–6                                                                                |
| §7 Progressive Method Ladder              | Section 8                                                                                   |
| §8 Detailed Neural Model Specification    | Boundary defined here; exact model frozen in Document 05                                    |
| §9 RL Training Specification              | Boundary defined here; exact RL procedure frozen in Document 06                             |
| §10 Solver Integration                    | Sections 5–7 and Document 04 dependency                                                     |
| §11 Controller Design                     | Controller interface in Sections 6–8; exact equations in Document 07                        |
| §12 Experimental Matrix                   | Run/config/evaluation scaffolding in Sections 9–10; protocol frozen in Document 09          |
| §13 Metrics and Logging                   | Sections 10 and core data contracts; full schema in Document 11                             |
| §14 Fairness and Statistical Analysis     | Section 15 agent contract; exact analysis in Document 13                                    |
| §15–17 Ablations, Generalization, Novelty | Architecture supports these; exact experiments and claims belong to Documents 09, 13 and 14 |
| §18–20 Risks, Team, Milestones            | Sections 11–13                                                                              |
| §23 Final Acceptance Criteria             | Section 16                                                                                  |

**End of Document 02 — Technical Architecture & Codebase Specification v1.0**

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


# V2.0 Architecture Amendment

## A. New explicit modules

The architecture must expose these components as independently testable modules:

```text
GraphStateBuilder
       |
       v
NeuralGuidanceProvider
       |
       +--> GuidanceOutput
       |
       +--> GraphEmbedding
                |
                v
          OODDetector
                |
SolverTelemetry ---> TrustGate
                         |
              +----------+----------+
              |                     |
             FAIL                   PASS
              |                     |
            VSIDS            RefreshBenefitEstimator
                                    |
                              RefreshDecision
                                    |
                            Refresh / Reuse / Fallback
```

## B. Ownership boundaries

`TrustGate` owns only neural admissibility.  
`RefreshBenefitEstimator` owns prediction of expected downstream benefit.  
`RefreshCostMeasurer` owns measured inference/graph-build cost.  
`NeuralRefreshCache` owns the lifetime and provenance of cached guidance.  
`BranchPolicy` owns the final legal branch action.

No component may silently perform another component's function.

## C. Required architectural invariant

The system must remain usable with `RefreshBenefitEstimator = disabled` so that fixed-K and no-refresh controls can be executed using exactly the same solver adapter and logging stack.

## D. Research/production separation

Counterfactual refresh-benefit data collection is an offline calibration/training workflow. It must not be enabled during final benchmark evaluation unless explicitly declared as an experimental treatment.



## V2.0 Cross-Document Consistency Check

This amendment is synchronized with Documents 01–20. The implementation agent must treat the newest V2.0 definitions as authoritative within this document and record any residual contradiction as a Decision Log item rather than silently choosing a meaning.
