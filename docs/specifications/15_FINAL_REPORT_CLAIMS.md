# ⚠️ V2.0 AUTHORITATIVE UPGRADE NOTICE — READ BEFORE USING THIS DOCUMENT

**This document has been upgraded to the Project 14 V2.0 Trust → Refresh → Branch architecture.**  
The V2.0 controller is the system of record for the final research candidate. Earlier formulations remain only as historical/baseline material unless explicitly re-used as an ablation. In particular, the former hand-weighted seven-parameter refresh gate is **not** the primary method and must not be described as a genuine value-of-computation estimator.

**Canonical final controller:**  
`Trust (OOD + uncertainty + search-turbulence) → Refresh (empirically learned future-benefit estimate vs measured refresh cost) → Branch (neural / VSIDS; optional blending only as a secondary ablation).`

**Scientific rule:** No expected winner, robustness claim, novelty claim, or positive performance outcome is assumed. All such statements must be established by the declared experiments.

DAC 203 \| Project 14

Final Report & Claim Specification

Neural Branching Heuristics for SAT

Document 15 \| v1.0 \| Final reporting and scientific-claims control \| 15 September 2026

Purpose. This document specifies exactly how the completed Project 14 work must be assembled into a final technical report, what evidence is required for each research claim, how results must be framed, and which statements are prohibited unless their evidentiary conditions are satisfied.

Scope. The specification covers the final report narrative, contribution taxonomy, claim-evidence mapping, result tables and figures, interpretation rules, novelty wording, limitations, reproducibility package, appendices, and final acceptance gate. It complements Documents 09, 11, 13, and 14 rather than replacing them.

# 1. Executive Reporting Contract

The final report is an evidence-first research document, not a project diary. Every substantive conclusion must be traceable to a frozen experiment, a defined metric, and an explicitly named comparison. The report must separate three things: (1) what the system implemented, (2) what the experiments measured, and (3) what those measurements justify claiming.

| **Contract item**  | **Required rule**                                                                                                   |
|--------------------|---------------------------------------------------------------------------------------------------------------------|
| Primary narrative  | Lead with the scientific question and evidence, not implementation chronology.                                      |
| Claim discipline   | Every major claim must have a corresponding result table/figure, metric definition, and experiment identifier.      |
| Baseline integrity | M0 VSIDS-only and M1 faithful Graph-Q-SAT reproduction must remain clearly separated from later project extensions. |
| Runtime honesty    | Never equate fewer decisions/conflicts with faster solving without end-to-end wall-clock evidence.                  |
| Novelty honesty    | Only mechanisms supported by Document 14 and the final literature audit may be labeled novel.                       |
| Negative results   | A failed hypothesis is a valid result and must be reported when the experiment was properly executed.               |
| Reproducibility    | All headline results must be regenerated from the stored configuration, logs, manifests, and checkpoints.           |

# 2. Required Final Report Structure

The report should follow the sequence below. Minor ordering changes are allowed only when required by the course template; scientific content and evidence requirements remain mandatory.

| **Section**                   | **Required content**                                                                                                                      |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Abstract                      | Problem, method ladder, primary result, overhead finding, adaptive-refresh finding, and restrained novelty statement.                     |
| 1\. Introduction              | SAT branching problem, motivation, research gap, research questions, contributions, and scope.                                            |
| 2\. Background                | SAT/CNF, DPLL/CDCL, VSIDS, graph representation, learned branching, and solver-neural integration concepts needed to understand the work. |
| 3\. Related Work              | NeuroSAT, Graph-Q-SAT, hybrid/one-shot/neural-guidance literature, and the novelty boundary established by Document 14.                   |
| 4\. System Design             | M0-M5 ladder, solver hook, graph extraction, GNN, RL, controller, cache/refresh logic, and instrumentation.                               |
| 5\. Experimental Methodology  | Datasets, splits, hardware, software, timeouts, seeds, metrics, fairness rules, and statistical analysis.                                 |
| 6\. Results                   | Baseline reproduction, neural overhead, one-shot, controller ladder, adaptive refresh, ablations, and generalization.                     |
| 7\. Discussion                | What improved, what did not, why, overhead trade-offs, failure modes, and practical interpretation.                                       |
| 8\. Novelty & Limitations     | Defensible contribution wording, prior-art boundary, threats to validity, and unresolved questions.                                       |
| 9\. Reproducibility           | Exact commands/configs/checkpoints/commit IDs and instructions to regenerate headline results.                                            |
| 10\. Conclusion & Future Work | Evidence-weighted conclusion; no new claims introduced here.                                                                              |
| Appendices                    | Full configuration tables, extra plots, per-family results, statistical details, and implementation details omitted from the main body.   |

# 3. Contribution Taxonomy

The report must classify project components by evidentiary status. This prevents an implementation detail, a reproduction, and a research contribution from being presented with equal novelty weight.

| **Category**                    | **Meaning in this project**                                                                | **Allowed wording**                            |
|---------------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------|
| Established prior art           | Method already demonstrated by prior literature.                                           | "Prior work demonstrates..."                   |
| Faithful reproduction           | Reimplemented/validated baseline behavior.                                                 | "We reproduce/validate..."                     |
| Engineering integration         | System glue, instrumentation, interfaces, or implementation needed to run the study.       | "We implement/integrate..."                    |
| Empirical finding               | Observed experimental pattern without methodological novelty.                              | "We observe/find that..."                      |
| Project extension               | A deliberate addition beyond the reproduced baseline, but novelty not yet established.     | "We investigate/propose..."                    |
| Supported research contribution | A mechanism shown by the literature audit and experiments to be a defensible contribution. | "We introduce..." only after the novelty gate. |
| Hypothesis/future idea          | Promising but not sufficiently validated or differentiated.                                | "We hypothesize/leave for future work..."      |

# 4. Mandatory Claim-Evidence Matrix

Before final submission, the team must complete a claim-evidence matrix. The matrix itself may be included in the appendix or maintained as a project-control artifact. The following rows are the minimum set.

| **Claim ID** | **Candidate claim**                                 | **Minimum evidence**                                                                        | **Status rule**                                                            |
|--------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| C1           | M0 provides a correct classical reference solver.   | Golden-suite correctness + frozen config + baseline metrics.                                | Must pass before interpreting neural results.                              |
| C2           | M1 is a faithful Graph-Q-SAT reproduction.          | Independent reproduction checks + matching interface semantics + validation statistics.     | If divergence remains, claim only "partial reproduction" and document why. |
| C3           | Neural guidance changes search behavior.            | Paired decisions/conflicts/propagations + solver traces.                                    | Do not infer from runtime alone.                                           |
| C4           | Neural inference introduces measurable overhead.    | Graph-build time, inference time, call count, and end-to-end runtime.                       | Must report separately from search metrics.                                |
| C5           | One-shot guidance has a measurable effect.          | M2 vs matched classical baseline and, where relevant, repeated reuse analysis.              | Effect may be positive, neutral, or negative.                              |
| C6           | Controller logic changes the neural-use pattern.    | Controller-action counts/rates and paired solver outcomes.                                  | Must show both policy behavior and solver consequence.                     |
| C7           | Importance/polarity/uncertainty features add value. | Ablation ladder with controlled comparisons.                                                | Exploratory unless predeclared as confirmatory.                            |
| C8           | Trust-aware empirical cost-calibrated refresh improves the trade-off. | M5 vs fixed-refresh/no-refresh and matched-call-budget controls; runtime + quality + calls. | Must demonstrate the trade-off, not just one metric.                       |
| C9           | The method generalizes.                             | Size-shift and structural-shift/OOD results held out from tuning.                           | Use "generalizes to tested distributions," not universal wording.          |
| C10          | The project contains a novel mechanism.             | Positive literature audit + exact mechanism description + distinguishing evidence.          | Use "novel" only after Document 14 gate is satisfied.                      |

# 5. Claim Strength Ladder

Use the lowest claim strength that is fully supported. Stronger language requires stronger evidence and tighter scope.

| **Level**                     | **Example language**                                                                                            | **Evidence threshold**                                                              |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| L0 Description                | "The system implements..."                                                                                      | Implementation artifact exists and passes tests.                                    |
| L1 Observation                | "We observe that..."                                                                                            | Measured result with traceable experiment.                                          |
| L2 Comparative finding        | "Method A solved more instances than Method B under the tested protocol."                                       | Paired comparison, fixed protocol, uncertainty reported.                            |
| L3 Conditional conclusion     | "Under the tested random-3-SAT size-shift setting, M5 reduced median capped runtime relative to fixed refresh." | Predefined scope + statistical/practical evidence + controls.                       |
| L4 Generalization             | "The approach transfers across the tested benchmark families."                                                  | Independent structural and/or size-shift evaluation not used for tuning.            |
| L5 Novel contribution         | "We introduce a cost-aware neural refresh mechanism..."                                                         | Mechanism-level differentiation + literature audit + controlled empirical evidence. |
| L6 Universal/field-wide claim | Avoid unless formally justified.                                                                                | Project scope does not support broad universal statements.                          |

# 6. Required Abstract Specification

The abstract must be self-contained and contain exactly five informational blocks in compact prose: problem, baseline/method ladder, main measured result, overhead/adaptive finding, and restrained contribution statement.

1\. State the SAT branching problem and why learned guidance can be expensive inside CDCL.

2\. Name the baseline progression: VSIDS -\> faithful Graph-Q-SAT -\> one-shot/controller variants -\> adaptive refresh.

3\. Report one or two headline measured outcomes using actual units and benchmark scope.

4\. State the computational-overhead finding and whether adaptive refresh helped, hurt, or produced a trade-off.

5\. End with a scope-limited contribution statement; do not claim universal superiority or novelty unless the final gate passes.

The abstract must not introduce a result that is absent from the Results section or rely on a metric that is not defined in the Experimental Methodology section.

# 7. Introduction Specification

## 7.1 Problem statement

Explain that CDCL performance depends strongly on branching choices and that neural policies may provide learned search information but incur computational overhead. Define the project as a study of when learned guidance is worth paying for, rather than as a claim that neural branching is automatically superior.

## 7.2 Research questions

List the PRD research questions in compact form, preserving their intended meaning: learned branching effectiveness, where neural guidance helps, overhead, one-shot guidance, basic controller behavior, richer guidance signals, cost-aware refresh, and generalization.

## 7.3 Contributions paragraph

Use a numbered contribution paragraph, with each contribution explicitly linked to an experiment family. A recommended structure is: (1) validated baseline infrastructure, (2) faithful neural baseline, (3) measured neural overhead characterization, (4) controlled controller/feature studies, and (5) adaptive-refresh study. The final item becomes a novelty claim only if Document 14 authorizes that wording.

# 8. Related Work & Novelty Wording Rules

The related-work section must distinguish what the project reproduces from what it changes. The final paragraph of the section must explicitly state the current novelty boundary and cite the literature audit maintained with Document 14.

| **Topic**                  | **Do not claim**                                         | **Permitted framing**                                                                                                    |
|----------------------------|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| GNN branching in SAT       | "Using a GNN for SAT branching is novel."                | "We reproduce/investigate learned GNN branching."                                                                        |
| Periodic neural use        | "Periodic neural inference is our new idea."             | "We evaluate when periodic intervention is worthwhile."                                                                  |
| One-shot guidance          | "One-shot GNN guidance is novel."                        | "We study one-shot guidance as a controlled project stage."                                                              |
| Polarity + learned weights | "Joint variable/polarity guidance is novel."             | "We include polarity as an experimental extension."                                                                      |
| Solver-cost awareness      | "Using compute cost in training is unique to this work." | "We explicitly measure computation cost in our refresh decision."                                                        |
| Adaptive refresh           | "Our refresh rule is the first ever."                    | Only make a literature-supported novelty claim if the broader audit confirms it; otherwise call it a proposed mechanism. |

# 9. Experimental Results Presentation Contract

The Results section must follow the same M0-M5 ladder as the PRD unless the frozen Experiment Protocol records a justified deviation.

## 9.1 Result block template

1\. Question: state the exact hypothesis or research question being tested.

2\. Setup: identify dataset split, methods compared, timeout, hardware, seed policy, and configuration ID.

3\. Primary endpoint: name the predeclared metric.

4\. Secondary endpoints: decisions, conflicts, propagations, restarts, memory, neural calls, graph-build time, inference time, etc.

5\. Result: give effect size, uncertainty, and practical interpretation.

6\. Interpretation: state what the result does and does not establish.

7\. Failure/limitation note: include relevant anomalies, timeouts, or benchmark-specific behavior.

## 9.2 Minimum headline tables

| **Table**                 | **Required contents**                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------|
| T1 Baseline correctness   | M0 golden-suite pass/fail, solver settings, key sanity metrics.                               |
| T2 Reproduction           | M1 vs reference/reproduction target, with deviations explicitly recorded.                     |
| T3 Main solver comparison | M0-M5 headline metrics by benchmark family and SAT/UNSAT.                                     |
| T4 Overhead decomposition | Graph construction, inference, neural calls, total runtime, and search-work metrics.          |
| T5 Adaptive refresh       | Fixed-K/no-refresh/M5, neural-call counts, runtime, solved fraction, and controller behavior. |
| T6 Ablations              | Importance/polarity/uncertainty, controller terms, cost term, and refresh policy ablations.   |
| T7 Generalization         | Size-shift and structural-shift results on held-out data.                                     |
| T8 Statistical summary    | Effect sizes, confidence intervals, and multiplicity-adjusted confirmatory tests.             |

## 9.3 Minimum headline figures

| **Figure**                          | **Purpose**                                                                    |
|-------------------------------------|--------------------------------------------------------------------------------|
| F1 Method ladder overview           | One diagram showing M0-M5 and the exact intervention introduced at each stage. |
| F2 Runtime distribution             | Per-instance paired runtime visualization; avoid mean-only bars.               |
| F3 Overhead decomposition           | Search work vs graph-build/inference overhead.                                 |
| F4 Neural call vs runtime trade-off | Show the empirical cost/benefit frontier for refresh policies.                 |
| F5 Generalization                   | Performance across in-distribution, size-shift, and structural-shift settings. |
| F6 Controller behavior              | Refresh/reuse/fallback frequencies and representative traces.                  |

# 10. Statistical Reporting Rules

The statistical analysis must follow Document 13. The final report must never select a test after seeing which method happens to win. Any analysis added after the initial analysis lock is exploratory and labeled as such.

| **Item**               | **Required reporting**                                                                                                            |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Solved fraction        | Absolute difference in percentage points plus uncertainty; paired outcomes should be preserved.                                   |
| Runtime                | Median and distributional summaries; report timeout/censoring policy and capped-time or survival-aware analysis where applicable. |
| Heavy tails            | Do not rely on arithmetic mean alone for runtime.                                                                                 |
| Paired design          | Use the same instances, hardware, timeout, and frozen solver settings across methods.                                             |
| Bootstrap              | Use paired instance-level resampling for confidence intervals when specified in Document 13.                                      |
| Seeds                  | Treat seeds as a stochastic-method dimension, not as extra independent instances.                                                 |
| Multiplicity           | Apply the predeclared correction to the confirmatory comparison family; label other tests exploratory.                            |
| Practical significance | Report effect size and whether it meets the project’s predeclared materiality threshold, not just p-values.                       |

# 11. Runtime and Timeout Presentation Rules

SAT runtime distributions are often dominated by difficult instances and timeouts. The report must therefore show both success probability and solving time, rather than hiding timeout effects behind averages.

- Always state the exact timeout and hardware used for the reported result.

- Report solved fraction separately from runtime among solved instances.

- Where Document 13 specifies censored-time analysis, report the corresponding survival-aware summary or restricted mean time through the timeout horizon.

- When using a capped-time score or penalty score, name the penalty and explain that it is a secondary comparison metric.

- Never discard timed-out instances from the main headline result without reporting how many were removed and why.

# 12. Adaptive Refresh Result Contract

M5 is the project’s most important end-to-end systems experiment. Its evaluation must demonstrate that the refresh policy changes computation use in a way that matters to solver performance. A result is incomplete if it shows only a faster or slower runtime without showing what happened to neural-call frequency and search behavior.

| **Required axis**   | **Question**                                                                                |
|---------------------|---------------------------------------------------------------------------------------------|
| Call economy        | How many neural refreshes were avoided relative to a fixed-refresh policy?                  |
| Search quality      | Did avoided calls materially change decisions, conflicts, propagations, or solved fraction? |
| End-to-end time     | Did savings in neural computation outweigh any search degradation?                          |
| Robustness          | Does the policy behave similarly across benchmark families, scales, and SAT/UNSAT classes?  |
| Controller behavior | Which signals triggered refreshes, reuse, or fallback?                                      |
| Ablation            | Does the cost term matter, and does adaptation beat an equally call-budgeted baseline?      |

# 13. Failure Analysis Specification

A high-quality report must include an explicit failure-analysis subsection. The objective is to explain where the method loses and what that implies for deployment or future work.

## 13.1 Required failure categories

- Neural overhead dominates any search benefit.

- Neural policy changes the search in a harmful direction.

- Confidence/staleness signals mispredict when refresh is useful.

- OOD or structurally different instances invalidate in-distribution assumptions.

- Integration or implementation artifacts cause unexpected behavior.

- Training instability produces seed-dependent performance.

- Literature overlap narrows the defensible novelty claim.

## 13.2 Required failure-report wording

Use precise statements such as: "On benchmark family X, the neural policy increased inference cost without producing a compensating reduction in search work." Avoid anthropomorphic or causal language unless the experiment actually isolates the causal mechanism.

# 14. Discussion Specification

The Discussion must answer the research questions directly. It must not become a second Results section.

| **Discussion question**                    | **Evidence to cite**                                                                |
|--------------------------------------------|-------------------------------------------------------------------------------------|
| Does neural guidance help?                 | M1/M2 result tables + search-work metrics.                                          |
| When does it help?                         | Difficulty/family/scale breakdowns.                                                 |
| When is it not worth the cost?             | Overhead decomposition + controller traces.                                         |
| Did richer guidance help?                  | Ablation table.                                                                     |
| Did adaptive refresh change the trade-off? | M5 table + call/runtime frontier.                                                   |
| Does it generalize?                        | Held-out size/structure results.                                                    |
| What remains uncertain?                    | Seed variability, OOD behavior, reproduction gaps, and unresolved novelty boundary. |

# 15. Novelty Gate for the Final Report

Before the team uses the word "novel" in the title, abstract, contribution list, or conclusion, all five gates below must pass.

1\. Mechanism gate: the report names the exact algorithmic mechanism, inputs, outputs, and decision rule.

2\. Prior-art gate: Document 14’s audit does not identify the same mechanism as established prior art.

3\. Differentiation gate: the report explains at least one technically meaningful difference from the closest prior methods.

4\. Implementation gate: the claimed mechanism actually exists in the frozen code/configuration used for results.

5\. Evidence gate: the mechanism is exercised in a controlled experiment and its effect is documented, including negative findings if present.

If any gate fails, replace "novel" with "proposed," "investigated," "implemented," or another evidence-appropriate term. The report must never manufacture novelty from benchmark speed alone.

# 16. Limitations & Threats to Validity

The report must state the project’s real limitations. At minimum, cover benchmark representativeness, hardware dependence, neural training cost, seed variability, solver implementation details, possible benchmark overfitting, OOD confidence reliability, timeouts/censoring, and the incompleteness of any literature audit.

| **Threat**                 | **Required mitigation/reporting**                                                                               |
|----------------------------|-----------------------------------------------------------------------------------------------------------------|
| Benchmark selection bias   | Report family composition and keep final test sets frozen from tuning.                                          |
| Hardware dependence        | Record CPU/GPU model, software versions, and whether GPU transfer cost is included.                             |
| Seed variability           | Use multiple training seeds and report seed-aware uncertainty.                                                  |
| Hyperparameter overfitting | Freeze settings before final test; distinguish validation from test use.                                        |
| Solver-specific effects    | State that findings are established for the implemented MiniSat-style interface unless broader evidence exists. |
| Neural training cost       | Separate training cost from per-solve inference overhead and state whether amortization is considered.          |
| Literature incompleteness  | Use careful language and state the audit boundary/date.                                                         |

# 17. Reproducibility Package Specification

The final report is incomplete until the evidence package can regenerate the headline tables. The following artifacts must be frozen and archived together.

| **Artifact**     | **Mandatory contents**                                                                                       |
|------------------|--------------------------------------------------------------------------------------------------------------|
| Code             | Exact source tree or repository commit used for final results.                                               |
| Environment      | Environment file/container specification, package versions, compiler versions, and hardware record.          |
| Configs          | Every benchmark, solver, training, controller, timeout, seed, and experiment configuration used.             |
| Data manifest    | Dataset IDs, hashes where applicable, train/validation/test split definitions, and generated-instance seeds. |
| Checkpoints      | Exact neural checkpoints used for each reported seed/model.                                                  |
| Raw logs         | Document 11 schema-compliant per-run/per-decision logs.                                                      |
| Analysis scripts | Scripts/notebooks that create final tables, figures, and statistical summaries from raw logs.                |
| Result manifest  | Mapping from report table/figure IDs to experiment IDs, config IDs, input artifacts, and output files.       |

# 18. Result Provenance Convention

Every headline table and figure must carry a provenance identifier. Use a stable naming scheme such as REPORT-T3, REPORT-F4, etc., and record the upstream experiment configuration. The goal is to make a result auditable without opening the source code.

| **Field**          | **Example form**                                |
|--------------------|-------------------------------------------------|
| Report artifact ID | REPORT-T5                                       |
| Experiment IDs     | EXP-M5-001, EXP-M5-002 ...                      |
| Configuration      | CFG-M5-FIXED-10 / CFG-M5-ADAPTIVE               |
| Data manifest      | DATA-R3SAT-SHIFT-V1                             |
| Seeds              | S1-S5                                           |
| Analysis version   | STAT-v1.0                                       |
| Code revision      | repository commit SHA recorded in final package |

# 19. Autonomous-Agent Writing Constraints

An AI agent generating the final report must not invent missing results, hyperparameters, benchmark counts, statistical significance, or novelty status. Where an evidence field is absent, it must insert an explicit unresolved marker in the draft and stop short of a stronger claim.

- Never fabricate numerical values to make a table look complete.

- Never infer a causal explanation from a correlation unless the experiment isolates the relevant factor.

- Never replace a failed reproduction with an undocumented "fixed" implementation and call it faithful reproduction.

- Never tune on the final test set and then report the result as held-out generalization.

- Never convert an engineering optimization into a research novelty claim without the novelty gate.

- Preserve the exact terminology and method IDs M0-M5 from the frozen project documents.

# 20. Final Submission Checklist

| **Gate**                | **Pass condition**                                                                    |
|-------------------------|---------------------------------------------------------------------------------------|
| Scientific completeness | Every PRD research question is answered, deferred, or explicitly marked inconclusive. |
| Baseline validity       | M0 and M1 are validated to the level claimed.                                         |
| Experimental fairness   | Common instances/settings/hardware/timeout rules are documented.                      |
| Overhead accounting     | Graph build, inference, call count, and end-to-end runtime are separately visible.    |
| Statistical integrity   | Document 13 analysis rules are followed and exploratory analyses are labeled.         |
| Ablation completeness   | Key controller, feature, and refresh terms are isolated.                              |
| Generalization          | Size and structural shifts are tested or the limitation is explicit.                  |
| Novelty control         | Document 14 gate is passed before any strong novelty wording is used.                 |
| Reproducibility         | Headline tables/figures regenerate from archived artifacts.                           |
| Failure analysis        | Negative and benchmark-specific failures are documented.                              |
| No invented claims      | Every factual conclusion is traceable to evidence.                                    |

# 21. Recommended Final Conclusion Template

The conclusion should contain four compact paragraphs: (1) what problem was studied and what system was built; (2) what the evidence established about neural guidance and overhead; (3) what the adaptive-refresh experiment established or failed to establish; and (4) what remains uncertain, including the exact novelty boundary and the most important future experiment. The conclusion must not introduce new numerical results.

# 22. Final Claim-Control Statement

The final report is considered scientifically complete when every major sentence that could be challenged as a factual, comparative, causal, or novelty claim can be traced to a frozen experiment, a defined metric, and an appropriate evidence level. The project should prefer a smaller set of defensible claims over a larger set of impressive but weakly supported claims. A result showing that neural guidance is not consistently faster, or that adaptive refresh provides only a conditional benefit, is still a valid and valuable outcome when measured rigorously.

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


# V2.0 Reporting Amendment

The final report must explain the controller as **three layers**:

1. Trust;
2. Refresh value;
3. Branch selection.

## Required primary result table

The central table must include at least:

| Method | Solved | Runtime | Decisions | Conflicts | Neural calls | Refreshes | Controller overhead |
|---|---:|---:|---:|---:|---:|---:|---:|

## Required controller evidence

Show at least one figure/table for:

- predicted vs realized refresh benefit;
- trust rejections by reason;
- neural-call reduction;
- runtime overhead decomposition;
- M5 vs Fixed-K per-instance paired outcome.

## Claim discipline

Never write:

> "The controller knows when refreshing is valuable."

Write:

> "The estimator predicts expected future benefit from observed calibration data, and the controller refreshes when predicted benefit exceeds measured/estimated refresh cost."

Never write:

> "OOD detection solves distribution shift."

Write:

> "The OOD safeguard rejects states whose embedding distance exceeds the calibrated threshold; its robustness under the tested shifts is evaluated empirically."

Never write:

> "M5 beats Fixed-K."

unless the final statistical analysis actually supports that scoped statement.


## V2.0 Cross-Document Consistency Check

This amendment is synchronized with Documents 01–20. The implementation agent must treat the newest V2.0 definitions as authoritative within this document and record any residual contradiction as a Decision Log item rather than silently choosing a meaning.
