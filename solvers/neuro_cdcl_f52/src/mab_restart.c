#include "mab_restart.h"
#include "nano_mlp_distilled.h"
#include "internal.h"
#include "logging.h"
#include "print.h"
#include "report.h"
#include "restart.h"

#include <math.h>
#include <float.h>
#include <inttypes.h>

void kissat_configure_parametric_preprocessing (kissat *solver) {
  (void) solver;
  kissat_verbose (solver, "Parametric Preprocessing: Native Kissat simplification pipeline active (100%% stock fidelity)");
}

static void apply_arm_settings (mab_restart *mab, unsigned arm) {
  mab->current_arm = arm;
  switch (arm) {
  case 0:
    mab->active_restart_margin = 3.0;
    mab->epoch_conflicts_budget = 400;
    break;

  case 1:
    mab->active_restart_margin = 12.0;
    mab->epoch_conflicts_budget = 600;
    break;

  case 2:
    mab->active_restart_margin = 40.0;
    mab->epoch_conflicts_budget = 1000;
    break;

  case 3:
    mab->active_restart_margin = 100.0;
    mab->epoch_conflicts_budget = 1200;
    break;
  }
}

// Strictly Causal Zero-Lookahead State Extraction: State observable BEFORE action runs
static void extract_causal_search_state (kissat *solver, float *s) {
  uint64_t vars = solver->vars > 0 ? solver->vars : 1;
  uint64_t irredundant = solver->statistics.clauses_irredundant;
  uint64_t binary = solver->statistics.clauses_binary;
  uint64_t total_clauses = irredundant + binary;
  if (total_clauses == 0) total_clauses = 1;

  // s0: Trail penetration depth ratio
  float trail_sz = (float) SIZE_STACK (solver->trail);
  s[0] = trail_sz / (float) vars;
  if (s[0] > 1.0f) s[0] = 1.0f;
  if (s[0] < 0.0f) s[0] = 0.0f;

  // s1: Fast vs Slow Glue trend (turbulence indicator)
  float slow_g = (float) AVERAGE (slow_glue);
  float fast_g = (float) AVERAGE (fast_glue);
  s[1] = (slow_g > 0.0f) ? (fast_g / slow_g) : 1.0f;
  if (s[1] > 2.0f) s[1] = 2.0f;
  if (s[1] < 0.0f) s[1] = 0.0f;

  // s2: Implication graph binary clause connectivity
  s[2] = (float) binary / (float) total_clauses;
  if (s[2] > 1.0f) s[2] = 1.0f;

  // s3: Normalized clause-to-variable density
  s[3] = (float) total_clauses / (float) vars / 10.0f;
  if (s[3] > 1.0f) s[3] = 1.0f;

  // s4: Search mode (Focused = 0.0, Stable = 1.0)
  s[4] = solver->stable ? 1.0f : 0.0f;

  // s5: Logarithmic conflict lifecycle progress (Early = 0.2, Mid = 0.6, Late = 0.9+)
  uint64_t conf = CONFLICTS > 0 ? CONFLICTS : 1;
  s[5] = log10f ((float) conf) / 6.0f;
  if (s[5] > 1.0f) s[5] = 1.0f;
  if (s[5] < 0.0f) s[5] = 0.0f;

  // s6: Normalized decision level tree depth
  s[6] = (float) solver->level / (float) vars;
  if (s[6] > 1.0f) s[6] = 1.0f;

  // s7: Historical lemma productivity EMA momentum
  s[7] = (float) solver->mab.ema_reward;
  if (s[7] > 1.0f) s[7] = 1.0f;
  if (s[7] < 0.0f) s[7] = 0.0f;
}

// Compute Neuro-Symbolic Initial Prior
static void compute_topological_prior (kissat *solver, double *priors, unsigned *best_arm_init) {
  uint64_t vars = solver->vars > 0 ? solver->vars : 1;
  uint64_t irredundant = solver->statistics.clauses_irredundant;
  uint64_t binary = solver->statistics.clauses_binary;
  uint64_t total_clauses = irredundant + binary;
  if (total_clauses == 0) total_clauses = 1;

  double clause_var_ratio = (double) total_clauses / (double) vars;

  // Structural Topology Archetype: Deep Industrial Hardware (vars >= 12,000 & c_v_ratio >= 3.8)
  // Deep Search (Arm 2) prevents trail thrashing and allows deep resolution refutations
  if (vars >= 12000 && clause_var_ratio >= 3.8) {
    priors[0] = 0.40;
    priors[1] = 0.55;
    priors[2] = 0.65;
    priors[3] = 0.55;
    *best_arm_init = 2;
    kissat_very_verbose (solver,
                         "Topological Prior: Deep Industrial Hardware (vars=%" PRIu64 ", c/v=%.2f) -> Arm 2 (Deep Search)",
                         vars, clause_var_ratio);
    return;
  }

  // Causal Neural Student Prior for Compact / Agile / General Formulas
  float s[NANO_PRECISION_INPUT_DIM];
  extract_causal_search_state (solver, s);

  float q_neural[NANO_PRECISION_OUTPUT_DIM];
  nano_distilled_forward (s, q_neural);

  float max_p = -1.0f;
  unsigned b_arm = 0;
  for (unsigned i = 0; i < MAB_NUM_ARMS; i++) {
    priors[i] = 0.40 + 0.30 * (double) q_neural[i];
    if (q_neural[i] > max_p) {
      max_p = q_neural[i];
      b_arm = i;
    }
  }
  *best_arm_init = b_arm;
  kissat_very_verbose (solver,
                       "Neural Student Contextual Prior: [A0:%.2f, A1:%.2f, A2:%.2f, A3:%.2f] -> Initial Arm %u",
                       priors[0], priors[1], priors[2], priors[3], b_arm);
}

void kissat_init_mab_restart (kissat *solver) {
  mab_restart *mab = &solver->mab;
  mab->initialized = true;
  mab->ema_reward = 0.05;

  double priors[MAB_NUM_ARMS];
  unsigned best_arm = 1;
  compute_topological_prior (solver, priors, &best_arm);

  apply_arm_settings (mab, best_arm);

  mab->epoch_conflicts_start = CONFLICTS;
  mab->epoch_conflicts = 0;
  mab->epoch_units = 0;
  mab->epoch_glue2 = 0;
  mab->epoch_glue4 = 0;

  mab->total_pulls = 0;
  for (unsigned i = 0; i < MAB_NUM_ARMS; i++) {
    mab->rewards[i] = priors[i];
    mab->pulls[i] = 2; // Confident prior initialization
    mab->total_pulls += 2;
  }

  // Extract initial state representation
  extract_causal_search_state (solver, mab->current_state);

  // Universal Deterministic Warm-Start: 50,000 conflicts of stock Kissat
  mab->thrashing_detected = false;
  mab->grace_period_conflicts = 50000;
  mab->consecutive_stalls = 0;
  mab->initial_exploration_step = 0;

  kissat_very_verbose (solver,
                       "initialized Neuro-Symbolic Adaptive MAB restart controller with prior arm %u",
                       best_arm);
}

void kissat_mab_on_learned (kissat *solver, unsigned glue, unsigned size) {
  if (!solver->mab.initialized)
    return;
  mab_restart *mab = &solver->mab;
  if (size == 1 || glue == 0) {
    mab->epoch_units++;
  } else if (glue <= 2) {
    mab->epoch_glue2++;
  } else if (glue <= 4) {
    mab->epoch_glue4++;
  }
}

void kissat_mab_update_and_select (kissat *solver) {
  if (!solver->mab.initialized)
    return;
  mab_restart *mab = &solver->mab;

  uint64_t current_conflicts = CONFLICTS;
  uint64_t delta_conflicts = (current_conflicts >= mab->epoch_conflicts_start)
                                 ? (current_conflicts - mab->epoch_conflicts_start)
                                 : 1;
  if (delta_conflicts == 0)
    delta_conflicts = 1;

  // Real-world lemma reward earned by mab->current_arm
  double r = (3.0 * (double) mab->epoch_units + 2.0 * (double) mab->epoch_glue2 +
              1.0 * (double) mab->epoch_glue4) /
             (double) delta_conflicts;
  if (r > 2.0)
    r = 2.0;
  if (r < 0.0)
    r = 0.0;

  unsigned active = mab->current_arm;
  const double alpha = 0.25;
  mab->rewards[active] = (1.0 - alpha) * mab->rewards[active] + alpha * r;
  mab->pulls[active]++;
  mab->total_pulls++;

  // Update exponential moving average momentum
  mab->ema_reward = 0.85 * mab->ema_reward + 0.15 * r;

  // Measure dynamic search state s_{t+1} and run Causal Neural Student
  extract_causal_search_state (solver, mab->current_state);

  float q_neural[NANO_PRECISION_OUTPUT_DIM];
  nano_distilled_forward (mab->current_state, q_neural);

  // Dynamic UCB1 + Neural Student Prior Selection
  const double c_explore = 0.35;
  const double lambda_prior = 0.30;
  double best_score = -1e9;
  unsigned best_arm = 0;
  double ln_n = log ((double) mab->total_pulls);

  for (unsigned i = 0; i < MAB_NUM_ARMS; i++) {
    double neural_prior = (lambda_prior / (1.0 + 0.08 * (double) mab->pulls[i])) * (double) q_neural[i];
    double explore = c_explore * sqrt (ln_n / (double) mab->pulls[i]);
    double score = mab->rewards[i] + neural_prior + explore;
    if (score > best_score) {
      best_score = score;
      best_arm = i;
    }
  }

  apply_arm_settings (mab, best_arm);
  mab->epoch_conflicts_start = current_conflicts;
  mab->epoch_units = 0;
  mab->epoch_glue2 = 0;
  mab->epoch_glue4 = 0;

  kissat_very_verbose (solver,
                       "Neuro-Symbolic MAB selected arm %u (Score: %.3f, mean_reward: %.3f, neural_q: %.3f, pulls: %" PRIu64 ")",
                       best_arm, best_score, mab->rewards[best_arm], q_neural[best_arm],
                       mab->pulls[best_arm]);
}

void kissat_mab_check_epoch (kissat *solver) {
  if (!solver->mab.initialized)
    return;
  uint64_t current_conflicts = CONFLICTS;
  if (!solver->mab.thrashing_detected) {
    if (current_conflicts >= solver->mab.grace_period_conflicts) {
      solver->mab.thrashing_detected = true;
      solver->mab.epoch_conflicts_start = current_conflicts;
      solver->mab.epoch_units = 0;
      solver->mab.epoch_glue2 = 0;
      solver->mab.epoch_glue4 = 0;

      kissat_verbose (solver,
                      "Transitioning to Stage 2: Adaptive Precision-Student MAB Controller (Threshold: %" PRIu64 " conflicts reached)",
                      solver->mab.grace_period_conflicts);
    }
    return;
  }

  if (current_conflicts >= solver->mab.epoch_conflicts_start + solver->mab.epoch_conflicts_budget) {
    kissat_mab_update_and_select (solver);
  }
}
