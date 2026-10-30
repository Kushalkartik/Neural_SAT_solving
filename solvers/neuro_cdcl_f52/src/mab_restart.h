#ifndef _mab_restart_h_INCLUDED
#define _mab_restart_h_INCLUDED

#include <stdint.h>
#include <stdbool.h>

struct kissat;

#define MAB_NUM_ARMS 4
#define NANO_PRECISION_INPUT_DIM 8
#define NANO_PRECISION_HIDDEN1_DIM 32
#define NANO_PRECISION_HIDDEN2_DIM 16
#define NANO_PRECISION_OUTPUT_DIM 4

// Arm 0: Agile Glucose Restarts (tight margin = 3%, budget = 400)
// Arm 1: Standard Balanced Glucose Restarts (nominal margin = 12%, budget = 600)
// Arm 2: Deep Search / Delayed Restarts (loose margin = 40%, budget = 1000)
// Arm 3: Ultra-Deep Search / Suppressed Restarts (margin = 100%, budget = 1200)

typedef struct mab_restart mab_restart;

struct mab_restart {
  bool initialized;
  unsigned current_arm;
  uint64_t epoch_conflicts_start;
  uint64_t epoch_conflicts_budget;

  // Real-time telemetry counters within current epoch
  uint64_t epoch_conflicts;
  uint64_t epoch_units;
  uint64_t epoch_glue2;
  uint64_t epoch_glue4;

  // Statistics per arm
  uint64_t pulls[MAB_NUM_ARMS];
  double rewards[MAB_NUM_ARMS];     // Exponential moving average reward
  uint64_t total_pulls;

  // Active control signals exported to Kissat
  double active_restart_margin;

  // Two-Stage Warm-Start & Adaptive Controller
  bool thrashing_detected;
  unsigned consecutive_stalls;
  uint64_t grace_period_conflicts;
  unsigned initial_exploration_step; // 0..4 for mandatory round-robin

  // Causal Zero-Lookahead State Tracking
  float current_state[NANO_PRECISION_INPUT_DIM];
  double ema_reward;
};

void kissat_configure_parametric_preprocessing (struct kissat *solver);
void kissat_init_mab_restart (struct kissat *solver);
void kissat_mab_on_learned (struct kissat *solver, unsigned glue, unsigned size);
void kissat_mab_check_epoch (struct kissat *solver);
void kissat_mab_update_and_select (struct kissat *solver);

#endif
