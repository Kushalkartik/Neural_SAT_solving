#include "neural_clause_scorer.h"
#include "internal.h"
#include "clause.h"
#include "heap.h"
#include "value.h"
#include "literal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if __has_include("nano_clause_net.h")
#include "nano_clause_net.h"
#define HAS_NANO_CLAUSE_NET 1
#else
#define HAS_NANO_CLAUSE_NET 0
#endif

/* ------------------------------------------------------------------------- */
/* Causal Ground-Truth Tracking Hash Table                                   */
/* ------------------------------------------------------------------------- */
#define TRACK_HASH_SIZE 65536
#define TRACK_HASH_MASK (TRACK_HASH_SIZE - 1)

typedef struct {
  const clause *c_ptr;
  float feats[6];
  bool used_in_future;
  bool occupied;
} tracked_clause_sample;

static tracked_clause_sample tracked_table[TRACK_HASH_SIZE];
static unsigned total_tracked = 0;
static FILE *harvest_file = NULL;
static int harvest_init_checked = 0;
int kissat_harvesting_enabled = 0;

static void check_harvest_init (void) {
  if (harvest_init_checked) return;
  harvest_init_checked = 1;
  const char *path = getenv ("KISSAT_HARVEST_CLAUSES");
  if (path && *path) {
    harvest_file = fopen (path, "ab");
    if (harvest_file)
      kissat_harvesting_enabled = 1;
  }
}

void kissat_extract_clause_features (kissat *solver,
                                     const clause *c,
                                     double max_act,
                                     float *feats) {
  (void) solver;
  (void) max_act;
  const unsigned size = c->size > 0 ? c->size : 1;
  const unsigned glue = c->glue;
  const unsigned used = c->used;

  // f0: normalized glue
  float f0 = (float) glue / 50.0f;
  if (f0 > 1.0f) f0 = 1.0f;

  // f1: normalized size
  float f1 = (float) size / 100.0f;
  if (f1 > 1.0f) f1 = 1.0f;

  // f2: normalized recent usage (0..31)
  float f2 = (float) used / 31.0f;
  if (f2 > 1.0f) f2 = 1.0f;

  // Fast L1-resident features (zero pointer chasing, zero heap cache misses)
  feats[0] = f0;
  feats[1] = f1;
  feats[2] = f2;
  feats[3] = f2; // Activity proxy via usage
  feats[4] = 0.5f;
  feats[5] = 0.0f;
}

uint32_t kissat_score_clause_quality (kissat *solver,
                                      const clause *c,
                                      double max_act) {
  float feats[6];
  kissat_extract_clause_features (solver, c, max_act, feats);

#if HAS_NANO_CLAUSE_NET
  float q = evaluate_nano_clause_quality (feats);
  if (q > 1.0f) q = 1.0f;
  if (q < 0.0f) q = 0.0f;
  return (uint32_t) (q * 4294967295.0f);
#else
  // Causal heuristic prior before neural distillation: activity + usage
  float q = 0.5f * feats[2] + 0.5f * feats[3];
  return (uint32_t) (q * 4294967295.0f);
#endif
}

/* ------------------------------------------------------------------------- */
/* Two-Phase Causal Tracking Implementation                                  */
/* ------------------------------------------------------------------------- */
void kissat_track_clause_at_reduce (kissat *solver,
                                    clause *c,
                                    double max_act) {
  check_harvest_init ();
  if (!harvest_file) return;

  uintptr_t h = (((uintptr_t) c) >> 3) & TRACK_HASH_MASK;
  for (unsigned step = 0; step < 16; step++) {
    unsigned idx = (unsigned) ((h + step) & TRACK_HASH_MASK);
    if (!tracked_table[idx].occupied || tracked_table[idx].c_ptr == c) {
      tracked_table[idx].c_ptr = c;
      kissat_extract_clause_features (solver, c, max_act, tracked_table[idx].feats);
      tracked_table[idx].used_in_future = false;
      tracked_table[idx].occupied = true;
      total_tracked++;
      return;
    }
  }
}

void kissat_notify_clause_used_in_search_impl (const clause *c) {
  if (!harvest_file) return;
  uintptr_t h = (((uintptr_t) c) >> 3) & TRACK_HASH_MASK;
  for (unsigned step = 0; step < 16; step++) {
    unsigned idx = (unsigned) ((h + step) & TRACK_HASH_MASK);
    if (!tracked_table[idx].occupied) return;
    if (tracked_table[idx].c_ptr == c) {
      tracked_table[idx].used_in_future = true;
      return;
    }
  }
}

void kissat_flush_tracked_clauses (kissat *solver) {
  (void) solver;
  check_harvest_init ();
  if (!harvest_file || total_tracked == 0) {
    memset (tracked_table, 0, sizeof (tracked_table));
    total_tracked = 0;
    return;
  }

  // Write out all tracked clauses from previous epoch
  float record[7];
  for (unsigned i = 0; i < TRACK_HASH_SIZE; i++) {
    if (tracked_table[i].occupied && tracked_table[i].c_ptr) {
      for (int k = 0; k < 6; k++) record[k] = tracked_table[i].feats[k];
      record[6] = tracked_table[i].used_in_future ? 1.0f : 0.0f;
      fwrite (record, sizeof (float), 7, harvest_file);
    }
  }
  fflush (harvest_file);

  // Reset table for next epoch
  memset (tracked_table, 0, sizeof (tracked_table));
  total_tracked = 0;
}
