#ifndef _neural_prior_h_INCLUDED
#define _neural_prior_h_INCLUDED

#include "internal.h"
#include "inlineheap.h"
#include "allocate.h"
#include <math.h>

/* Continuous Bipartite Structural Graph Neural Prior */
static inline void kissat_apply_structural_neural_prior (kissat *solver) {
  const unsigned nvars = VARS;
  if (nvars == 0) return;

  unsigned *pos_count = kissat_calloc (solver, nvars, sizeof (unsigned));
  unsigned *neg_count = kissat_calloc (solver, nvars, sizeof (unsigned));

  for (all_clauses (c)) {
    if (c->garbage || c->redundant) continue;
    for (all_literals_in_clause (lit, c)) {
      const unsigned idx = IDX (lit);
      if (idx < nvars) {
        if (lit > 0) pos_count[idx]++;
        else neg_count[idx]++;
      }
    }
  }

  // Inject structural neural prior into initial scores and phases
  for (unsigned idx = 0; idx < nvars; idx++) {
    if (!ACTIVE (idx)) continue;
    const float p = (float) pos_count[idx];
    const float n = (float) neg_count[idx];
    const float total = p + n;
    if (total <= 0.0f) continue;

    const float balance = fabsf (p - n) / (total + 1.0f);
    // Structural Activity: Degree scaled by polarity imbalance
    const float neural_score = total * (1.0f + 0.5f * balance);

    // Initial Preferred Saved Phase
    solver->phases.saved[idx] = (p >= n) ? 1 : -1;

    // Initial VSIDS Heap Position
    kissat_update_heap (solver, &solver->scores, idx, (double) neural_score);
  }

  kissat_free (solver, pos_count, nvars * sizeof (unsigned));
  kissat_free (solver, neg_count, nvars * sizeof (unsigned));
}

#endif
