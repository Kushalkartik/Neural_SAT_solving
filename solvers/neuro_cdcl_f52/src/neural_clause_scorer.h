#ifndef _NEURAL_CLAUSE_SCORER_H_
#define _NEURAL_CLAUSE_SCORER_H_

#include <stdbool.h>
#include <stdint.h>

struct kissat;
struct clause;

extern int kissat_harvesting_enabled;

/* Extracts 6 normalized features for candidate clause c into feats[6] */
void kissat_extract_clause_features (struct kissat *solver,
                                     const struct clause *c,
                                     double max_act,
                                     float *feats);

/* Evaluates the distilled Nano-MLP student on the 6 features */
/* Returns a 32-bit usefulness rank score (higher = keep/useful) */
uint32_t kissat_score_clause_quality (struct kissat *solver,
                                      const struct clause *c,
                                      double max_act);

/* Causal Harvester hooks: */
void kissat_track_clause_at_reduce (struct kissat *solver,
                                    struct clause *c,
                                    double max_act);

void kissat_notify_clause_used_in_search_impl (const struct clause *c);

static inline void kissat_notify_clause_used_in_search (const struct clause *c) {
  if (__builtin_expect (kissat_harvesting_enabled != 0, 0)) {
    kissat_notify_clause_used_in_search_impl (c);
  }
}

void kissat_flush_tracked_clauses (struct kissat *solver);

#endif /* _NEURAL_CLAUSE_SCORER_H_ */
