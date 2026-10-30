#include "deduce.h"
#include "inline.h"
#include "promote.h"
#include "strengthen.h"
#include "neural_clause_scorer.h"

static inline void recompute_and_promote (kissat *solver, clause *c) {
  assert (c->redundant);
  const unsigned old_glue = c->glue;
  const unsigned new_glue = kissat_recompute_glue (solver, c, old_glue);
  if (new_glue < old_glue)
    kissat_promote_clause (solver, c, new_glue);
}

static inline void mark_clause_as_used (kissat *solver, clause *c) {
  if (!c->redundant)
    return;
  INC (clauses_used);
  c->used = MAX_USED;
  kissat_notify_clause_used_in_search (c);
  LOGCLS (c, "using");
  recompute_and_promote (solver, c);
  unsigned glue = MIN (c->glue, MAX_GLUE_USED);
  solver->statistics.used[solver->stable].glue[glue]++;
  if (solver->stable)
    INC (clauses_used_stable);
  else
    INC (clauses_used_focused);
}

bool kissat_recompute_and_promote (kissat *solver, clause *c) {
  assert (c->redundant);
  const unsigned old_glue = c->glue;
  const unsigned new_glue = kissat_recompute_glue (solver, c, old_glue);
  if (new_glue >= old_glue)
    return false;
  kissat_promote_clause (solver, c, new_glue);
  return true;
}

static inline bool analyze_literal (kissat *solver, assigned *all_assigned,
                                    frame *frames, unsigned lit) {
  assert (VALUE (lit) < 0);
  const unsigned idx = IDX (lit);
  assigned *a = all_assigned + idx;
  const unsigned level = a->level;
  if (!level)
    return false;
  solver->antecedent_size++;
  if (a->analyzed)
    return false;
  LOG ("analyzing literal %s", LOGLIT (lit));
  kissat_push_analyzed (solver, all_assigned, idx);
  assert (level <= solver->level);
#if defined(LOGGING) || !defined(NDEBUG)
  PUSH_STACK (solver->resolvent, lit);
#endif
  solver->resolvent_size++;
  if (level == solver->level)
    return true;
  assert (a->analyzed);
  PUSH_STACK (solver->clause, lit);
  LOG ("learned literal %s", LOGLIT (lit));
  frame *f = frames + level;
  if (f->used++)
    return false;
  LOG ("pulling in decision level %u", level);
  PUSH_STACK (solver->levels, level);
  return false;
}

clause *kissat_deduce_first_uip_clause (kissat *solver, clause *conflict) {
  START (deduce);
  assert (EMPTY_STACK (solver->analyzed));
  assert (EMPTY_STACK (solver->levels));
  assert (EMPTY_STACK (solver->clause));
#if defined(LOGGING) || !defined(NDEBUG)
  CLEAR_STACK (solver->resolvent);
#endif
  if (conflict->size > 2)
    mark_clause_as_used (solver, conflict);
  PUSH_STACK (solver->clause, INVALID_LIT);
  solver->antecedent_size = 0;
  solver->resolvent_size = 0;
  unsigned unresolved_on_current_level = 0, conflict_size = 0;
  assigned *all_assigned = solver->assigned;
  frame *frames = BEGIN_STACK (solver->frames);
  const float c_size_f = (float)(conflict->size > 0 ? conflict->size : 1);
  const float base_flow = 1.0f / c_size_f;
  for (all_literals_in_clause (lit, conflict)) {
    assert (VALUE (lit) < 0);
    if (LEVEL (lit))
      conflict_size++;
    if (solver->cif_weights) {
      const unsigned lit_idx = IDX (lit);
      solver->cif_weights[lit_idx] += base_flow;
    }
    if (analyze_literal (solver, all_assigned, frames, lit))
      unresolved_on_current_level++;
  }
  assert (unresolved_on_current_level > 1);
  LOG ("starting with %u unresolved literals on current decision level",
       unresolved_on_current_level);
  assert (solver->antecedent_size == solver->resolvent_size);
  LOGRES2 ("initial");
  const bool otfs = GET_OPTION (otfs);
  unsigned const *t = END_ARRAY (solver->trail);
  unsigned uip = INVALID_LIT;
  unsigned resolved = 0;
  assigned *a = 0;
  for (;;) {
    do {
      assert (t > BEGIN_ARRAY (solver->trail));
      uip = *--t;
      a = ASSIGNED (uip);
    } while (!a->analyzed || a->level != solver->level);
    if (unresolved_on_current_level == 1)
      break;
    assert (a->reason != DECISION_REASON);
    assert (a->level == solver->level);
    solver->antecedent_size = 1;
    resolved++;
    const unsigned uip_idx = IDX (uip);
    float uip_flow = (solver->cif_weights) ? solver->cif_weights[uip_idx] : 1.0f;
    if (uip_flow < 1e-4f) uip_flow = 1.0f;
    if (a->binary) {
      const unsigned other = a->reason;
      if (solver->cif_weights) {
        const unsigned other_idx = IDX (other);
        solver->cif_weights[other_idx] += uip_flow;
      }
      LOGBINARY (uip, other, "resolving %s reason", LOGLIT (uip));
      if (analyze_literal (solver, all_assigned, frames, other))
        unresolved_on_current_level++;
    } else {
      const reference ref = a->reason;
      LOGREF (ref, "resolving %s reason", LOGLIT (uip));
      clause *reason = kissat_dereference_clause (solver, ref);
      const unsigned r_size = reason->size;
      if (r_size > 1 && r_size <= 32 && solver->cif_weights) {
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
        count = 0;
        for (all_literals_in_clause (lit, reason)) {
          if (lit != uip) {
            const float alpha = scores[count++] * inv_sum;
            solver->cif_weights[IDX (lit)] += alpha * uip_flow;
            if (analyze_literal (solver, all_assigned, frames, lit))
              unresolved_on_current_level++;
          }
        }
      } else {
        const float split_flow = (r_size > 1) ? (uip_flow / (float)(r_size - 1)) : uip_flow;
        for (all_literals_in_clause (lit, reason)) {
          if (lit != uip) {
            if (solver->cif_weights) solver->cif_weights[IDX(lit)] += split_flow;
            if (analyze_literal (solver, all_assigned, frames, lit))
              unresolved_on_current_level++;
          }
        }
      }
      mark_clause_as_used (solver, reason);
    }
    assert (unresolved_on_current_level > 0);
    unresolved_on_current_level--;
    LOG ("after resolving %s there are %u literals left "
         "on current decision level",
         LOGLIT (uip), unresolved_on_current_level);
    assert (solver->resolvent_size > 0);
    solver->resolvent_size--;
#if defined(LOGGING) || !defined(NDEBUG)
    LOG2 ("actual antecedent size %u", solver->antecedent_size);
    REMOVE_STACK (unsigned, solver->resolvent, NOT (uip));
    assert (SIZE_STACK (solver->resolvent) == solver->resolvent_size);
    LOGRES2 ("new");
#endif
    if (otfs && solver->antecedent_size > 2 &&
        solver->resolvent_size < solver->antecedent_size) {
      assert (!a->binary);
      assert (solver->antecedent_size && solver->resolvent_size + 1);
      clause *reason = kissat_dereference_clause (solver, a->reason);
      assert (!reason->garbage);
      clause *res = kissat_on_the_fly_strengthen (solver, reason, uip);
      if (resolved == 1 && solver->resolvent_size < conflict_size) {
        assert (!conflict->garbage);
        assert (conflict_size > 2);
        kissat_on_the_fly_subsume (solver, res, conflict);
      }
      STOP (deduce);
      return res;
    }
  }
  assert (uip != INVALID_LIT);
  LOG ("first unique implication point %s (1st UIP)", LOGLIT (uip));
  assert (PEEK_STACK (solver->clause, 0) == INVALID_LIT);
  POKE_STACK (solver->clause, 0, NOT (uip));
  LOGTMP ("deduced not yet minimized 1st UIP");
  if (!solver->probing)
    ADD (literals_deduced, SIZE_STACK (solver->clause));
  STOP (deduce);
  return 0;
}
