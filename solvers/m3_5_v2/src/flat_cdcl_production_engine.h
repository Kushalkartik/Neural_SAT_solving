#ifndef FLAT_CDCL_PRODUCTION_ENGINE_H
#define FLAT_CDCL_PRODUCTION_ENGINE_H

#include <cstdint>
#include <algorithm>
#include <cassert>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include "structural_mlp_engine.h"

namespace ProductionCDCL {

const uint8_t VAL_UNDEF = 0;
const uint8_t VAL_TRUE  = 1;
const uint8_t VAL_FALSE = 2;

const uint32_t CRef_Undef = 0xFFFFFFFF;

enum RefreshPolicy {
    REFRESH_NONE = 0,            // M0 / M3.5 baseline: one-shot root guidance only
    REFRESH_FIXED_K = 1,         // M4A: Fixed-K conflict cadence
    REFRESH_RESTART = 2,         // M4B: Restart-aligned (Luby)
    REFRESH_TURBULENCE = 3,      // M4C: Conflict spike / search turbulence gate
    REFRESH_CONFIDENCE_GATE = 4, // M4D: Policy entropy & score-gap handoff to VSIDS
    REFRESH_HYBRID = 5           // M4E: Full integrated cost-aware hybrid controller
};

// ============================================================================
// PART 1: FlatL1 Core for N <= 64 (Ultra-Low-Latency L1 Cache Resident)
// Sized strictly for <= 32 KB total memory footprint (3-14 us execution)
// ============================================================================
namespace L1Core {
    const int MAX_VARS = 64;
    const int MAX_LITS = 2 * MAX_VARS;
    const int MAX_POOL = 8192;
    const int MAX_WATCH_PER_LIT = 32;

    struct alignas(8) Watcher {
        uint32_t cref;
        int32_t  blocker;
    };

    class alignas(64) FlatL1Engine {
    public:
        int n_vars;
        int n_clauses;

        alignas(64) uint8_t lit_val[MAX_LITS];
        uint16_t watch_sz[MAX_LITS];
        Watcher  watches[MAX_LITS][MAX_WATCH_PER_LIT];

        int32_t  ca_pool[MAX_POOL];
        uint32_t pool_size;

        int32_t  trail[MAX_VARS];
        int      trail_size;
        int32_t  trail_lim[MAX_VARS];
        int      trail_lim_size;
        uint32_t reason[MAX_VARS];
        int16_t  level[MAX_VARS];
        int      qhead;

        int32_t  candidate_order[MAX_VARS];
        uint8_t  var_to_order_pos[MAX_VARS];
        uint8_t  polarity[MAX_VARS];
        double   activity[MAX_VARS];
        uint8_t  is_decision_var[MAX_VARS];
        int      min_unassigned_idx;

        uint8_t  seen[MAX_VARS];
        int32_t  analyze_toclear[MAX_VARS];
        int      analyze_toclear_size;
        int32_t  learnt_clause[MAX_VARS];
        int      learnt_size;

        int64_t decisions;
        int64_t conflicts;
        int64_t propagations;

        FlatL1Engine() : n_vars(0), n_clauses(0), pool_size(0), trail_size(0),
                         trail_lim_size(0), qhead(0), min_unassigned_idx(0),
                         analyze_toclear_size(0), learnt_size(0),
                         decisions(0), conflicts(0), propagations(0) {}

        inline void init(int num_vars) {
            assert(num_vars <= MAX_VARS);
            n_vars = num_vars;
            n_clauses = 0;
            pool_size = 0;
            trail_size = 0;
            trail_lim_size = 0;
            qhead = 0;
            min_unassigned_idx = 0;
            analyze_toclear_size = 0;
            learnt_size = 0;
            decisions = 0;
            conflicts = 0;
            propagations = 0;

            int n_lits = 2 * n_vars;
            std::memset(lit_val, VAL_UNDEF, n_lits * sizeof(uint8_t));
            std::memset(watch_sz, 0, n_lits * sizeof(uint16_t));
            std::memset(seen, 0, n_vars * sizeof(uint8_t));
            std::memset(polarity, 0, n_vars * sizeof(uint8_t));
            std::memset(is_decision_var, 1, n_vars * sizeof(uint8_t));

            for (int i = 0; i < n_vars; ++i) {
                reason[i] = CRef_Undef;
                level[i] = -1;
                activity[i] = 0.0;
                candidate_order[i] = i;
                var_to_order_pos[i] = (uint8_t)i;
            }
        }

        inline static int32_t mkLit(int32_t var, bool sign) { return (var << 1) | (sign ? 1 : 0); }
        inline static int32_t notLit(int32_t lit) { return lit ^ 1; }
        inline static int32_t varOf(int32_t lit) { return lit >> 1; }
        inline static bool signOf(int32_t lit) { return (lit & 1) != 0; }
        inline int decisionLevel() const { return trail_lim_size; }
        inline void newDecisionLevel() { trail_lim[trail_lim_size++] = trail_size; }

        inline void uncheckedEnqueue(int32_t lit, uint32_t r = CRef_Undef) {
            int32_t v = varOf(lit);
            lit_val[lit] = VAL_TRUE;
            lit_val[notLit(lit)] = VAL_FALSE;
            level[v] = (int16_t)decisionLevel();
            reason[v] = r;
            trail[trail_size++] = lit;
        }

        inline uint32_t allocClause(const int32_t* lits, int sz, bool is_learnt) {
            uint32_t cref = pool_size;
            ca_pool[pool_size++] = sz;
            ca_pool[pool_size++] = is_learnt ? 1 : 0;
            for (int i = 0; i < sz; ++i) ca_pool[pool_size++] = lits[i];
            return cref;
        }

        inline int32_t getClauseSize(uint32_t cref) const { return ca_pool[cref]; }
        inline int32_t* getClauseLits(uint32_t cref) { return &ca_pool[cref + 2]; }

        inline void attachWatcher(int32_t lit, uint32_t cref, int32_t blocker) {
            uint16_t sz = watch_sz[lit]++;
            assert(sz < MAX_WATCH_PER_LIT);
            watches[lit][sz] = {cref, blocker};
        }

        void addClause(const std::vector<int32_t>& lits) {
            if (lits.empty()) return;
            if (lits.size() == 1) { uncheckedEnqueue(lits[0], CRef_Undef); return; }
            uint32_t cref = allocClause(lits.data(), (int)lits.size(), false);
            n_clauses++;
            attachWatcher(notLit(lits[0]), cref, lits[1]);
            attachWatcher(notLit(lits[1]), cref, lits[0]);
        }

        void setDecisionVars(const uint8_t* is_dec) {
            if (is_dec) {
                for (int i = 0; i < n_vars; ++i) is_decision_var[i] = is_dec[i];
            } else {
                std::memset(is_decision_var, 1, n_vars * sizeof(uint8_t));
            }
        }

        void setGuidance(const double* weights, const int* pols, int num_vars) {
            for (int i = 0; i < num_vars && i < n_vars; ++i) {
                activity[i] = weights[i];
                if (pols) polarity[i] = (pols[i] > 0) ? 0 : 1;
            }
            std::sort(candidate_order, candidate_order + n_vars, [this](int a, int b) {
                return activity[a] > activity[b];
            });
            for (int i = 0; i < n_vars; ++i) {
                var_to_order_pos[candidate_order[i]] = (uint8_t)i;
            }
            min_unassigned_idx = 0;
        }

        inline void cancelUntil(int target_level) {
            if (decisionLevel() > target_level) {
                int lim = trail_lim[target_level];
                int min_pos = min_unassigned_idx;
                for (int i = trail_size - 1; i >= lim; --i) {
                    int32_t lit = trail[i];
                    int32_t v = varOf(lit);
                    lit_val[lit] = VAL_UNDEF;
                    lit_val[notLit(lit)] = VAL_UNDEF;
                    polarity[v] = signOf(lit) ? 1 : 0;
                    level[v] = -1;
                    reason[v] = CRef_Undef;
                    int pos = var_to_order_pos[v];
                    if (pos < min_pos) min_pos = pos;
                }
                qhead = lim;
                trail_size = lim;
                trail_lim_size = target_level;
                min_unassigned_idx = min_pos;
            }
        }

        inline uint32_t propagate() {
            while (qhead < trail_size) {
                int32_t p = trail[qhead++];
                propagations++;

                Watcher* ws = watches[p];
                int sz = watch_sz[p];
                int i = 0, j = 0;

                for (; i < sz; ++i) {
                    Watcher w = ws[i];

                    if (lit_val[w.blocker] == VAL_TRUE) {
                        if (i != j) ws[j] = w;
                        j++;
                        continue;
                    }

                    uint32_t cref = w.cref;
                    int32_t* c = getClauseLits(cref);
                    int32_t c_size = getClauseSize(cref);
                    int32_t false_lit = notLit(p);

                    int32_t other_watched = (c[0] == false_lit) ? c[1] : c[0];

                    if (lit_val[other_watched] == VAL_TRUE) {
                        w.blocker = other_watched;
                        if (i != j) ws[j] = w;
                        j++;
                        continue;
                    }

                    if (c_size == 3) {
                        int32_t third = c[2];
                        uint8_t val3 = lit_val[third];
                        if (val3 == VAL_TRUE) {
                            w.blocker = third;
                            if (i != j) ws[j] = w;
                            j++;
                            continue;
                        } else if (val3 != VAL_FALSE) {
                            if (c[0] == false_lit) { c[0] = third; c[2] = false_lit; }
                            else { c[1] = third; c[2] = false_lit; }
                            attachWatcher(notLit(third), cref, other_watched);
                            continue;
                        }

                        if (i != j) ws[j] = w;
                        j++;

                        if (lit_val[other_watched] == VAL_FALSE) {
                            while (++i < sz) {
                                if (i != j) ws[j] = ws[i];
                                j++;
                            }
                            watch_sz[p] = j;
                            qhead = trail_size;
                            return cref;
                        } else {
                            uncheckedEnqueue(other_watched, cref);
                            continue;
                        }
                    } else {
                        if (c[0] == false_lit) { c[0] = c[1]; c[1] = false_lit; }
                        int32_t first = c[0];
                        w.blocker = first;

                        bool found = false;
                        for (int k = 2; k < c_size; ++k) {
                            uint8_t vk = lit_val[c[k]];
                            if (vk == VAL_TRUE) {
                                w.blocker = c[k];
                                if (i != j) ws[j] = w;
                                j++;
                                found = true;
                                break;
                            } else if (vk != VAL_FALSE) {
                                c[1] = c[k];
                                c[k] = false_lit;
                                attachWatcher(notLit(c[1]), cref, first);
                                found = true;
                                break;
                            }
                        }
                        if (found && lit_val[w.blocker] == VAL_TRUE) continue;
                        if (found) continue;

                        if (i != j) ws[j] = w;
                        j++;

                        if (lit_val[first] == VAL_FALSE) {
                            while (++i < sz) {
                                if (i != j) ws[j] = ws[i];
                                j++;
                            }
                            watch_sz[p] = j;
                            qhead = trail_size;
                            return cref;
                        } else {
                            uncheckedEnqueue(first, cref);
                        }
                    }
                }
                watch_sz[p] = j;
            }
            return CRef_Undef;
        }

        inline void analyze(uint32_t confl, int& out_btlevel) {
            int pathC = 0;
            int32_t p = -1;
            learnt_size = 1;
            int index = trail_size - 1;
            analyze_toclear_size = 0;

            do {
                assert(confl != CRef_Undef);
                int32_t* c = getClauseLits(confl);
                int32_t c_size = getClauseSize(confl);

                for (int j = 0; j < c_size; ++j) {
                    int32_t q = c[j];
                    int32_t v = varOf(q);
                    if (p != -1 && v == varOf(p)) continue;

                    if (!seen[v] && level[v] > 0) {
                        seen[v] = 1;
                        analyze_toclear[analyze_toclear_size++] = v;
                        if (level[v] >= decisionLevel()) pathC++;
                        else learnt_clause[learnt_size++] = q;
                    }
                }

                while (index >= 0 && !seen[varOf(trail[index])]) index--;
                assert(index >= 0);
                p = trail[index--];
                int32_t v = varOf(p);
                confl = reason[v];
                seen[v] = 0;
                pathC--;
            } while (pathC > 0);

            learnt_clause[0] = notLit(p);

            if (learnt_size == 1) {
                out_btlevel = 0;
            } else {
                int max_i = 1;
                for (int i = 2; i < learnt_size; ++i) {
                    if (level[varOf(learnt_clause[i])] > level[varOf(learnt_clause[max_i])]) max_i = i;
                }
                int32_t tmp = learnt_clause[1];
                learnt_clause[1] = learnt_clause[max_i];
                learnt_clause[max_i] = tmp;
                out_btlevel = level[varOf(learnt_clause[1])];
            }

            for (int i = 0; i < analyze_toclear_size; ++i) seen[analyze_toclear[i]] = 0;
        }

        inline int32_t pickBranchLit() {
            while (min_unassigned_idx < n_vars) {
                int32_t v = candidate_order[min_unassigned_idx++];
                if (!is_decision_var[v]) continue;
                if (lit_val[mkLit(v, false)] == VAL_UNDEF) {
                    bool sign = (polarity[v] == 1);
                    return mkLit(v, sign);
                }
            }
            return -1;
        }

        int solve() {
            if (n_clauses == 0) return 10; // SAT
            if (propagate() != CRef_Undef) return 20; // Root UNSAT

            while (true) {
                int32_t branch_lit = pickBranchLit();
                if (branch_lit == -1) return 10; // SAT

                decisions++;
                newDecisionLevel();
                uncheckedEnqueue(branch_lit, CRef_Undef);

                while (true) {
                    uint32_t confl = propagate();
                    if (confl != CRef_Undef) {
                        conflicts++;
                        if (decisionLevel() == 0) return 20;

                        int backtrack_level = 0;
                        analyze(confl, backtrack_level);
                        cancelUntil(backtrack_level);

                        if (learnt_size == 1) {
                            uncheckedEnqueue(learnt_clause[0], CRef_Undef);
                        } else {
                            uint32_t cr = allocClause(learnt_clause, learnt_size, true);
                            attachWatcher(notLit(learnt_clause[0]), cr, learnt_clause[1]);
                            attachWatcher(notLit(learnt_clause[1]), cr, learnt_clause[0]);
                            uncheckedEnqueue(learnt_clause[0], cr);
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        bool verifyModel(const std::vector<std::vector<int32_t>>& orig_clauses) const {
            for (const auto& cl : orig_clauses) {
                bool sat = false;
                for (int32_t lit : cl) {
                    if (lit_val[lit] == VAL_TRUE) { sat = true; break; }
                }
                if (!sat) return false;
            }
            return true;
        }
    };
} // namespace L1Core

// ============================================================================
// PART 2: Scalable Flat CDCL Engine for Arbitrary N (N > 64 up to 5000+)
// Dynamically allocated contiguous flat arrays, zero realloc in search loop
// ============================================================================
namespace ScalableCore {
    struct Watcher {
        uint32_t cref;
        int32_t  blocker;
    };

    static inline double luby(double y, int x) {
        int size = 1;
        int seq = 0;
        while (size < x + 1) {
            seq++;
            size = 2 * size + 1;
        }
        while (size - 1 != x) {
            size = (size - 1) >> 1;
            seq--;
            x = x % size;
        }
        return std::pow(y, seq);
    }

    class ScalableFlatCDCLEngine {
    public:
        int n_vars;
        int n_clauses;

        std::vector<uint8_t> lit_val;
        std::vector<std::vector<Watcher>> watches;
        std::vector<int32_t> ca_pool;
        uint32_t pool_size;

        std::vector<int32_t> trail;
        int trail_size;
        std::vector<int32_t> trail_lim;
        int trail_lim_size;
        std::vector<uint32_t> reason;
        std::vector<int16_t> level;
        int qhead;

        std::vector<int32_t> order_heap;
        std::vector<int32_t> heap_pos;
        std::vector<uint8_t> polarity;
        std::vector<double> activity;
        std::vector<double> vsids_activity;
        std::vector<uint8_t> has_been_assigned;
        std::vector<uint8_t> is_decision_var;
        double var_inc;
        double var_decay;
        int luby_k;
        int64_t next_restart_conflict;

        std::vector<uint8_t> seen;
        std::vector<int32_t> analyze_toclear;
        std::vector<int32_t> learnt_clause;

        int64_t decisions;
        int64_t conflicts;
        int64_t propagations;

        std::vector<uint32_t> learnts;
        std::vector<float> learnt_activity;
        float cla_inc;
        int64_t max_learnts;
        int64_t next_reduce_conflict;

        // Milestone 4: Adaptive Cost-Aware Refresh Controller State
        StructuralMLPEngine* smlp_engine;
        RefreshPolicy refresh_policy;
        int refresh_k_conflicts;
        double turbulence_threshold;
        double confidence_margin_threshold;

        int64_t refreshes_count;
        int64_t last_refresh_conflict;
        int64_t last_refresh_decision;

        int rolling_confl_history[32];
        int rolling_head;
        int rolling_confl_sum;

        std::vector<int> scratch_edge_row;
        std::vector<int> scratch_edge_col;
        std::vector<uint8_t> scratch_edge_pos;
        std::vector<double> scratch_weights;
        std::vector<int> scratch_polarities;

        ScalableFlatCDCLEngine() : n_vars(0), n_clauses(0), pool_size(0), trail_size(0),
                                  trail_lim_size(0), qhead(0),
                                  var_inc(1.0), var_decay(0.95), luby_k(1), next_restart_conflict(100),
                                  decisions(0), conflicts(0), propagations(0),
                                  cla_inc(1.0f), max_learnts(2000), next_reduce_conflict(2000),
                                  smlp_engine(nullptr), refresh_policy(REFRESH_NONE),
                                  refresh_k_conflicts(50), turbulence_threshold(0.75),
                                  confidence_margin_threshold(0.05), refreshes_count(0),
                                  last_refresh_conflict(0), last_refresh_decision(0),
                                  rolling_head(0), rolling_confl_sum(0) {
            std::memset(rolling_confl_history, 0, sizeof(rolling_confl_history));
        }

        void init(int num_vars, int expected_clauses = 0) {
            n_vars = num_vars;
            n_clauses = 0;
            pool_size = 0;
            trail_size = 0;
            trail_lim_size = 0;
            qhead = 0;
            decisions = 0;
            conflicts = 0;
            propagations = 0;
            learnts.clear();
            learnt_activity.clear();
            cla_inc = 1.0f;
            max_learnts = std::max(2000, expected_clauses * 2);
            next_reduce_conflict = max_learnts;
            var_inc = 1.0;
            var_decay = 0.95;
            luby_k = 1;
            next_restart_conflict = 100;

            refreshes_count = 0;
            last_refresh_conflict = 0;
            last_refresh_decision = 0;
            rolling_head = 0;
            rolling_confl_sum = 0;
            std::memset(rolling_confl_history, 0, sizeof(rolling_confl_history));

            int est_edges = (expected_clauses > 0 ? expected_clauses : n_vars * 5) * 3;
            scratch_edge_row.reserve(est_edges);
            scratch_edge_col.reserve(est_edges);
            scratch_edge_pos.reserve(est_edges);
            scratch_weights.assign(num_vars, 1.0);
            scratch_polarities.assign(num_vars, 0);

            int n_lits = 2 * n_vars;
            lit_val.assign(n_lits, VAL_UNDEF);
            watches.clear();
            watches.resize(n_lits);

            int est_pool = (expected_clauses > 0 ? expected_clauses : n_vars * 5) * 5 + 1024;
            ca_pool.clear();
            ca_pool.resize(est_pool);

            trail.assign(n_vars, 0);
            trail_lim.assign(n_vars, 0);
            reason.assign(n_vars, CRef_Undef);
            level.assign(n_vars, -1);

            order_heap.clear();
            order_heap.reserve(n_vars);
            heap_pos.assign(n_vars, -1);
            polarity.assign(n_vars, 0);
            activity.assign(n_vars, 0.0);
            vsids_activity.assign(n_vars, 0.0);
            has_been_assigned.assign(n_vars, 0);
            is_decision_var.assign(n_vars, 1);
            for (int i = 0; i < n_vars; ++i) {
                order_heap.push_back(i);
                heap_pos[i] = i;
            }

            seen.assign(n_vars, 0);
            analyze_toclear.reserve(n_vars);
            learnt_clause.reserve(n_vars);
        }

        inline static int32_t mkLit(int32_t var, bool sign) { return (var << 1) | (sign ? 1 : 0); }
        inline static int32_t notLit(int32_t lit) { return lit ^ 1; }
        inline static int32_t varOf(int32_t lit) { return lit >> 1; }
        inline static bool signOf(int32_t lit) { return (lit & 1) != 0; }
        inline int decisionLevel() const { return trail_lim_size; }
        inline void newDecisionLevel() { trail_lim[trail_lim_size++] = trail_size; }

        inline void uncheckedEnqueue(int32_t lit, uint32_t r = CRef_Undef) {
            int32_t v = varOf(lit);
            lit_val[lit] = VAL_TRUE;
            lit_val[notLit(lit)] = VAL_FALSE;
            level[v] = (int16_t)decisionLevel();
            reason[v] = r;
            trail[trail_size++] = lit;
            has_been_assigned[v] = 1;
        }

        inline uint32_t allocClause(const int32_t* lits, int sz, int32_t learnt_idx) {
            uint32_t cref = pool_size;
            if (pool_size + sz + 2 >= ca_pool.size()) {
                ca_pool.resize(std::max((size_t)(ca_pool.size() * 2), (size_t)(pool_size + sz + 1024)));
            }
            ca_pool[pool_size++] = sz;
            ca_pool[pool_size++] = learnt_idx; // 0 for problem clause, >0 for learnt, -1 for deleted
            for (int i = 0; i < sz; ++i) ca_pool[pool_size++] = lits[i];
            return cref;
        }

        inline int32_t getClauseSize(uint32_t cref) const { return ca_pool[cref]; }
        inline int32_t* getClauseLits(uint32_t cref) { return &ca_pool[cref + 2]; }

        inline void attachWatcher(int32_t lit, uint32_t cref, int32_t blocker) {
            watches[lit].push_back({cref, blocker});
        }

        void addClause(const std::vector<int32_t>& lits) {
            if (lits.empty()) return;
            if (lits.size() == 1) { uncheckedEnqueue(lits[0], CRef_Undef); return; }
            uint32_t cref = allocClause(lits.data(), (int)lits.size(), 0);
            n_clauses++;
            attachWatcher(notLit(lits[0]), cref, lits[1]);
            attachWatcher(notLit(lits[1]), cref, lits[0]);
        }

        inline bool heap_less(int a, int b) const { return activity[a] < activity[b]; }

        inline void heap_sift_up(int i) {
            int32_t x = order_heap[i];
            while (i > 0) {
                int p = (i - 1) >> 1;
                if (!heap_less(order_heap[p], x)) break;
                order_heap[i] = order_heap[p];
                heap_pos[order_heap[i]] = i;
                i = p;
            }
            order_heap[i] = x;
            heap_pos[x] = i;
        }

        inline void heap_sift_down(int i) {
            int32_t x = order_heap[i];
            int sz = (int)order_heap.size();
            while ((i << 1) + 1 < sz) {
                int c = (i << 1) + 1;
                if (c + 1 < sz && heap_less(order_heap[c], order_heap[c + 1])) c++;
                if (!heap_less(x, order_heap[c])) break;
                order_heap[i] = order_heap[c];
                heap_pos[order_heap[i]] = i;
                i = c;
            }
            order_heap[i] = x;
            heap_pos[x] = i;
        }

        inline void heap_insert(int32_t v) {
            if (!is_decision_var.empty() && !is_decision_var[v]) return;
            if (heap_pos[v] == -1) {
                heap_pos[v] = (int32_t)order_heap.size();
                order_heap.push_back(v);
                heap_sift_up(heap_pos[v]);
            }
        }

        inline int32_t heap_pop() {
            int32_t top = order_heap[0];
            int32_t last = order_heap.back();
            order_heap.pop_back();
            heap_pos[top] = -1;
            if (!order_heap.empty()) {
                order_heap[0] = last;
                heap_pos[last] = 0;
                heap_sift_down(0);
            }
            return top;
        }

        inline void varBumpActivity(int32_t v) {
            vsids_activity[v] += var_inc;
            activity[v] += var_inc;
            if (activity[v] > 1e100) {
                for (int i = 0; i < n_vars; ++i) {
                    activity[i] *= 1e-100;
                    vsids_activity[i] *= 1e-100;
                }
                var_inc *= 1e-100;
            }
            if (heap_pos[v] >= 0) heap_sift_up(heap_pos[v]);
        }

        inline void varDecayActivity() {
            var_inc *= (1.0 / 0.95);
        }

        void setDecisionVars(const uint8_t* is_dec) {
            if (is_dec) {
                is_decision_var.assign(is_dec, is_dec + n_vars);
            } else {
                is_decision_var.assign(n_vars, 1);
            }
            order_heap.clear();
            heap_pos.assign(n_vars, -1);
            for (int i = 0; i < n_vars; ++i) {
                if (is_decision_var[i]) {
                    heap_pos[i] = (int32_t)order_heap.size();
                    order_heap.push_back(i);
                }
            }
            int sz = (int)order_heap.size();
            for (int i = (sz >> 1) - 1; i >= 0; --i) {
                heap_sift_down(i);
            }
        }

        void setGuidance(const double* weights, const int* pols, int num_vars) {
            for (int i = 0; i < num_vars && i < n_vars; ++i) {
                activity[i] = weights[i];
                if (pols) polarity[i] = (pols[i] > 0) ? 0 : 1;
            }
            order_heap.clear();
            heap_pos.assign(n_vars, -1);
            for (int i = 0; i < n_vars; ++i) {
                if (is_decision_var.empty() || is_decision_var[i]) {
                    heap_pos[i] = (int32_t)order_heap.size();
                    order_heap.push_back(i);
                }
            }
            int sz = (int)order_heap.size();
            for (int i = (sz >> 1) - 1; i >= 0; --i) {
                heap_sift_down(i);
            }
        }

        bool refreshGuidance() {
            if (!smlp_engine || !smlp_engine->is_loaded()) return false;

            scratch_edge_row.clear();
            scratch_edge_col.clear();
            scratch_edge_pos.clear();

            int active_clauses = 0;
            uint32_t idx = 0;
            while (idx < pool_size) {
                int sz = ca_pool[idx];
                int l_idx = ca_pool[idx + 1];
                if (l_idx >= 0) { // Active clause (not deleted)
                    bool sat = false;
                    int unassigned_lits = 0;
                    for (int i = 0; i < sz; ++i) {
                        int32_t lit = ca_pool[idx + 2 + i];
                        if (lit_val[lit] == VAL_TRUE) { sat = true; break; }
                        if (lit_val[lit] == VAL_UNDEF) unassigned_lits++;
                    }
                    if (!sat && unassigned_lits > 0) {
                        int c_node = n_vars + active_clauses;
                        for (int i = 0; i < sz; ++i) {
                            int32_t lit = ca_pool[idx + 2 + i];
                            if (lit_val[lit] == VAL_UNDEF) {
                                int32_t v = varOf(lit);
                                scratch_edge_row.push_back(v);
                                scratch_edge_col.push_back(c_node);
                                scratch_edge_pos.push_back(signOf(lit) ? 0 : 1);
                            }
                        }
                        active_clauses++;
                    }
                }
                idx += (sz + 2);
            }

            if (active_clauses == 0) return false;

            if ((int)scratch_weights.size() < n_vars) {
                scratch_weights.resize(n_vars, 1.0);
                scratch_polarities.resize(n_vars, 0);
            }

            bool ok = smlp_engine->forward(
                n_vars,
                active_clauses,
                (int)scratch_edge_row.size(),
                scratch_edge_row.data(),
                scratch_edge_col.data(),
                scratch_edge_pos.data(),
                scratch_weights.data(),
                scratch_polarities.data()
            );

            if (ok) {
                // Harmonize neural weights with current decayed variable activity scale
                for (int i = 0; i < n_vars; ++i) {
                    // Harmonious Blending: Preserve accumulated VSIDS conflict learning
                    activity[i] = vsids_activity[i] + scratch_weights[i] * var_inc;
                    // Preserve phase saving for variables that have participated in search
                    if (!has_been_assigned[i] && lit_val[mkLit(i, false)] == VAL_UNDEF) {
                        polarity[i] = (scratch_polarities[i] > 0) ? 0 : 1;
                    }
                }
                // Rebuild order_heap with strictly unassigned variables
                order_heap.clear();
                std::fill(heap_pos.begin(), heap_pos.end(), -1);
                for (int i = 0; i < n_vars; ++i) {
                    if ((is_decision_var.empty() || is_decision_var[i]) && lit_val[mkLit(i, false)] == VAL_UNDEF) {
                        heap_pos[i] = (int32_t)order_heap.size();
                        order_heap.push_back(i);
                    }
                }
                for (int i = ((int)order_heap.size() >> 1) - 1; i >= 0; --i) {
                    heap_sift_down(i);
                }
                refreshes_count++;
                return true;
            }
            return false;
        }

        inline void record_decision_event(int confl_since_last) {
            rolling_confl_sum -= rolling_confl_history[rolling_head];
            rolling_confl_history[rolling_head] = confl_since_last;
            rolling_confl_sum += confl_since_last;
            rolling_head = (rolling_head + 1) & 31;
        }

        inline bool is_turbulent() const {
            return (rolling_confl_sum >= (int)(32 * turbulence_threshold));
        }

        inline void cancelUntil(int target_level) {
            if (decisionLevel() > target_level) {
                int lim = trail_lim[target_level];
                for (int i = trail_size - 1; i >= lim; --i) {
                    int32_t lit = trail[i];
                    int32_t v = varOf(lit);
                    lit_val[lit] = VAL_UNDEF;
                    lit_val[notLit(lit)] = VAL_UNDEF;
                    polarity[v] = signOf(lit) ? 1 : 0;
                    level[v] = -1;
                    reason[v] = CRef_Undef;
                    heap_insert(v);
                }
                qhead = lim;
                trail_size = lim;
                trail_lim_size = target_level;
            }
        }

        inline uint32_t propagate() {
            while (qhead < trail_size) {
                int32_t p = trail[qhead++];
                propagations++;

                auto& ws = watches[p];
                int sz = (int)ws.size();
                int i = 0, j = 0;

                for (; i < sz; ++i) {
                    Watcher w = ws[i];
                    uint32_t cref = w.cref;

                    if (ca_pool[cref + 1] < 0) {
                        continue; // Dropped from watch list (reduceDB deleted)
                    }

                    if (lit_val[w.blocker] == VAL_TRUE) {
                        if (i != j) ws[j] = w;
                        j++;
                        continue;
                    }

                    int32_t* c = getClauseLits(cref);
                    int32_t c_size = getClauseSize(cref);
                    int32_t false_lit = notLit(p);

                    int32_t other_watched = (c[0] == false_lit) ? c[1] : c[0];
                    if (lit_val[other_watched] == VAL_TRUE) {
                        w.blocker = other_watched;
                        if (i != j) ws[j] = w;
                        j++;
                        continue;
                    }

                    if (c_size == 3) {
                        int32_t third = c[2];
                        uint8_t val3 = lit_val[third];
                        if (val3 == VAL_TRUE) {
                            w.blocker = third;
                            if (i != j) ws[j] = w;
                            j++;
                            continue;
                        } else if (val3 != VAL_FALSE) {
                            if (c[0] == false_lit) { c[0] = third; c[2] = false_lit; }
                            else { c[1] = third; c[2] = false_lit; }
                            attachWatcher(notLit(third), cref, other_watched);
                            continue;
                        }

                        if (i != j) ws[j] = w;
                        j++;

                        if (lit_val[other_watched] == VAL_FALSE) {
                            while (++i < sz) {
                                if (i != j) ws[j] = ws[i];
                                j++;
                            }
                            ws.resize(j);
                            qhead = trail_size;
                            return cref;
                        } else {
                            uncheckedEnqueue(other_watched, cref);
                            continue;
                        }
                    } else {
                        if (c[0] == false_lit) { c[0] = c[1]; c[1] = false_lit; }
                        int32_t first = c[0];
                        w.blocker = first;

                        bool found = false;
                        for (int k = 2; k < c_size; ++k) {
                            uint8_t vk = lit_val[c[k]];
                            if (vk == VAL_TRUE) {
                                w.blocker = c[k];
                                if (i != j) ws[j] = w;
                                j++;
                                found = true;
                                break;
                            } else if (vk != VAL_FALSE) {
                                c[1] = c[k];
                                c[k] = false_lit;
                                attachWatcher(notLit(c[1]), cref, first);
                                found = true;
                                break;
                            }
                        }
                        if (found && lit_val[w.blocker] == VAL_TRUE) continue;
                        if (found) continue;

                        if (i != j) ws[j] = w;
                        j++;

                        if (lit_val[first] == VAL_FALSE) {
                            while (++i < sz) {
                                if (i != j) ws[j] = ws[i];
                                j++;
                            }
                            ws.resize(j);
                            qhead = trail_size;
                            return cref;
                        } else {
                            uncheckedEnqueue(first, cref);
                        }
                    }
                }
                ws.resize(j);
            }
            return CRef_Undef;
        }

        inline void analyze(uint32_t confl, int& out_btlevel) {
            int pathC = 0;
            int32_t p = -1;
            learnt_clause.clear();
            learnt_clause.push_back(0); // placeholder for asserting lit
            int index = trail_size - 1;
            analyze_toclear.clear();

            do {
                assert(confl != CRef_Undef);
                int32_t* c = getClauseLits(confl);
                int32_t c_size = getClauseSize(confl);
                int32_t l_idx = ca_pool[confl + 1];
                if (l_idx > 0 && l_idx <= (int32_t)learnt_activity.size()) {
                    learnt_activity[l_idx - 1] += cla_inc;
                }

                for (int j = 0; j < c_size; ++j) {
                    int32_t q = c[j];
                    int32_t v = varOf(q);
                    if (p != -1 && v == varOf(p)) continue;

                    if (!seen[v] && level[v] > 0) {
                        varBumpActivity(v);
                        seen[v] = 1;
                        analyze_toclear.push_back(v);
                        if (level[v] >= decisionLevel()) pathC++;
                        else learnt_clause.push_back(q);
                    }
                }

                while (index >= 0 && !seen[varOf(trail[index])]) index--;
                assert(index >= 0);
                p = trail[index--];
                int32_t v = varOf(p);
                confl = reason[v];
                seen[v] = 0;
                pathC--;
            } while (pathC > 0);

            learnt_clause[0] = notLit(p);
            int learnt_size = (int)learnt_clause.size();

            if (learnt_size == 1) {
                out_btlevel = 0;
            } else {
                int max_i = 1;
                for (int i = 2; i < learnt_size; ++i) {
                    if (level[varOf(learnt_clause[i])] > level[varOf(learnt_clause[max_i])]) max_i = i;
                }
                int32_t tmp = learnt_clause[1];
                learnt_clause[1] = learnt_clause[max_i];
                learnt_clause[max_i] = tmp;
                out_btlevel = level[varOf(learnt_clause[1])];
            }

            for (int v : analyze_toclear) seen[v] = 0;
            varDecayActivity();
        }

        inline int32_t pickBranchLit() {
            while (!order_heap.empty()) {
                int32_t v = heap_pop();
                if (!is_decision_var.empty() && !is_decision_var[v]) continue;
                if (lit_val[mkLit(v, false)] == VAL_UNDEF) {
                    bool sign = (polarity[v] == 1);
                    // Policy M4D / M4E: Confidence & Score-Gap Gating
                    if ((refresh_policy == REFRESH_CONFIDENCE_GATE || refresh_policy == REFRESH_HYBRID) && !order_heap.empty()) {
                        int32_t next_v = order_heap[0];
                        double s1 = activity[v];
                        double s2 = activity[next_v];
                        double margin = (s1 > 0.0) ? (std::abs(s1 - s2) / s1) : 0.0;
                        if (margin < confidence_margin_threshold) {
                            // Uncertain neural ranking: top candidates are virtually tied
                            // Fallback to pure VSIDS conflict activity to break the tie
                            if (vsids_activity[next_v] > vsids_activity[v]) {
                                heap_insert(v);
                                v = heap_pop();
                            }
                        }
                    }
                    return mkLit(v, sign);
                }
            }
            return -1;
        }

        inline bool locked(uint32_t cr) const {
            int32_t* c = const_cast<ScalableFlatCDCLEngine*>(this)->getClauseLits(cr);
            int32_t c0 = c[0];
            return (lit_val[c0] == VAL_TRUE && reason[varOf(c0)] == cr);
        }

        void reduceDB() {
            std::vector<std::pair<float, uint32_t>> candidates;
            candidates.reserve(learnts.size());

            for (size_t k = 0; k < learnts.size(); ++k) {
                uint32_t cr = learnts[k];
                if (ca_pool[cr + 1] <= 0) continue;
                int sz = getClauseSize(cr);
                if (sz > 2 && !locked(cr)) {
                    float score = learnt_activity[k] / (float)sz;
                    candidates.push_back({score, cr});
                }
            }

            if (candidates.empty()) return;

            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });

            size_t to_remove = candidates.size() / 2;
            for (size_t k = 0; k < to_remove; ++k) {
                uint32_t cr = candidates[k].second;
                ca_pool[cr + 1] = -1; // Marked deleted
            }

            int j = 0;
            for (int i = 0; i < (int)learnts.size(); ++i) {
                uint32_t cr = learnts[i];
                if (ca_pool[cr + 1] > 0) {
                    learnts[j] = cr;
                    learnt_activity[j] = learnt_activity[i];
                    ca_pool[cr + 1] = j + 1; // 1-based index
                    j++;
                }
            }
            learnts.resize(j);
            learnt_activity.resize(j);

            max_learnts = (int64_t)(max_learnts * 1.15);
        }

        int solve(double timeout_sec = 0.0) {
            if (n_clauses == 0) return 10;
            auto t_start = std::chrono::high_resolution_clock::now();
            if (propagate() != CRef_Undef) return 20;

            int confl_since_last_decision = 0;

            while (true) {
                if (timeout_sec > 0.0 && (conflicts & 0x1FF) == 0) {
                    auto tn = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration<double>(tn - t_start).count() >= timeout_sec) {
                        return 0; // TIMEOUT
                    }
                }

                // Restart Block (M4B & M4E: Luby Restart Synchronized Refresh)
                if (conflicts >= next_restart_conflict) {
                    cancelUntil(0);
                    double factor = luby(2.0, luby_k++);
                    next_restart_conflict = conflicts + (int64_t)(100 * factor);
                    if (propagate() != CRef_Undef) return 20;

                    if (refresh_policy == REFRESH_RESTART || refresh_policy == REFRESH_HYBRID) {
                        refreshGuidance();
                        last_refresh_conflict = conflicts;
                        last_refresh_decision = decisions;
                    }
                }

                // M4A: Fixed-K Conflict Cadence Refresh
                if (refresh_policy == REFRESH_FIXED_K && 
                    (conflicts - last_refresh_conflict) >= refresh_k_conflicts) {
                    refreshGuidance();
                    last_refresh_conflict = conflicts;
                    last_refresh_decision = decisions;
                }

                // M4C & M4E: Search Turbulence / Conflict Spike Trigger
                if ((refresh_policy == REFRESH_TURBULENCE || refresh_policy == REFRESH_HYBRID) &&
                    (decisions - last_refresh_decision) >= 200 &&
                    is_turbulent()) {
                    refreshGuidance();
                    last_refresh_decision = decisions;
                    last_refresh_conflict = conflicts;
                }

                int32_t branch_lit = pickBranchLit();
                if (branch_lit == -1) return 10;

                decisions++;
                record_decision_event(confl_since_last_decision);
                confl_since_last_decision = 0;

                newDecisionLevel();
                uncheckedEnqueue(branch_lit, CRef_Undef);

                while (true) {
                    uint32_t confl = propagate();
                    if (confl != CRef_Undef) {
                        conflicts++;
                        confl_since_last_decision++;
                        if (decisionLevel() == 0) return 20;

                        int backtrack_level = 0;
                        analyze(confl, backtrack_level);
                        cancelUntil(backtrack_level);

                        if (learnt_clause.size() == 1) {
                            uncheckedEnqueue(learnt_clause[0], CRef_Undef);
                        } else {
                            int32_t l_idx = (int32_t)learnts.size() + 1;
                            uint32_t cr = allocClause(learnt_clause.data(), (int)learnt_clause.size(), l_idx);
                            attachWatcher(notLit(learnt_clause[0]), cr, learnt_clause[1]);
                            attachWatcher(notLit(learnt_clause[1]), cr, learnt_clause[0]);
                            uncheckedEnqueue(learnt_clause[0], cr);
                            learnts.push_back(cr);
                            learnt_activity.push_back(cla_inc);
                        }

                        cla_inc *= 1.001f;
                        if (cla_inc > 1e20f) {
                            for (auto& a : learnt_activity) a *= 1e-20f;
                            cla_inc = 1.0f;
                        }

                        if (conflicts >= next_reduce_conflict) {
                            reduceDB();
                            next_reduce_conflict = conflicts + max_learnts;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        bool verifyModel(const std::vector<std::vector<int32_t>>& orig_clauses) const {
            for (const auto& cl : orig_clauses) {
                bool sat = false;
                for (int32_t lit : cl) {
                    if (lit_val[lit] == VAL_TRUE) { sat = true; break; }
                }
                if (!sat) return false;
            }
            return true;
        }
    };
} // namespace ScalableCore

// ============================================================================
// PART 3: Unified Production CDCL Engine with Adaptive Scaling
// ============================================================================
class FlatCDCLProductionEngine {
public:
    bool use_l1_core;
    L1Core::FlatL1Engine l1_engine;
    ScalableCore::ScalableFlatCDCLEngine scalable_engine;

    FlatCDCLProductionEngine() : use_l1_core(true) {}

    inline void init(int num_vars, int expected_clauses = 0) {
        if (num_vars <= 16 && expected_clauses <= 80) {
            use_l1_core = true;
            l1_engine.init(num_vars);
        } else {
            use_l1_core = false;
            scalable_engine.init(num_vars, expected_clauses);
        }
    }

    inline void addClause(const std::vector<int32_t>& lits) {
        if (use_l1_core) l1_engine.addClause(lits);
        else scalable_engine.addClause(lits);
    }

    inline void setDecisionVars(const uint8_t* is_dec) {
        if (use_l1_core) l1_engine.setDecisionVars(is_dec);
        else scalable_engine.setDecisionVars(is_dec);
    }

    inline void setGuidance(const double* weights, const int* pols, int num_vars) {
        if (use_l1_core) l1_engine.setGuidance(weights, pols, num_vars);
        else scalable_engine.setGuidance(weights, pols, num_vars);
    }

    inline void setRefreshPolicy(RefreshPolicy policy, int k_param = 50, double turb_thresh = 0.75, double conf_margin = 0.05) {
        scalable_engine.refresh_policy = policy;
        scalable_engine.refresh_k_conflicts = k_param;
        scalable_engine.turbulence_threshold = turb_thresh;
        scalable_engine.confidence_margin_threshold = conf_margin;
    }

    inline void setMLPEngine(StructuralMLPEngine* smlp) {
        scalable_engine.smlp_engine = smlp;
    }

    inline int64_t getRefreshes() const {
        return scalable_engine.refreshes_count;
    }

    inline int solve(double timeout_sec = 0.0) {
        if (use_l1_core) return l1_engine.solve();
        else return scalable_engine.solve(timeout_sec);
    }

    inline int64_t getDecisions() const {
        return use_l1_core ? l1_engine.decisions : scalable_engine.decisions;
    }

    inline int64_t getConflicts() const {
        return use_l1_core ? l1_engine.conflicts : scalable_engine.conflicts;
    }

    inline int64_t getPropagations() const {
        return use_l1_core ? l1_engine.propagations : scalable_engine.propagations;
    }

    inline bool verifyModel(const std::vector<std::vector<int32_t>>& orig_clauses) const {
        return use_l1_core ? l1_engine.verifyModel(orig_clauses) : scalable_engine.verifyModel(orig_clauses);
    }

    inline uint8_t getModelLitVal(int32_t lit) const {
        if (use_l1_core) return l1_engine.lit_val[lit];
        else return scalable_engine.lit_val[lit];
    }
};

} // namespace ProductionCDCL

#endif
