#include "minisat_adapter.h"
#include "gnn_engine.h"
#include "structural_mlp_engine.h"
#include "flat_cdcl_production_engine.h"
#include "minisat/simp/SimpSolver.h"
#include "minisat/utils/ParseUtils.h"
#include "minisat/core/Dimacs.h"
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <zlib.h>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace Minisat;

class InteractiveSolver : public SimpSolver {
public:
    int status_val;
    bool is_done_flag;
    bool with_restarts_flag;
    int max_decision_cap_val;
    int current_decisions_count;
    bool stepping_ready;
    double accumulated_cpu_time;
    Lit last_selected_lit;   // The literal actually branched in the most recent step_decision() call

    bool isDecisionVar(int v) const {
        if (v < 0 || v >= decision.size()) return false;
        return decision[v] && !isEliminated(v);
    }

    void extend_model() {
        extendModel();
    }

    // Milestone 3.5: Persistent zero-allocation scratch buffers for GNN graph extraction
    std::vector<int> scratch_unassigned_vars;
    std::vector<int> scratch_var_remapping;
    std::vector<int> scratch_edge_row;
    std::vector<int> scratch_edge_col;
    std::vector<uint8_t> scratch_edge_is_pos;
    std::vector<double> scratch_local_weights;
    std::vector<int> scratch_local_polarities;
    std::vector<double> scratch_solve_w;
    std::vector<int> scratch_solve_p;

    InteractiveSolver() : status_val(STATUS_UNDEF), is_done_flag(false), with_restarts_flag(true),
                          max_decision_cap_val(-1), current_decisions_count(0),
                          stepping_ready(false), accumulated_cpu_time(0.0),
                          last_selected_lit(lit_Undef) {
        verbosity = 0;
    }

    double get_current_cpu_time() const {
#ifdef DISABLE_TELEMETRY
        return 0.0;
#else
#ifdef _WIN32
        FILETIME a, b, c, d;
        if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d)) {
            ULARGE_INTEGER uUser, uKernel;
            uUser.LowPart = d.dwLowDateTime;
            uUser.HighPart = d.dwHighDateTime;
            uKernel.LowPart = c.dwLowDateTime;
            uKernel.HighPart = c.dwHighDateTime;
            return ((double)uUser.QuadPart + (double)uKernel.QuadPart) / 10000000.0;
        }
        return 0.0;
#else
        return cpuTime();
#endif
#endif
    }

    int load_dimacs_file(const char* filepath, bool simplify_formula) {
        gzFile in = gzopen(filepath, "rb");
        if (in == NULL) {
            status_val = STATUS_ERROR;
            is_done_flag = true;
            return -1;
        }

        StreamBuffer buf(in);
        vec<Lit> lits;
        int vars = 0, clauses = 0, parsed_clauses = 0;

        while (true) {
            skipWhitespace(buf);
            if (*buf == EOF) break;
            if (*buf == '%') break; // SATLIB EOF marker
            if (*buf == 'c') {
                skipLine(buf);
            } else if (*buf == 'p') {
                if (!eagerMatch(buf, "p cnf")) {
                    gzclose(in);
                    status_val = STATUS_ERROR;
                    is_done_flag = true;
                    return -1;
                }
                skipWhitespace(buf);
                if (*buf < '0' || *buf > '9') {
                    gzclose(in);
                    status_val = STATUS_ERROR;
                    is_done_flag = true;
                    return -1;
                }
                vars = 0;
                while (*buf >= '0' && *buf <= '9') {
                    vars = vars * 10 + (*buf - '0');
                    ++buf;
                }
                skipWhitespace(buf);
                if (*buf < '0' || *buf > '9') {
                    gzclose(in);
                    status_val = STATUS_ERROR;
                    is_done_flag = true;
                    return -1;
                }
                clauses = 0;
                while (*buf >= '0' && *buf <= '9') {
                    clauses = clauses * 10 + (*buf - '0');
                    ++buf;
                }
            } else {
                // Clause literal loop
                lits.clear();
                while (true) {
                    skipWhitespace(buf);
                    if (*buf == EOF || *buf == '%') break;
                    bool neg = false;
                    if (*buf == '-') { neg = true; ++buf; }
                    else if (*buf == '+') { ++buf; }
                    if (*buf < '0' || *buf > '9') {
                        gzclose(in);
                        status_val = STATUS_ERROR;
                        is_done_flag = true;
                        return -1; // Malformed char
                    }
                    int val = 0;
                    while (*buf >= '0' && *buf <= '9') {
                        val = val * 10 + (*buf - '0');
                        ++buf;
                    }
                    int parsed_lit = neg ? -val : val;
                    if (parsed_lit == 0) break;
                    Var v = std::abs(parsed_lit) - 1;
                    while (v >= nVars()) newVar();
                    lits.push((parsed_lit > 0) ? mkLit(v) : ~mkLit(v));
                }
                parsed_clauses++;
                addClause_(lits);
            }
        }
        gzclose(in);

        if (simplify_formula) {
            eliminate(true);
            if (!okay()) {
                status_val = STATUS_UNSAT;
                is_done_flag = true;
                return 1; // UNSAT proved by preprocessing
            }
        }
        return 0;
    }

    void finalize_sat_model() {
        if (model.size() < nVars()) {
            model.growTo(nVars());
            for (int i = 0; i < nVars(); i++) {
                lbool v = value(i);
                model[i] = (v == l_Undef) ? l_False : v;
            }
        } else {
            for (int i = 0; i < nVars(); i++) {
                if (model[i] == l_Undef) {
                    lbool v = value(i);
                    model[i] = (v == l_Undef) ? l_False : v;
                }
            }
        }
        extendModel();
    }

    int solve_continuous() {
#ifndef DISABLE_TELEMETRY
        double t0 = get_current_cpu_time();
#endif
        if (!with_restarts_flag) {
            restart_first = INT32_MAX;
        }
        vec<Lit> dummy;
        bool sat = solve(dummy);
#ifndef DISABLE_TELEMETRY
        double t1 = get_current_cpu_time();
        accumulated_cpu_time += (t1 - t0);
#endif
        is_done_flag = true;
        if (sat) {
            status_val = STATUS_SAT;
#ifndef DISABLE_TELEMETRY
            finalize_sat_model();
#endif
        } else {
            status_val = STATUS_UNSAT;
        }
        return status_val;
    }

    int solve_continuous_timeout(double timeout_sec) {
#ifndef DISABLE_TELEMETRY
        double t0 = get_current_cpu_time();
#endif
        if (!with_restarts_flag) {
            restart_first = INT32_MAX;
        }
        clearInterrupt();
        std::atomic<bool> finished(false);
        std::thread watchdog([this, &finished, timeout_sec]() {
            auto start = std::chrono::steady_clock::now();
            while (!finished.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= timeout_sec) {
                    this->interrupt();
                    break;
                }
            }
        });

        vec<Lit> dummy;
        lbool res = solveLimited(dummy);
        finished.store(true);
        if (watchdog.joinable()) watchdog.join();
        clearInterrupt();

#ifndef DISABLE_TELEMETRY
        double t1 = get_current_cpu_time();
        accumulated_cpu_time += (t1 - t0);
#endif
        is_done_flag = true;
        if (res == l_True) {
            status_val = STATUS_SAT;
#ifndef DISABLE_TELEMETRY
            finalize_sat_model();
#endif
        } else if (res == l_False) {
            status_val = STATUS_UNSAT;
        } else {
            status_val = STATUS_CAP_EXCEEDED;
        }
        return status_val;
    }

    int init_stepping(bool restarts, int decision_cap) {
        with_restarts_flag = restarts;
        max_decision_cap_val = decision_cap;
        current_decisions_count = 0;
        stepping_ready = true;

        if (!okay()) {
            status_val = STATUS_UNSAT;
            is_done_flag = true;
            return STATUS_UNSAT;
        }

        // Run level 0 propagation
        CRef confl = propagate();
        if (confl != CRef_Undef) {
            status_val = STATUS_UNSAT;
            is_done_flag = true;
            return STATUS_UNSAT;
        }

        // Check if all variables already assigned
        if (nAssigns() == nVars()) {
            finalize_sat_model();
            status_val = STATUS_SAT;
            is_done_flag = true;
            return STATUS_SAT;
        }

        Lit next = pickBranchLit();
        if (next == lit_Undef) {
            finalize_sat_model();
            status_val = STATUS_SAT;
            is_done_flag = true;
            return STATUS_SAT;
        } else {
            insertVarOrder(var(next));
        }

        is_done_flag = false;
        status_val = STATUS_UNDEF;
        return STATUS_UNDEF;
    }

    Lit pick_vsids() {
        Lit p = pickBranchLit();
        if (p != lit_Undef) {
            insertVarOrder(var(p));
        }
        return p;
    }

    int step_decision(int decision_lit) {
        if (is_done_flag) return status_val;

        if (max_decision_cap_val > 0 && current_decisions_count >= max_decision_cap_val) {
            status_val = STATUS_CAP_EXCEEDED;
            is_done_flag = true;
            return STATUS_CAP_EXCEEDED;
        }

#ifndef DISABLE_TELEMETRY
        double t0 = get_current_cpu_time();
        last_selected_lit = lit_Undef;   // reset before each step
#endif
        Lit branch_lit;
        if (decision_lit == 32767 || decision_lit == 0) {
            branch_lit = pickBranchLit();
            if (branch_lit == lit_Undef) {
#ifndef DISABLE_TELEMETRY
                finalize_sat_model();
                accumulated_cpu_time += (get_current_cpu_time() - t0);
#endif
                status_val = STATUS_SAT;
                is_done_flag = true;
                return STATUS_SAT;
            }
        } else {
            Var v = std::abs(decision_lit) - 1;
            bool sign = (decision_lit < 0);
            branch_lit = mkLit(v, sign);
        }
#ifndef DISABLE_TELEMETRY
        last_selected_lit = branch_lit;   // record BEFORE enqueue so it's always set
#endif

        // Enqueue branch
        decisions++;
        current_decisions_count++;
        newDecisionLevel();
        uncheckedEnqueue(branch_lit);

        // CDCL propagation & conflict resolution loop
        while (true) {
            CRef confl = propagate();
            if (confl != CRef_Undef) {
                conflicts++;
                if (decisionLevel() == 0) {
                    status_val = STATUS_UNSAT;
                    is_done_flag = true;
#ifndef DISABLE_TELEMETRY
                    accumulated_cpu_time += (get_current_cpu_time() - t0);
#endif
                    return STATUS_UNSAT;
                }

                vec<Lit> learnt_clause;
                int backtrack_level;
                analyze(confl, learnt_clause, backtrack_level);
                cancelUntil(backtrack_level);

                if (learnt_clause.size() == 1) {
                    uncheckedEnqueue(learnt_clause[0]);
                } else {
                    CRef cr = ca.alloc(learnt_clause, true);
                    learnts.push(cr);
                    attachClause(cr);
                    claBumpActivity(ca[cr]);
                    uncheckedEnqueue(learnt_clause[0], cr);
                }

                varDecayActivity();
                claDecayActivity();

                if (--learntsize_adjust_cnt == 0) {
                    learntsize_adjust_confl *= learntsize_adjust_inc;
                    learntsize_adjust_cnt = (int)learntsize_adjust_confl;
                    max_learnts *= learntsize_inc;
                }
            } else {
                // No conflict
                if (max_decision_cap_val > 0 && current_decisions_count >= max_decision_cap_val) {
                    status_val = STATUS_CAP_EXCEEDED;
                    is_done_flag = true;
#ifndef DISABLE_TELEMETRY
                    accumulated_cpu_time += (get_current_cpu_time() - t0);
#endif
                    return STATUS_CAP_EXCEEDED;
                }

                if (nAssigns() == nVars()) {
#ifndef DISABLE_TELEMETRY
                    finalize_sat_model();
                    accumulated_cpu_time += (get_current_cpu_time() - t0);
#endif
                    status_val = STATUS_SAT;
                    is_done_flag = true;
                    return STATUS_SAT;
                }

                Lit next_pick = pickBranchLit();
                if (next_pick == lit_Undef) {
#ifndef DISABLE_TELEMETRY
                    finalize_sat_model();
                    accumulated_cpu_time += (get_current_cpu_time() - t0);
#endif
                    status_val = STATUS_SAT;
                    is_done_flag = true;
                    return STATUS_SAT;
                } else {
                    insertVarOrder(var(next_pick));
                }

                status_val = STATUS_UNDEF;
#ifndef DISABLE_TELEMETRY
                accumulated_cpu_time += (get_current_cpu_time() - t0);
#endif
                return STATUS_UNDEF;
            }
        }
    }

    int get_dec_level() const { return decisionLevel(); }
    double get_var_activity(Var v) const { return (v >= 0 && v < activity.size()) ? activity[v] : 0.0; }

    int get_active_clauses_info(bool include_learnts, int* out_num_clauses, int* out_total_literals) {
        if (!out_num_clauses || !out_total_literals) return -1;
        int num_cl = 0;
        int tot_lits = 0;

        for (int j = 0; j < clauses.size(); j++) {
            Clause& c = ca[clauses[j]];
            if (!satisfied(c)) {
                int cl_lits = 0;
                for (int i = 0; i < c.size(); i++) {
                    if (value(c[i]) != l_False) {
                        cl_lits++;
                    }
                }
                num_cl++;
                tot_lits += cl_lits;
            }
        }

        if (include_learnts) {
            for (int j = 0; j < learnts.size(); j++) {
                Clause& c = ca[learnts[j]];
                if (!satisfied(c)) {
                    int cl_lits = 0;
                    for (int i = 0; i < c.size(); i++) {
                        if (value(c[i]) != l_False) {
                            cl_lits++;
                        }
                    }
                    num_cl++;
                    tot_lits += cl_lits;
                }
            }
        }

        *out_num_clauses = num_cl;
        *out_total_literals = tot_lits;
        return 0;
    }

    int get_active_clauses(bool include_learnts, int* out_flat_literals, int* out_clause_lengths, int max_literals, int max_clauses) {
        if (!out_flat_literals || !out_clause_lengths) return -1;
        int cl_idx = 0;
        int lit_idx = 0;

        for (int j = 0; j < clauses.size(); j++) {
            if (cl_idx >= max_clauses) break;
            Clause& c = ca[clauses[j]];
            if (!satisfied(c)) {
                int cl_len = 0;
                for (int i = 0; i < c.size(); i++) {
                    if (value(c[i]) != l_False) {
                        if (lit_idx < max_literals) {
                            int signed_lit = sign(c[i]) ? (-(var(c[i]) + 1)) : (var(c[i]) + 1);
                            out_flat_literals[lit_idx++] = signed_lit;
                            cl_len++;
                        }
                    }
                }
                out_clause_lengths[cl_idx++] = cl_len;
            }
        }

        if (include_learnts) {
            for (int j = 0; j < learnts.size(); j++) {
                if (cl_idx >= max_clauses) break;
                Clause& c = ca[learnts[j]];
                if (!satisfied(c)) {
                    int cl_len = 0;
                    for (int i = 0; i < c.size(); i++) {
                        if (value(c[i]) != l_False) {
                            if (lit_idx < max_literals) {
                                int signed_lit = sign(c[i]) ? (-(var(c[i]) + 1)) : (var(c[i]) + 1);
                                out_flat_literals[lit_idx++] = signed_lit;
                                cl_len++;
                            }
                        }
                    }
                    out_clause_lengths[cl_idx++] = cl_len;
                }
            }
        }

        return cl_idx;
    }

    void set_initial_activities(const double* act_arr, int n_vars) {
        int n = nVars();
        int count = (n < n_vars) ? n : n_vars;
        for (int i = 0; i < count; i++) {
            activity[i] = act_arr[i];
        }
        rebuildOrderHeap();
    }

    void set_initial_polarities(const int* pol_arr, int n_vars) {
        int n = nVars();
        int count = (n < n_vars) ? n : n_vars;
        for (int i = 0; i < count; i++) {
            // pol_arr[i]: 1 for true (sign = false / polarity[i] = 0 in MiniSat),
            // 0 for false (sign = true / polarity[i] = 1 in MiniSat)
            polarity[i] = (pol_arr[i] > 0) ? 0 : 1;
        }
    }

    // Milestone 2C RLAF Dynamic Guidance (Tönshoff & Grohe ICLR 2026 Eq 20)
    void set_rlaf_guidance(const double* weights_arr, const int* pol_arr, int n_vars) {
        int n = nVars();
        int count = (n < n_vars) ? n : n_vars;
        rlaf_guidance_active = true;
        rlaf_weights.growTo(n, 1.0);
        for (int i = 0; i < count; i++) {
            rlaf_weights[i] = weights_arr[i];
            activity[i] = weights_arr[i] * 1.0;
            if (pol_arr) {
                polarity[i] = (pol_arr[i] > 0) ? 0 : 1;
            }
        }
        rebuildOrderHeap();
    }

    void clear_rlaf_guidance() {
        rlaf_guidance_active = false;
        rlaf_weights.clear();
    }

    int get_native_gnn_guidance(double* out_weights, int* out_polarities, int n_vars_arg, double* out_gnn_time_ms) {
        if (!Minisat::get_global_gnn_engine().is_loaded()) {
            return -1;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        int n = nVars();
        if (n_vars_arg < n) {
            return -2;
        }

        // 1. Identify unassigned variables using persistent scratch buffer
        scratch_unassigned_vars.clear();
        scratch_unassigned_vars.reserve(n);

        if ((int)scratch_var_remapping.size() < n) {
            scratch_var_remapping.resize(n, -1);
        }
        for (int i = 0; i < n; ++i) {
            scratch_var_remapping[i] = -1;
        }

        for (int i = 0; i < n; ++i) {
            if (value(i) == l_Undef) {
                scratch_var_remapping[i] = (int)scratch_unassigned_vars.size();
                scratch_unassigned_vars.push_back(i);
            }
        }
        int num_vars = (int)scratch_unassigned_vars.size();
        if (num_vars == 0) {
            auto t1 = std::chrono::high_resolution_clock::now();
            if (out_gnn_time_ms) *out_gnn_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return 0;
        }

        // 2. Identify active unsatisfied clauses and construct bipartite edges
        scratch_edge_row.clear();
        scratch_edge_col.clear();
        scratch_edge_is_pos.clear();

        scratch_edge_row.reserve(clauses.size() * 6);
        scratch_edge_col.reserve(clauses.size() * 6);
        scratch_edge_is_pos.reserve(clauses.size() * 6);

        int num_clauses = 0;
        for (int j = 0; j < clauses.size(); ++j) {
            Clause& c = ca[clauses[j]];
            if (!satisfied(c)) {
                int clause_node_id = num_vars + num_clauses;
                for (int i = 0; i < c.size(); ++i) {
                    if (value(c[i]) != l_False) {
                        int orig_v = var(c[i]);
                        if (orig_v >= 0 && orig_v < n && scratch_var_remapping[orig_v] != -1) {
                            int local_v = scratch_var_remapping[orig_v];
                            uint8_t is_pos = sign(c[i]) ? 0 : 1;

                            // Edge 1: local_v -> clause_node_id
                            scratch_edge_row.push_back(local_v);
                            scratch_edge_col.push_back(clause_node_id);
                            scratch_edge_is_pos.push_back(is_pos);

                            // Edge 2: clause_node_id -> local_v
                            scratch_edge_row.push_back(clause_node_id);
                            scratch_edge_col.push_back(local_v);
                            scratch_edge_is_pos.push_back(is_pos);
                        }
                    }
                }
                num_clauses++;
            }
        }

        int num_edges = (int)scratch_edge_row.size();

        if ((int)scratch_local_weights.size() < num_vars) {
            scratch_local_weights.resize(num_vars, 1.0);
            scratch_local_polarities.resize(num_vars, 0);
        }

        bool ok = Minisat::get_global_gnn_engine().forward(
            num_vars,
            num_clauses,
            num_edges,
            scratch_edge_row.data(),
            scratch_edge_col.data(),
            scratch_edge_is_pos.data(),
            scratch_local_weights.data(),
            scratch_local_polarities.data()
        );

        if (!ok) {
            return -3;
        }

        // Map back to solver variable indexing
        for (int i = 0; i < n; ++i) {
            out_weights[i] = 1.0;
            out_polarities[i] = 0;
        }
        for (int lv = 0; lv < num_vars; ++lv) {
            int orig_v = scratch_unassigned_vars[lv];
            out_weights[orig_v] = scratch_local_weights[lv];
            out_polarities[orig_v] = scratch_local_polarities[lv];
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (out_gnn_time_ms) {
            *out_gnn_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        return 0;
    }

    int solve_with_native_gnn(double* out_gnn_time_ms) {
        int n = nVars();
        if ((int)scratch_solve_w.size() < n) {
            scratch_solve_w.resize(n, 1.0);
            scratch_solve_p.resize(n, 0);
        }

        int rc = get_native_gnn_guidance(scratch_solve_w.data(), scratch_solve_p.data(), n, out_gnn_time_ms);
        if (rc != 0) {
            return solve_continuous();
        }

        set_rlaf_guidance(scratch_solve_w.data(), scratch_solve_p.data(), n);
        return solve_continuous();
    }
};

extern "C" {

ADAPTER_API void* minisat_create() {
    return new InteractiveSolver();
}

ADAPTER_API void minisat_destroy(void* solver_ptr) {
    if (solver_ptr) {
        delete static_cast<InteractiveSolver*>(solver_ptr);
    }
}

ADAPTER_API void minisat_set_verbosity(void* solver_ptr, int verbosity) {
    if (solver_ptr) {
        static_cast<InteractiveSolver*>(solver_ptr)->verbosity = verbosity;
    }
}

ADAPTER_API void minisat_set_random_seed(void* solver_ptr, double seed) {
    if (solver_ptr) {
        static_cast<InteractiveSolver*>(solver_ptr)->random_seed = seed;
    }
}

ADAPTER_API void minisat_set_restart_policy(void* solver_ptr, bool with_restarts) {
    if (solver_ptr) {
        static_cast<InteractiveSolver*>(solver_ptr)->with_restarts_flag = with_restarts;
    }
}

ADAPTER_API int minisat_load_dimacs(void* solver_ptr, const char* filepath, bool simplify) {
    if (!solver_ptr || !filepath) return -1;
    return static_cast<InteractiveSolver*>(solver_ptr)->load_dimacs_file(filepath, simplify);
}

ADAPTER_API int minisat_solve(void* solver_ptr) {
    if (!solver_ptr) return STATUS_ERROR;
    return static_cast<InteractiveSolver*>(solver_ptr)->solve_continuous();
}

ADAPTER_API int minisat_init_stepping(void* solver_ptr, bool with_restarts, int max_decision_cap) {
    if (!solver_ptr) return STATUS_ERROR;
    return static_cast<InteractiveSolver*>(solver_ptr)->init_stepping(with_restarts, max_decision_cap);
}

ADAPTER_API int minisat_step(void* solver_ptr, int decision_lit) {
    if (!solver_ptr) return STATUS_ERROR;
    return static_cast<InteractiveSolver*>(solver_ptr)->step_decision(decision_lit);
}

ADAPTER_API int minisat_pick_vsids_literal(void* solver_ptr) {
    if (!solver_ptr) return 0;
    Lit l = static_cast<InteractiveSolver*>(solver_ptr)->pick_vsids();
    if (l == lit_Undef) return 0;
    int var_1based = var(l) + 1;
    return sign(l) ? -var_1based : var_1based;
}

ADAPTER_API int minisat_get_last_branch_lit(void* solver_ptr) {
#ifdef DISABLE_TELEMETRY
    return 0;
#else
    // Returns the signed 1-based literal that was actually branched on in the most
    // recent call to minisat_step().  Returns 0 if no step has been performed yet
    // or if the solver terminated without making a decision in the last step.
    if (!solver_ptr) return 0;
    Lit l = static_cast<InteractiveSolver*>(solver_ptr)->last_selected_lit;
    if (l == lit_Undef) return 0;
    int var_1based = var(l) + 1;
    return sign(l) ? -var_1based : var_1based;
#endif
}

ADAPTER_API int minisat_get_num_vars(void* solver_ptr) {
    if (!solver_ptr) return 0;
    return static_cast<InteractiveSolver*>(solver_ptr)->nVars();
}

ADAPTER_API int minisat_get_num_clauses(void* solver_ptr) {
    if (!solver_ptr) return 0;
    return static_cast<InteractiveSolver*>(solver_ptr)->nClauses();
}

ADAPTER_API int minisat_get_num_learnts(void* solver_ptr) {
    if (!solver_ptr) return 0;
    return static_cast<InteractiveSolver*>(solver_ptr)->nLearnts();
}

ADAPTER_API int minisat_get_decision_level(void* solver_ptr) {
    if (!solver_ptr) return 0;
    return static_cast<InteractiveSolver*>(solver_ptr)->get_dec_level();
}

ADAPTER_API int minisat_get_status(void* solver_ptr) {
    if (!solver_ptr) return STATUS_ERROR;
    return static_cast<InteractiveSolver*>(solver_ptr)->status_val;
}

ADAPTER_API bool minisat_is_done(void* solver_ptr) {
    if (!solver_ptr) return true;
    return static_cast<InteractiveSolver*>(solver_ptr)->is_done_flag;
}

ADAPTER_API void minisat_get_stats(void* solver_ptr, SolverStatsDTO* out_stats) {
    if (!solver_ptr || !out_stats) return;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    out_stats->decisions    = s->decisions;
    out_stats->conflicts    = s->conflicts;
    out_stats->propagations = s->propagations;
    out_stats->restarts     = (s->starts > 0) ? (s->starts - 1) : 0;
    out_stats->num_clauses  = s->nClauses();
    out_stats->num_learnts  = s->nLearnts();
#ifdef DISABLE_TELEMETRY
    out_stats->cpu_time     = 0.0;
#else
    out_stats->cpu_time     = s->accumulated_cpu_time;
#endif
}

ADAPTER_API int minisat_get_assignment(void* solver_ptr, int var_id) {
    if (!solver_ptr) return -1;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    if (var_id < 0 || var_id >= s->nVars()) return -1;
    lbool val = s->value(var_id);
    if (val == l_True) return 1;
    if (val == l_False) return 0;
    return -1;
}

ADAPTER_API int minisat_get_assignments(void* solver_ptr, int* out_arr, int max_n) {
    if (!solver_ptr || !out_arr) return 0;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int n = s->nVars();
    int count = (n < max_n) ? n : max_n;
    for (int i = 0; i < count; i++) {
        lbool val = s->value(i);
        if (val == l_True) out_arr[i] = 1;
        else if (val == l_False) out_arr[i] = 0;
        else out_arr[i] = -1;
    }
    return count;
}

ADAPTER_API int minisat_get_activities(void* solver_ptr, double* out_arr, int max_n) {
    if (!solver_ptr || !out_arr) return 0;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int n = s->nVars();
    int count = (n < max_n) ? n : max_n;
    for (int i = 0; i < count; i++) {
        out_arr[i] = s->get_var_activity(i);
    }
    return count;
}

ADAPTER_API int minisat_get_model(void* solver_ptr, int* out_arr, int max_n) {
    if (!solver_ptr || !out_arr) return 0;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    if (s->status_val != STATUS_SAT) return 0;
    int n = s->nVars();
    int count = (n < max_n) ? n : max_n;
    for (int i = 0; i < count; i++) {
        lbool val = (s->model.size() > i) ? s->model[i] : s->value(i);
        int lit_val = (i + 1);
        if (val == l_False) lit_val = -lit_val;
        out_arr[i] = lit_val;
    }
    return count;
}

ADAPTER_API int minisat_get_active_clauses_info(void* solver_ptr, bool include_learnts, int* out_num_clauses, int* out_total_literals) {
    if (!solver_ptr) return -1;
    return static_cast<InteractiveSolver*>(solver_ptr)->get_active_clauses_info(include_learnts, out_num_clauses, out_total_literals);
}

ADAPTER_API int minisat_get_active_clauses(void* solver_ptr, bool include_learnts, int* out_flat_literals, int* out_clause_lengths, int max_literals, int max_clauses) {
    if (!solver_ptr) return -1;
    return static_cast<InteractiveSolver*>(solver_ptr)->get_active_clauses(include_learnts, out_flat_literals, out_clause_lengths, max_literals, max_clauses);
}

ADAPTER_API int minisat_set_initial_activities(void* solver_ptr, const double* activities, int n_vars) {
    if (!solver_ptr || !activities) return -1;
    static_cast<InteractiveSolver*>(solver_ptr)->set_initial_activities(activities, n_vars);
    return 0;
}

ADAPTER_API int minisat_set_initial_polarities(void* solver_ptr, const int* polarities, int n_vars) {
    if (!solver_ptr || !polarities) return -1;
    static_cast<InteractiveSolver*>(solver_ptr)->set_initial_polarities(polarities, n_vars);
    return 0;
}

ADAPTER_API int minisat_set_decision_vars(void* solver_ptr, const int* decision_flags, int n_vars) {
    if (!solver_ptr || !decision_flags) return -1;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int n = s->nVars();
    int count = (n < n_vars) ? n : n_vars;
    for (int i = 0; i < count; i++) {
        s->setDecisionVar(i, decision_flags[i] > 0);
    }
    return 0;
}

ADAPTER_API int minisat_set_rlaf_guidance(void* solver_ptr, const double* weights, const int* polarities, int n_vars) {
    if (!solver_ptr || !weights) return -1;
    static_cast<InteractiveSolver*>(solver_ptr)->set_rlaf_guidance(weights, polarities, n_vars);
    return 0;
}

ADAPTER_API int minisat_clear_rlaf_guidance(void* solver_ptr) {
    if (!solver_ptr) return -1;
    static_cast<InteractiveSolver*>(solver_ptr)->clear_rlaf_guidance();
    return 0;
}

ADAPTER_API int minisat_init_native_gnn(const char* weights_path) {
    if (!weights_path) return -1;
    return Minisat::get_global_gnn_engine().load_weights(weights_path) ? 0 : -1;
}

ADAPTER_API int minisat_is_native_gnn_ready() {
    return Minisat::get_global_gnn_engine().is_loaded() ? 1 : 0;
}

ADAPTER_API int minisat_get_native_gnn_guidance(void* solver_ptr, double* out_weights, int* out_polarities, int n_vars, double* out_gnn_time_ms) {
    if (!solver_ptr || !out_weights || !out_polarities) return -1;
    return static_cast<InteractiveSolver*>(solver_ptr)->get_native_gnn_guidance(out_weights, out_polarities, n_vars, out_gnn_time_ms);
}

ADAPTER_API int minisat_solve_with_native_gnn(void* solver_ptr, double* out_gnn_time_ms) {
    if (!solver_ptr) return STATUS_ERROR;
    return static_cast<InteractiveSolver*>(solver_ptr)->solve_with_native_gnn(out_gnn_time_ms);
}

// Global instance of Structural MLP Engine
static StructuralMLPEngine g_smlp_engine;

ADAPTER_API int minisat_init_structural_mlp(const char* weights_path) {
    if (!weights_path) return -1;
    return g_smlp_engine.load_weights(weights_path) ? 0 : -1;
}

ADAPTER_API int minisat_is_structural_mlp_ready() {
    return g_smlp_engine.is_loaded() ? 1 : 0;
}

ADAPTER_API int minisat_get_structural_mlp_guidance(void* solver_ptr, double* out_weights, int* out_polarities, int n_vars, double* out_gnn_time_ms) {
    if (!solver_ptr || !out_weights || !out_polarities || !g_smlp_engine.is_loaded()) return -1;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int num_clauses = 0, total_lits = 0;
    s->get_active_clauses_info(false, &num_clauses, &total_lits);
    
    std::vector<int> flat_lits(total_lits);
    std::vector<int> clause_lens(num_clauses);
    s->get_active_clauses(false, flat_lits.data(), clause_lens.data(), total_lits, num_clauses);

    std::vector<int> edge_row;
    std::vector<int> edge_col;
    std::vector<uint8_t> edge_pos;
    edge_row.reserve(total_lits);
    edge_col.reserve(total_lits);
    edge_pos.reserve(total_lits);

    int act_vars = s->nVars();
    int lit_idx = 0;
    for (int c = 0; c < num_clauses; ++c) {
        int clen = clause_lens[c];
        int c_node = act_vars + c;
        for (int k = 0; k < clen; ++k) {
            int lit = flat_lits[lit_idx++];
            edge_row.push_back(std::abs(lit) - 1);
            edge_col.push_back(c_node);
            edge_pos.push_back((lit > 0) ? 1 : 0);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = g_smlp_engine.forward(act_vars, num_clauses, (int)edge_row.size(),
                                    edge_row.data(), edge_col.data(), edge_pos.data(),
                                    out_weights, out_polarities);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (out_gnn_time_ms) {
        *out_gnn_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return ok ? 0 : -1;
}

ADAPTER_API int minisat_solve_with_structural_mlp(void* solver_ptr, double* out_gnn_time_ms) {
    if (!solver_ptr) return STATUS_ERROR;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int n_vars = s->nVars();
    std::vector<double> weights(n_vars, 1.0);
    std::vector<int> polarities(n_vars, 0);
    
    int gnn_res = minisat_get_structural_mlp_guidance(solver_ptr, weights.data(), polarities.data(), n_vars, out_gnn_time_ms);
    if (gnn_res == 0) {
        s->set_rlaf_guidance(weights.data(), polarities.data(), n_vars);
    }
    return s->solve_continuous();
}

ADAPTER_API int minisat_solve_with_new_native_engine(void* solver_ptr, const double* weights, const int* polarities, int n_vars, double* out_solver_time_ms) {
    if (!solver_ptr) return STATUS_ERROR;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int act_vars = s->nVars();

    int num_clauses = 0, total_lits = 0;
    s->get_active_clauses_info(false, &num_clauses, &total_lits);

    if (num_clauses == 0) {
        if (out_solver_time_ms) *out_solver_time_ms = 0.01;
        s->decisions = 0;
        s->conflicts = 0;
        s->propagations = 0;
        s->status_val = STATUS_SAT;
        s->is_done_flag = true;
        s->model.growTo(act_vars);
        for (int i = 0; i < act_vars; ++i) s->model[i] = l_True;
        s->extend_model();
        return STATUS_SAT;
    }

    std::vector<int> flat_lits(total_lits);
    std::vector<int> clause_lens(num_clauses);
    s->get_active_clauses(false, flat_lits.data(), clause_lens.data(), total_lits, num_clauses);

    ProductionCDCL::FlatCDCLProductionEngine engine;
    engine.init(act_vars, num_clauses);

    std::vector<uint8_t> var_in_clause(act_vars, 0);
    int lit_idx = 0;
    for (int c = 0; c < num_clauses; ++c) {
        int clen = clause_lens[c];
        std::vector<int32_t> cl;
        cl.reserve(clen);
        for (int k = 0; k < clen; ++k) {
            int lit = flat_lits[lit_idx++];
            int v = std::abs(lit) - 1;
            if (v < act_vars) var_in_clause[v] = 1;
            bool sign = (lit < 0);
            cl.push_back(ProductionCDCL::L1Core::FlatL1Engine::mkLit(v, sign));
        }
        engine.addClause(cl);
    }

    std::vector<uint8_t> is_decision(act_vars, 0);
    for (int i = 0; i < act_vars; ++i) {
        if (s->isDecisionVar(i) && var_in_clause[i]) {
            is_decision[i] = 1;
        }
    }
    engine.setDecisionVars(is_decision.data());

    if (weights) {
        engine.setGuidance(weights, polarities, act_vars);
    }

    auto ts0 = std::chrono::high_resolution_clock::now();
    int res = engine.solve();
    auto ts1 = std::chrono::high_resolution_clock::now();
    if (out_solver_time_ms) {
        *out_solver_time_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    }

    s->decisions = engine.getDecisions();
    s->conflicts = engine.getConflicts();
    s->propagations = engine.getPropagations();
    s->status_val = (res == 10) ? STATUS_SAT : (res == 20 ? STATUS_UNSAT : STATUS_UNDEF);
    s->is_done_flag = true;

    if (res == 10) {
        s->model.growTo(act_vars);
        for (int i = 0; i < act_vars; ++i) {
            uint8_t v = engine.getModelLitVal(ProductionCDCL::L1Core::FlatL1Engine::mkLit(i, false));
            s->model[i] = (v == ProductionCDCL::VAL_TRUE) ? l_True : (v == ProductionCDCL::VAL_FALSE ? l_False : l_Undef);
        }
        s->extend_model();
    }

    return s->status_val;
}

ADAPTER_API int minisat_solve_m3_5_unified(void* solver_ptr, double* out_gnn_time_ms, double* out_solver_time_ms) {
    if (!solver_ptr) return STATUS_ERROR;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int act_vars = s->nVars();

    int num_clauses = 0, total_lits = 0;
    s->get_active_clauses_info(false, &num_clauses, &total_lits);

    std::vector<int> flat_lits(total_lits);
    std::vector<int> clause_lens(num_clauses);
    s->get_active_clauses(false, flat_lits.data(), clause_lens.data(), total_lits, num_clauses);

    // Single-pass: simultaneously construct bipartite graph edges AND production engine clauses
    std::vector<int> edge_row;
    std::vector<int> edge_col;
    std::vector<uint8_t> edge_pos;
    edge_row.reserve(total_lits);
    edge_col.reserve(total_lits);
    edge_pos.reserve(total_lits);

    ProductionCDCL::FlatCDCLProductionEngine engine;
    engine.init(act_vars, num_clauses);

    int lit_idx = 0;
    for (int c = 0; c < num_clauses; ++c) {
        int clen = clause_lens[c];
        int c_node = act_vars + c;
        std::vector<int32_t> cl;
        cl.reserve(clen);
        for (int k = 0; k < clen; ++k) {
            int lit = flat_lits[lit_idx++];
            int v = std::abs(lit) - 1;
            bool sign = (lit < 0);
            cl.push_back(ProductionCDCL::L1Core::FlatL1Engine::mkLit(v, sign));

            edge_row.push_back(v);
            edge_col.push_back(c_node);
            edge_pos.push_back(lit > 0 ? 1 : 0);
        }
        engine.addClause(cl);
    }

    std::vector<double> weights(act_vars, 1.0);
    std::vector<int> polarities(act_vars, 0);

    auto tg0 = std::chrono::high_resolution_clock::now();
    bool gnn_ok = g_smlp_engine.is_loaded() && 
                  g_smlp_engine.forward(act_vars, num_clauses, (int)edge_row.size(),
                                        edge_row.data(), edge_col.data(), edge_pos.data(),
                                        weights.data(), polarities.data());
    auto tg1 = std::chrono::high_resolution_clock::now();
    if (out_gnn_time_ms) {
        *out_gnn_time_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    }

    if (gnn_ok) {
        engine.setGuidance(weights.data(), polarities.data(), act_vars);
    }

    auto ts0 = std::chrono::high_resolution_clock::now();
    int res = engine.solve();
    auto ts1 = std::chrono::high_resolution_clock::now();
    if (out_solver_time_ms) {
        *out_solver_time_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    }

    s->decisions = engine.getDecisions();
    s->conflicts = engine.getConflicts();
    s->propagations = engine.getPropagations();
    s->status_val = (res == 10) ? STATUS_SAT : (res == 20 ? STATUS_UNSAT : STATUS_UNDEF);
    s->is_done_flag = true;

    if (res == 10) {
        s->model.growTo(act_vars);
        for (int i = 0; i < act_vars; ++i) {
            uint8_t v = engine.getModelLitVal(ProductionCDCL::L1Core::FlatL1Engine::mkLit(i, false));
            s->model[i] = (v == ProductionCDCL::VAL_TRUE) ? l_True : (v == ProductionCDCL::VAL_FALSE ? l_False : l_Undef);
        }
    }

    return s->status_val;
}

ADAPTER_API int minisat_solve_with_timeout(void* solver_ptr, double timeout_sec) {
    if (!solver_ptr) return STATUS_ERROR;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    if (timeout_sec <= 0.0) {
        return s->solve_continuous();
    }
    return s->solve_continuous_timeout(timeout_sec);
}

ADAPTER_API int minisat_solve_with_structural_mlp_timeout(void* solver_ptr, double timeout_sec, double* out_gnn_time_ms) {
    if (!solver_ptr) return STATUS_ERROR;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int n_vars = s->nVars();
    std::vector<double> weights(n_vars, 1.0);
    std::vector<int> polarities(n_vars, 0);
    
    int gnn_res = minisat_get_structural_mlp_guidance(solver_ptr, weights.data(), polarities.data(), n_vars, out_gnn_time_ms);
    if (gnn_res == 0) {
        s->set_rlaf_guidance(weights.data(), polarities.data(), n_vars);
    }
    if (timeout_sec <= 0.0) {
        return s->solve_continuous();
    }
    return s->solve_continuous_timeout(timeout_sec);
}

ADAPTER_API int minisat_solve_m3_5_unified_timeout(void* solver_ptr, double timeout_sec, double* out_gnn_time_ms, double* out_solver_time_ms) {
    if (!solver_ptr) return STATUS_ERROR;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int act_vars = s->nVars();

    int num_clauses = 0, total_lits = 0;
    s->get_active_clauses_info(false, &num_clauses, &total_lits);

    std::vector<int> flat_lits(total_lits);
    std::vector<int> clause_lens(num_clauses);
    s->get_active_clauses(false, flat_lits.data(), clause_lens.data(), total_lits, num_clauses);

    // Single-pass: simultaneously construct bipartite graph edges AND production engine clauses
    std::vector<int> edge_row;
    std::vector<int> edge_col;
    std::vector<uint8_t> edge_pos;
    edge_row.reserve(total_lits);
    edge_col.reserve(total_lits);
    edge_pos.reserve(total_lits);

    ProductionCDCL::FlatCDCLProductionEngine engine;
    engine.init(act_vars, num_clauses);

    int lit_idx = 0;
    for (int c = 0; c < num_clauses; ++c) {
        int clen = clause_lens[c];
        int c_node = act_vars + c;
        std::vector<int32_t> cl;
        cl.reserve(clen);
        for (int k = 0; k < clen; ++k) {
            int lit = flat_lits[lit_idx++];
            int v = std::abs(lit) - 1;
            bool sign = (lit < 0);
            cl.push_back(ProductionCDCL::L1Core::FlatL1Engine::mkLit(v, sign));

            edge_row.push_back(v);
            edge_col.push_back(c_node);
            edge_pos.push_back(lit > 0 ? 1 : 0);
        }
        engine.addClause(cl);
    }

    std::vector<double> weights(act_vars, 1.0);
    std::vector<int> polarities(act_vars, 0);

    auto tg0 = std::chrono::high_resolution_clock::now();
    bool gnn_ok = g_smlp_engine.is_loaded() && 
                  g_smlp_engine.forward(act_vars, num_clauses, (int)edge_row.size(),
                                        edge_row.data(), edge_col.data(), edge_pos.data(),
                                        weights.data(), polarities.data());
    auto tg1 = std::chrono::high_resolution_clock::now();
    if (out_gnn_time_ms) {
        *out_gnn_time_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    }

    if (gnn_ok) {
        engine.setGuidance(weights.data(), polarities.data(), act_vars);
    }

    auto ts0 = std::chrono::high_resolution_clock::now();
    int res = engine.solve(timeout_sec);
    auto ts1 = std::chrono::high_resolution_clock::now();
    if (out_solver_time_ms) {
        *out_solver_time_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    }

    s->decisions = engine.getDecisions();
    s->conflicts = engine.getConflicts();
    s->propagations = engine.getPropagations();
    s->status_val = (res == 10) ? STATUS_SAT : (res == 20 ? STATUS_UNSAT : STATUS_CAP_EXCEEDED);
    s->is_done_flag = true;

    if (res == 10) {
        s->model.growTo(act_vars);
        for (int i = 0; i < act_vars; ++i) {
            uint8_t v = engine.getModelLitVal(ProductionCDCL::L1Core::FlatL1Engine::mkLit(i, false));
            s->model[i] = (v == ProductionCDCL::VAL_TRUE) ? l_True : (v == ProductionCDCL::VAL_FALSE ? l_False : l_Undef);
        }
    }

    return s->status_val;
}

ADAPTER_API int minisat_solve_m4_controller(
    void* solver_ptr,
    int controller_mode,
    int k_param,
    double turb_param,
    double conf_margin,
    double timeout_sec,
    double* out_gnn_time_ms,
    double* out_solver_time_ms,
    int64_t* out_refreshes
) {
    if (!solver_ptr) return STATUS_ERROR;
    InteractiveSolver* s = static_cast<InteractiveSolver*>(solver_ptr);
    int act_vars = s->nVars();

    int num_clauses = 0, total_lits = 0;
    s->get_active_clauses_info(false, &num_clauses, &total_lits);

    std::vector<int> flat_lits(total_lits);
    std::vector<int> clause_lens(num_clauses);
    s->get_active_clauses(false, flat_lits.data(), clause_lens.data(), total_lits, num_clauses);

    std::vector<int> edge_row;
    std::vector<int> edge_col;
    std::vector<uint8_t> edge_pos;
    edge_row.reserve(total_lits);
    edge_col.reserve(total_lits);
    edge_pos.reserve(total_lits);

    ProductionCDCL::FlatCDCLProductionEngine engine;
    engine.init(act_vars, num_clauses);

    int lit_idx = 0;
    for (int c = 0; c < num_clauses; ++c) {
        int clen = clause_lens[c];
        int c_node = act_vars + c;
        std::vector<int32_t> cl;
        cl.reserve(clen);
        for (int k = 0; k < clen; ++k) {
            int lit = flat_lits[lit_idx++];
            int v = std::abs(lit) - 1;
            bool sign = (lit < 0);
            cl.push_back(ProductionCDCL::L1Core::FlatL1Engine::mkLit(v, sign));

            edge_row.push_back(v);
            edge_col.push_back(c_node);
            edge_pos.push_back(lit > 0 ? 1 : 0);
        }
        engine.addClause(cl);
    }

    ProductionCDCL::RefreshPolicy pol = static_cast<ProductionCDCL::RefreshPolicy>(controller_mode);
    engine.setRefreshPolicy(pol, k_param, turb_param, conf_margin);
    engine.setMLPEngine(&g_smlp_engine);

    std::vector<double> weights(act_vars, 1.0);
    std::vector<int> polarities(act_vars, 0);

    auto tg0 = std::chrono::high_resolution_clock::now();
    bool gnn_ok = g_smlp_engine.is_loaded() && 
                  g_smlp_engine.forward(act_vars, num_clauses, (int)edge_row.size(),
                                        edge_row.data(), edge_col.data(), edge_pos.data(),
                                        weights.data(), polarities.data());
    auto tg1 = std::chrono::high_resolution_clock::now();
    if (out_gnn_time_ms) {
        *out_gnn_time_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    }

    if (gnn_ok) {
        engine.setGuidance(weights.data(), polarities.data(), act_vars);
    }

    auto ts0 = std::chrono::high_resolution_clock::now();
    int res = engine.solve(timeout_sec);
    auto ts1 = std::chrono::high_resolution_clock::now();
    if (out_solver_time_ms) {
        *out_solver_time_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    }

    if (out_refreshes) {
        *out_refreshes = engine.getRefreshes();
    }

    s->decisions = engine.getDecisions();
    s->conflicts = engine.getConflicts();
    s->propagations = engine.getPropagations();
    s->status_val = (res == 10) ? STATUS_SAT : (res == 20 ? STATUS_UNSAT : STATUS_CAP_EXCEEDED);
    s->is_done_flag = true;

    if (res == 10) {
        s->model.growTo(act_vars);
        for (int i = 0; i < act_vars; ++i) {
            uint8_t v = engine.getModelLitVal(ProductionCDCL::L1Core::FlatL1Engine::mkLit(i, false));
            s->model[i] = (v == ProductionCDCL::VAL_TRUE) ? l_True : (v == ProductionCDCL::VAL_FALSE ? l_False : l_Undef);
        }
    }

    return s->status_val;
}

}



