#ifndef MINISAT_ADAPTER_H
#define MINISAT_ADAPTER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
  #define ADAPTER_API __declspec(dllexport)
#else
  #define ADAPTER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Solver Status Codes
#define STATUS_UNDEF        0
#define STATUS_SAT          1
#define STATUS_UNSAT        2
#define STATUS_CAP_EXCEEDED 3
#define STATUS_ERROR        4

// Telemetry structure
typedef struct {
    uint64_t decisions;
    uint64_t conflicts;
    uint64_t propagations;
    uint64_t restarts;
    uint64_t num_clauses;
    uint64_t num_learnts;
    double   cpu_time;
} SolverStatsDTO;

// Solver instance lifecycle
ADAPTER_API void* minisat_create();
ADAPTER_API void  minisat_destroy(void* solver_ptr);

// Configuration
ADAPTER_API void  minisat_set_verbosity(void* solver_ptr, int verbosity);
ADAPTER_API void  minisat_set_random_seed(void* solver_ptr, double seed);
ADAPTER_API void  minisat_set_restart_policy(void* solver_ptr, bool with_restarts);

// Problem loading
ADAPTER_API int   minisat_load_dimacs(void* solver_ptr, const char* filepath, bool simplify);

// Pure classical solve (continuous VSIDS search)
ADAPTER_API int   minisat_solve(void* solver_ptr);

// Step-driven CDCL search (for branching policy interception)
ADAPTER_API int   minisat_init_stepping(void* solver_ptr, bool with_restarts, int max_decision_cap);
ADAPTER_API int   minisat_step(void* solver_ptr, int decision_lit);
ADAPTER_API int   minisat_pick_vsids_literal(void* solver_ptr); // returns signed 1-based literal, or 0 if none
ADAPTER_API int   minisat_get_last_branch_lit(void* solver_ptr); // signed 1-based literal actually branched in last step, 0 if none

// Solver state and inspection
ADAPTER_API int   minisat_get_num_vars(void* solver_ptr);
ADAPTER_API int   minisat_get_num_clauses(void* solver_ptr);
ADAPTER_API int   minisat_get_num_learnts(void* solver_ptr);
ADAPTER_API int   minisat_get_decision_level(void* solver_ptr);
ADAPTER_API int   minisat_get_status(void* solver_ptr);
ADAPTER_API bool  minisat_is_done(void* solver_ptr);

// Telemetry extraction
ADAPTER_API void  minisat_get_stats(void* solver_ptr, SolverStatsDTO* out_stats);
ADAPTER_API int   minisat_get_assignment(void* solver_ptr, int var_id); // 1=True, 0=False, -1=Unassigned
ADAPTER_API int   minisat_get_assignments(void* solver_ptr, int* out_arr, int max_n);
ADAPTER_API int   minisat_get_activities(void* solver_ptr, double* out_arr, int max_n);
ADAPTER_API int   minisat_get_model(void* solver_ptr, int* out_arr, int max_n); // signed 1-based model

// Graph state extraction for neural branching (M1+)
ADAPTER_API int   minisat_get_active_clauses_info(void* solver_ptr, bool include_learnts, int* out_num_clauses, int* out_total_literals);
ADAPTER_API int   minisat_get_active_clauses(void* solver_ptr, bool include_learnts, int* out_flat_literals, int* out_clause_lengths, int max_literals, int max_clauses);

// Milestone 2B Neural Prior Injection (NeuroComb Initial Heap & Polarity Injection)
ADAPTER_API int   minisat_set_initial_activities(void* solver_ptr, const double* activities, int n_vars);
ADAPTER_API int   minisat_set_initial_polarities(void* solver_ptr, const int* polarities, int n_vars);
ADAPTER_API int   minisat_set_decision_vars(void* solver_ptr, const int* decision_flags, int n_vars);

// Milestone 2C RLAF Dynamic Guidance (Tönshoff & Grohe ICLR 2026 Eq 20)
ADAPTER_API int   minisat_set_rlaf_guidance(void* solver_ptr, const double* weights, const int* polarities, int n_vars);
ADAPTER_API int   minisat_clear_rlaf_guidance(void* solver_ptr);

// Native C++ In-Solver GNN Engine (Zero-Overhead Inference)
ADAPTER_API int   minisat_init_native_gnn(const char* weights_path);
ADAPTER_API int   minisat_is_native_gnn_ready();
ADAPTER_API int   minisat_get_native_gnn_guidance(void* solver_ptr, double* out_weights, int* out_polarities, int n_vars, double* out_gnn_time_ms);
ADAPTER_API int   minisat_solve_with_native_gnn(void* solver_ptr, double* out_gnn_time_ms);

// Milestone 3 / 3.5: Ultra-Fast Structural Distillation MLP (4.4 µs)
ADAPTER_API int   minisat_init_structural_mlp(const char* weights_path);
ADAPTER_API int   minisat_is_structural_mlp_ready();
ADAPTER_API int   minisat_get_structural_mlp_guidance(void* solver_ptr, double* out_weights, int* out_polarities, int n_vars, double* out_gnn_time_ms);
ADAPTER_API int   minisat_solve_with_structural_mlp(void* solver_ptr, double* out_gnn_time_ms);

// Milestone 3.5: Unified Native SOTA Dual-Engine (FlatCDCL + Structural MLP)
ADAPTER_API int   minisat_solve_m3_5_unified(void* solver_ptr, double* out_gnn_time_ms, double* out_solver_time_ms);
ADAPTER_API int   minisat_solve_with_new_native_engine(void* solver_ptr, const double* weights, const int* polarities, int n_vars, double* out_solver_time_ms);

// Competition Timeout Solvers
ADAPTER_API int   minisat_solve_with_timeout(void* solver_ptr, double timeout_sec);
ADAPTER_API int   minisat_solve_with_structural_mlp_timeout(void* solver_ptr, double timeout_sec, double* out_gnn_time_ms);
ADAPTER_API int   minisat_solve_m3_5_unified_timeout(void* solver_ptr, double timeout_sec, double* out_gnn_time_ms, double* out_solver_time_ms);

// Milestone 4: Adaptive Cost-Aware Refresh Controller (M4A - M4E)
ADAPTER_API int   minisat_solve_m4_controller(
    void* solver_ptr,
    int controller_mode,
    int k_param,
    double turb_param,
    double conf_margin,
    double timeout_sec,
    double* out_gnn_time_ms,
    double* out_solver_time_ms,
    int64_t* out_refreshes
);

#ifdef __cplusplus
}
#endif

#endif // MINISAT_ADAPTER_H

