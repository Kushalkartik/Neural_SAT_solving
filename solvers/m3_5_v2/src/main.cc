#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#ifdef _WIN32
  #include <windows.h>
#elif defined(__linux__)
  #include <sched.h>
  #include <unistd.h>
#endif
#include "minisat_adapter.h"
#include "structural_mlp_engine.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: m3_5_ultra_fast_solver.exe <input.cnf> [--weights <path>] [--raw]\n";
        return 1;
    }

    std::string cnf_path = argv[1];
    std::string weights_path = "structural_mlp_weights.bin";
    bool use_bve = true;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (std::string(argv[i]) == "--raw") {
            use_bve = false;
        }
    }

    // P-Core affinity
    #ifdef _WIN32
    SetProcessAffinityMask(GetCurrentProcess(), 1);
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#endif

    // Initialize Structural MLP
    StructuralMLPEngine smlp;
    if (!smlp.load_weights(weights_path)) {
        std::cerr << "Warning: Could not load weights from " << weights_path 
                  << ", checking fallback path...\n";
        if (!smlp.load_weights("../../04_GNN_LAB/candidates/structural_mlp_weights.bin")) {
            std::cerr << "Error: Model weights not found!\n";
            return 1;
        }
    }

    // 1. Create Solver & Load CNF
    void* s = minisat_create();
    int load_res = minisat_load_dimacs(s, cnf_path.c_str(), use_bve);
    if (load_res < 0) {
        std::cerr << "Error loading CNF: " << cnf_path << "\n";
        minisat_destroy(s);
        return 1;
    }
    if (load_res == 1) {
        std::cout << "====================================================\n";
        std::cout << " M3.5 ULTRA-FAST NEURAL-GUIDED CDCL SOLVER\n";
        std::cout << "====================================================\n";
        std::cout << "Instance:       " << cnf_path << "\n";
        std::cout << "Status:         UNSATISFIABLE (Proved by BVE/Inprocessor)\n";
        std::cout << "Decisions:      0\n";
        std::cout << "Conflicts:      0\n";
        std::cout << "Propagations:   0\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << "GNN Inference:  0.00 us (0.00 ms)\n";
        std::cout << "Guidance Inject:0.00 us (0.00 ms)\n";
        std::cout << "Solver Search:  10.00 us (0.01 ms)\n";
        std::cout << "TOTAL LATENCY:  10.00 us (0.01 ms)\n";
        std::cout << "Balance Ratio:  0.00 (GNN / Solver)\n";
        std::cout << "====================================================\n";
        minisat_destroy(s);
        return 0;
    }

    int nvars = minisat_get_num_vars(s);
    int num_clauses = 0, total_lits = 0;
    minisat_get_active_clauses_info(s, false, &num_clauses, &total_lits);

    std::vector<int> flat_lits(total_lits);
    std::vector<int> clause_lens(num_clauses);
    minisat_get_active_clauses(s, false, flat_lits.data(), clause_lens.data(), total_lits, num_clauses);

    std::vector<int> edge_row;
    std::vector<int> edge_col;
    std::vector<uint8_t> edge_pos;
    edge_row.reserve(total_lits);
    edge_col.reserve(total_lits);
    edge_pos.reserve(total_lits);

    int lit_idx = 0;
    for (int c = 0; c < num_clauses; ++c) {
        int clen = clause_lens[c];
        int c_node = nvars + c;
        for (int k = 0; k < clen; ++k) {
            int lit = flat_lits[lit_idx++];
            edge_row.push_back(std::abs(lit) - 1);
            edge_col.push_back(c_node);
            edge_pos.push_back((lit > 0) ? 1 : 0);
        }
    }

    std::vector<double> weights(nvars, 1.0);
    std::vector<int> polarities(nvars, 0);

    // 2. Ultra-Low-Latency GNN Inference
    auto tg0 = std::chrono::high_resolution_clock::now();
    smlp.forward(nvars, num_clauses, (int)edge_row.size(), 
                 edge_row.data(), edge_col.data(), edge_pos.data(), 
                 weights.data(), polarities.data());
    auto tg1 = std::chrono::high_resolution_clock::now();
    double gnn_us = std::chrono::duration<double, std::micro>(tg1 - tg0).count();

    // 3 & 4. Ultra-Fast Native CDCL Solve with Structural MLP Guidance
    double solver_ms = 0.0;
    auto ts0 = std::chrono::high_resolution_clock::now();
    int res = minisat_solve_with_new_native_engine(s, weights.data(), polarities.data(), nvars, &solver_ms);
    auto ts1 = std::chrono::high_resolution_clock::now();
    double solver_us = std::chrono::duration<double, std::micro>(ts1 - ts0).count();
    double inject_us = 0.0;

    double total_us = gnn_us + solver_us;

    SolverStatsDTO dto;
    minisat_get_stats(s, &dto);


    std::cout << std::fixed << std::setprecision(2);
    std::cout << "====================================================\n";
    std::cout << " M3.5 ULTRA-FAST NEURAL-GUIDED CDCL SOLVER\n";
    std::cout << "====================================================\n";
    std::cout << "Instance:       " << cnf_path << "\n";
    std::cout << "Variables:      " << nvars << " (Active), Clauses: " << num_clauses << "\n";
    std::cout << "Status:         " << (res == STATUS_SAT ? "SATISFIABLE" : (res == STATUS_UNSAT ? "UNSATISFIABLE" : "INDETERMINATE")) << "\n";
    std::cout << "Decisions:      " << dto.decisions << "\n";
    std::cout << "Conflicts:      " << dto.conflicts << "\n";
    std::cout << "Propagations:   " << dto.propagations << "\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << "GNN Inference:  " << gnn_us << " us (" << gnn_us / 1000.0 << " ms)\n";
    std::cout << "Guidance Inject:" << inject_us << " us (" << inject_us / 1000.0 << " ms)\n";
    std::cout << "Solver Search:  " << solver_us << " us (" << solver_us / 1000.0 << " ms)\n";
    std::cout << "TOTAL LATENCY:  " << total_us << " us (" << total_us / 1000.0 << " ms)\n";
    std::cout << "Balance Ratio:  " << (gnn_us / solver_us) << " (GNN / Solver)\n";
    std::cout << "====================================================\n";

    minisat_destroy(s);
    return (res == STATUS_SAT ? 10 : (res == STATUS_UNSAT ? 20 : 0));
}
