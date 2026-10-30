#ifndef STRUCTURAL_MLP_ENGINE_H
#define STRUCTURAL_MLP_ENGINE_H

#include <vector>
#include <string>
#include <cstdint>

class StructuralMLPEngine {
public:
    StructuralMLPEngine();
    ~StructuralMLPEngine();

    bool load_weights(const std::string& path);
    bool is_loaded() const { return weights_loaded; }

    // Inference: computes features and predicts variable weights in microsecond latency
    bool forward(
        int num_vars,
        int num_clauses,
        int num_edges,
        const int* edge_row,
        const int* edge_col,
        const uint8_t* edge_is_pos,
        double* out_weights,
        int* out_polarities
    );

private:
    bool weights_loaded;
    std::vector<float> w1;   // 16 x 6
    std::vector<float> b1;   // 16
    std::vector<float> ln_w; // 16
    std::vector<float> ln_b; // 16
    std::vector<float> w2;   // 16
    float b2;

    // Persistent scratch buffers (zero-allocation)
    std::vector<float> scratch_pos_occ;
    std::vector<float> scratch_neg_occ;
    std::vector<float> scratch_clause_len;
    std::vector<float> scratch_weighted_len;
    std::vector<float> scratch_short_count;
};

#endif // STRUCTURAL_MLP_ENGINE_H
