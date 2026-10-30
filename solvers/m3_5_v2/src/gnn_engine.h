#ifndef GNN_ENGINE_H
#define GNN_ENGINE_H

#include <vector>
#include <string>
#include <cstdint>

namespace Minisat {

struct GNNTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;

    const float* ptr() const { return data.data(); }
    int size() const { return (int)data.size(); }
};

class GNNInferenceEngine {
public:
    GNNInferenceEngine();
    ~GNNInferenceEngine();

    // Load serialized weights from binary file (Version 1 or Version 2)
    bool load_weights(const std::string& weights_path);
    bool is_loaded() const { return weights_loaded; }
    int get_d_model() const { return d_model; }
    int get_core_steps() const { return core_steps; }

    // Run forward inference on an active bipartite variable-clause graph
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
    uint32_t file_version;
    int d_model;
    int core_steps;
    int enc_d;

    std::vector<GNNTensor> tensors;

    // Direct pointers to key weight buffers
    // Encoder
    const float* enc_edge_0_w; const float* enc_edge_0_b;
    const float* enc_edge_ln_w; const float* enc_edge_ln_b;
    const float* enc_node_0_w; const float* enc_node_0_b;
    const float* enc_node_ln_w; const float* enc_node_ln_b;
    const float* enc_glob_0_w; const float* enc_glob_0_b;
    const float* enc_glob_ln_w; const float* enc_glob_ln_b;

    // Core
    const float* core_edge_0_w; const float* core_edge_0_b;
    const float* core_edge_2_w; const float* core_edge_2_b;
    const float* core_edge_ln_w; const float* core_edge_ln_b;

    const float* core_node_0_w; const float* core_node_0_b;
    const float* core_node_2_w; const float* core_node_2_b;
    const float* core_node_ln_w; const float* core_node_ln_b;

    const float* core_glob_0_w; const float* core_glob_0_b;
    const float* core_glob_2_w; const float* core_glob_2_b;
    const float* core_glob_ln_w; const float* core_glob_ln_b;

    // Decoder
    const float* dec_node_0_w; const float* dec_node_0_b;
    const float* dec_node_ln_w; const float* dec_node_ln_b;

    // Weight Head
    const float* wh_0_w; const float* wh_0_b;
    const float* wh_ln_w; const float* wh_ln_b;
    const float* wh_3_w; const float* wh_3_b;

    // Polarity Head
    const float* ph_0_w; const float* ph_0_b;
    const float* ph_ln_w; const float* ph_ln_b;
    const float* ph_3_w; const float* ph_3_b;

    // Distributive sliced weights for L1-cache GEMV
    std::vector<float> w_core_edge_src_tpl;
    std::vector<float> w_core_edge_src_lat;
    std::vector<float> w_core_edge_dest_tpl;
    std::vector<float> w_core_edge_dest_lat;
    std::vector<float> w_core_edge_e_tpl;
    std::vector<float> w_core_edge_e_lat;
    std::vector<float> w_core_edge_glob_tpl;
    std::vector<float> w_core_edge_glob_lat;

    std::vector<float> w_core_node_tpl;
    std::vector<float> w_core_node_lat;
    std::vector<float> w_core_node_edgesum;
    std::vector<float> w_core_node_glob_tpl;
    std::vector<float> w_core_node_glob_lat;

    // Template vectors
    std::vector<float> enc_var_tpl, enc_clause_tpl;
    std::vector<float> enc_pos_edge_tpl, enc_neg_edge_tpl;
    std::vector<float> enc_glob_tpl;

    // Precomputed constant projections
    std::vector<float> proj_src_tpl_var, proj_src_tpl_clause;
    std::vector<float> proj_dest_tpl_var, proj_dest_tpl_clause;
    std::vector<float> proj_edge_tpl_pos, proj_edge_tpl_neg;
    std::vector<float> proj_glob_tpl;

    std::vector<float> proj_node_tpl_var, proj_node_tpl_clause;
    std::vector<float> proj_node_glob_tpl;

    // Persistent scratch buffers (zero allocation in hot path)
    std::vector<float> scratch_latent_x;
    std::vector<float> scratch_next_latent_x;
    std::vector<float> scratch_latent_e;
    std::vector<float> scratch_next_latent_e;
    std::vector<float> scratch_edge_sum;
    std::vector<float> scratch_proj_node_src;
    std::vector<float> scratch_proj_node_dest;

    const GNNTensor* find_tensor(const std::string& name) const;
    void bind_pointers();

    // Fast-path forward engines
    bool forward_16(int num_vars, int num_clauses, int num_edges, const int* edge_row, const int* edge_col, const uint8_t* edge_is_pos, double* out_weights, int* out_polarities);
    bool forward_32(int num_vars, int num_clauses, int num_edges, const int* edge_row, const int* edge_col, const uint8_t* edge_is_pos, double* out_weights, int* out_polarities);
    bool forward_64(int num_vars, int num_clauses, int num_edges, const int* edge_row, const int* edge_col, const uint8_t* edge_is_pos, double* out_weights, int* out_polarities);

    // Neural layer primitives (AVX2-accelerated)
    static void gemv_avx2(const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out, int in_dim, int out_dim);
    static void gemv_16x16_avx2(const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out);
    static void gemv_64x64_avx2(const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out);
    static void relu_avx2(float* data, int dim);
    static void relu_avx2_16(float* data);
    static void relu_avx2_64(float* data);
    static void layernorm_avx2(const float* __restrict__ in, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, int dim, float eps = 1e-5f);
    static void layernorm_avx2_16(const float* __restrict__ in, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, float eps = 1e-5f);
    static void layernorm_avx2_64(const float* __restrict__ in, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, float eps = 1e-5f);
};

// Global singleton instance for in-solver evaluation
GNNInferenceEngine& get_global_gnn_engine();

} // namespace Minisat

#endif // GNN_ENGINE_H
