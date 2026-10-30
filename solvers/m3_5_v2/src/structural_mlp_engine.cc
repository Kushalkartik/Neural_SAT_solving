#include "structural_mlp_engine.h"
#include <fstream>
#include <cmath>
#include <cstring>
#include <immintrin.h>

StructuralMLPEngine::StructuralMLPEngine() : weights_loaded(false), b2(0.0f) {
    w1.resize(16 * 6);
    b1.resize(16);
    ln_w.resize(16);
    ln_b.resize(16);
    w2.resize(16);
}

StructuralMLPEngine::~StructuralMLPEngine() {}

bool StructuralMLPEngine::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "SMLP", 4) != 0) return false;

    f.read(reinterpret_cast<char*>(w1.data()), 16 * 6 * sizeof(float));
    f.read(reinterpret_cast<char*>(b1.data()), 16 * sizeof(float));
    f.read(reinterpret_cast<char*>(ln_w.data()), 16 * sizeof(float));
    f.read(reinterpret_cast<char*>(ln_b.data()), 16 * sizeof(float));
    f.read(reinterpret_cast<char*>(w2.data()), 16 * sizeof(float));
    f.read(reinterpret_cast<char*>(&b2), sizeof(float));

    weights_loaded = true;
    return true;
}

bool StructuralMLPEngine::forward(
    int num_vars,
    int num_clauses,
    int num_edges,
    const int* edge_row,
    const int* edge_col,
    const uint8_t* edge_is_pos,
    double* out_weights,
    int* out_polarities
) {
    if (!weights_loaded || num_vars <= 0) return false;

    // Sizing scratch buffers (only grows, zero-allocation steady state)
    if ((int)scratch_pos_occ.size() < num_vars) {
        scratch_pos_occ.resize(num_vars);
        scratch_neg_occ.resize(num_vars);
        scratch_weighted_len.resize(num_vars);
        scratch_short_count.resize(num_vars);
    }
    if ((int)scratch_clause_len.size() < num_clauses) {
        scratch_clause_len.resize(num_clauses);
    }

    std::memset(scratch_pos_occ.data(), 0, num_vars * sizeof(float));
    std::memset(scratch_neg_occ.data(), 0, num_vars * sizeof(float));
    std::memset(scratch_weighted_len.data(), 0, num_vars * sizeof(float));
    std::memset(scratch_short_count.data(), 0, num_vars * sizeof(float));
    std::memset(scratch_clause_len.data(), 0, num_clauses * sizeof(float));

    // Pass 1: Compute clause lengths
    for (int e = 0; e < num_edges; ++e) {
        int u = edge_row[e];
        int v = edge_col[e];
        if (u < num_vars && v >= num_vars) {
            int c = v - num_vars;
            if (c < num_clauses) {
                scratch_clause_len[c] += 1.0f;
            }
        }
    }

    // Pass 2: Accumulate structural features
    for (int e = 0; e < num_edges; ++e) {
        int u = edge_row[e];
        int v = edge_col[e];
        if (u < num_vars && v >= num_vars) {
            int c = v - num_vars;
            if (edge_is_pos[e]) {
                scratch_pos_occ[u] += 1.0f;
            } else {
                scratch_neg_occ[u] += 1.0f;
            }
            float clen = scratch_clause_len[c];
            if (clen < 1.0f) clen = 1.0f;
            scratch_weighted_len[u] += 1.0f / clen;
            if (clen <= 3.0f) {
                scratch_short_count[u] += 1.0f;
            }
        }
    }

    // Pass 3: In-Register AVX2 MLP Forward Pass
    __m256 zero_vec = _mm256_setzero_ps();
    __m256 eps_vec = _mm256_set1_ps(1e-5f);
    __m256 inv_16 = _mm256_set1_ps(1.0f / 16.0f);

    for (int v = 0; v < num_vars; ++v) {
        float pos = scratch_pos_occ[v];
        float neg = scratch_neg_occ[v];
        float tot = pos + neg;
        float norm_deg = tot * 0.05f;
        float bal = (pos - neg) / (tot + 1.0f);
        float wlen = scratch_weighted_len[v];
        float scnt = scratch_short_count[v];

        float x[6] = {pos, neg, norm_deg, bal, wlen, scnt};

        // Linear 1: 6 -> 16
        float h[16];
        for (int i = 0; i < 16; ++i) {
            float sum = b1[i];
            const float* w_row = w1.data() + i * 6;
            for (int j = 0; j < 6; ++j) {
                sum += w_row[j] * x[j];
            }
            h[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU
        }

        // LayerNorm: 16 elements
        __m256 h0 = _mm256_loadu_ps(h);
        __m256 h1 = _mm256_loadu_ps(h + 8);
        __m256 sum_h = _mm256_add_ps(h0, h1);

        // Horizontal sum of sum_h
        __m128 vlow = _mm256_castps256_ps128(sum_h);
        __m128 vhigh = _mm256_extractf128_ps(sum_h, 1);
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        float mean = _mm_cvtss_f32(vsum) * (1.0f / 16.0f);

        __m256 mean_vec = _mm256_set1_ps(mean);
        __m256 d0 = _mm256_sub_ps(h0, mean_vec);
        __m256 d1 = _mm256_sub_ps(h1, mean_vec);

        __m256 sq0 = _mm256_mul_ps(d0, d0);
        __m256 sq1 = _mm256_mul_ps(d1, d1);
        __m256 sum_sq = _mm256_add_ps(sq0, sq1);

        vlow = _mm256_castps256_ps128(sum_sq);
        vhigh = _mm256_extractf128_ps(sum_sq, 1);
        vsum = _mm_add_ps(vlow, vhigh);
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        float var = _mm_cvtss_f32(vsum) * (1.0f / 16.0f);
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);

        // Normalized * ln_w + ln_b
        __m256 norm0 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(d0, inv_std_vec), _mm256_loadu_ps(ln_w.data())), _mm256_loadu_ps(ln_b.data()));
        __m256 norm1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(d1, inv_std_vec), _mm256_loadu_ps(ln_w.data() + 8)), _mm256_loadu_ps(ln_b.data() + 8));

        // Linear 2: 16 -> 1
        __m256 prod0 = _mm256_mul_ps(norm0, _mm256_loadu_ps(w2.data()));
        __m256 prod1 = _mm256_mul_ps(norm1, _mm256_loadu_ps(w2.data() + 8));
        __m256 sum_prod = _mm256_add_ps(prod0, prod1);

        vlow = _mm256_castps256_ps128(sum_prod);
        vhigh = _mm256_extractf128_ps(sum_prod, 1);
        vsum = _mm_add_ps(vlow, vhigh);
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        float mu = _mm_cvtss_f32(vsum) + b2;

        out_weights[v] = (double)std::exp(mu);
        out_polarities[v] = (bal >= 0.0f) ? 1 : 0; // Native phase saving / balance prior
    }

    return true;
}
