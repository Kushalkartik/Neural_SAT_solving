#include "gnn_engine.h"
#include <immintrin.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace Minisat {

static const uint32_t EXPECTED_MAGIC = 0x53415447; // "SATG"

GNNInferenceEngine::GNNInferenceEngine() 
    : weights_loaded(false), file_version(0), d_model(0), core_steps(0), enc_d(0) {
    enc_edge_0_w = enc_edge_0_b = enc_edge_ln_w = enc_edge_ln_b = nullptr;
    enc_node_0_w = enc_node_0_b = enc_node_ln_w = enc_node_ln_b = nullptr;
    enc_glob_0_w = enc_glob_0_b = enc_glob_ln_w = enc_glob_ln_b = nullptr;

    core_edge_0_w = core_edge_0_b = core_edge_2_w = core_edge_2_b = core_edge_ln_w = core_edge_ln_b = nullptr;
    core_node_0_w = core_node_0_b = core_node_2_w = core_node_2_b = core_node_ln_w = core_node_ln_b = nullptr;
    core_glob_0_w = core_glob_0_b = core_glob_2_w = core_glob_2_b = core_glob_ln_w = core_glob_ln_b = nullptr;

    dec_node_0_w = dec_node_0_b = dec_node_ln_w = dec_node_ln_b = nullptr;

    wh_0_w = wh_0_b = wh_ln_w = wh_ln_b = wh_3_w = wh_3_b = nullptr;
    ph_0_w = ph_0_b = ph_ln_w = ph_ln_b = ph_3_w = ph_3_b = nullptr;
}

GNNInferenceEngine::~GNNInferenceEngine() {}

const GNNTensor* GNNInferenceEngine::find_tensor(const std::string& name) const {
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].name == name) return &tensors[i];
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Generic AVX2 Primitives
// ---------------------------------------------------------------------------
void GNNInferenceEngine::gemv_avx2(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int in_dim,
    int out_dim
) {
    for (int j = 0; j < out_dim; ++j) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        const float* w_row = w + j * in_dim;
        int k = 0;
        for (; k + 15 < in_dim; k += 16) {
            __m256 v0 = _mm256_loadu_ps(in + k);
            __m256 w0 = _mm256_loadu_ps(w_row + k);
            acc0 = _mm256_fmadd_ps(v0, w0, acc0);

            __m256 v1 = _mm256_loadu_ps(in + k + 8);
            __m256 w1 = _mm256_loadu_ps(w_row + k + 8);
            acc1 = _mm256_fmadd_ps(v1, w1, acc1);
        }
        for (; k + 7 < in_dim; k += 8) {
            __m256 v0 = _mm256_loadu_ps(in + k);
            __m256 w0 = _mm256_loadu_ps(w_row + k);
            acc0 = _mm256_fmadd_ps(v0, w0, acc0);
        }
        __m256 sum256 = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(sum256);
        __m128 hi = _mm256_extractf128_ps(sum256, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        s128 = _mm_hadd_ps(s128, s128);
        s128 = _mm_hadd_ps(s128, s128);
        float s = _mm_cvtss_f32(s128);
        for (; k < in_dim; ++k) {
            s += in[k] * w_row[k];
        }
        if (bias) s += bias[j];
        out[j] = s;
    }
}

void GNNInferenceEngine::relu_avx2(float* data, int dim) {
    __m256 zero = _mm256_setzero_ps();
    int k = 0;
    for (; k + 7 < dim; k += 8) {
        __m256 v = _mm256_loadu_ps(data + k);
        _mm256_storeu_ps(data + k, _mm256_max_ps(v, zero));
    }
    for (; k < dim; ++k) {
        if (data[k] < 0.0f) data[k] = 0.0f;
    }
}

void GNNInferenceEngine::layernorm_avx2(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int dim,
    float eps
) {
    float sum = 0.0f;
    __m256 sum_vec = _mm256_setzero_ps();
    int k = 0;
    for (; k + 7 < dim; k += 8) {
        sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(in + k));
    }
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    sum = _mm_cvtss_f32(s128);
    for (; k < dim; ++k) sum += in[k];
    float mean = sum / (float)dim;

    float sq_sum = 0.0f;
    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 sq_vec = _mm256_setzero_ps();
    k = 0;
    for (; k + 7 < dim; k += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(in + k), mean_vec);
        sq_vec = _mm256_fmadd_ps(diff, diff, sq_vec);
    }
    lo = _mm256_castps256_ps128(sq_vec);
    hi = _mm256_extractf128_ps(sq_vec, 1);
    s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    sq_sum = _mm_cvtss_f32(s128);
    for (; k < dim; ++k) {
        float d = in[k] - mean;
        sq_sum += d * d;
    }
    float var = sq_sum / (float)dim;
    float inv_std = 1.0f / sqrtf(var + eps);

    __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    k = 0;
    for (; k + 7 < dim; k += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(in + k), mean_vec);
        __m256 norm = _mm256_mul_ps(diff, inv_std_vec);
        __m256 g = _mm256_loadu_ps(gamma + k);
        __m256 b = _mm256_loadu_ps(beta + k);
        _mm256_storeu_ps(out + k, _mm256_fmadd_ps(norm, g, b));
    }
    for (; k < dim; ++k) {
        out[k] = ((in[k] - mean) * inv_std) * gamma[k] + beta[k];
    }
}

// ---------------------------------------------------------------------------
// Specialized 16x16 AVX2 Primitives
// ---------------------------------------------------------------------------
void GNNInferenceEngine::gemv_16x16_avx2(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out
) {
    __m256 vx0 = _mm256_loadu_ps(in);
    __m256 vx1 = _mm256_loadu_ps(in + 8);

    for (int j = 0; j < 16; ++j) {
        const float* w_row = w + j * 16;
        __m256 w0 = _mm256_loadu_ps(w_row);
        __m256 w1 = _mm256_loadu_ps(w_row + 8);

        __m256 prod = _mm256_mul_ps(vx0, w0);
        prod = _mm256_fmadd_ps(vx1, w1, prod);

        __m128 lo = _mm256_castps256_ps128(prod);
        __m128 hi = _mm256_extractf128_ps(prod, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float s = _mm_cvtss_f32(sum128);
        if (bias) s += bias[j];
        out[j] = s;
    }
}

void GNNInferenceEngine::relu_avx2_16(float* data) {
    __m256 zero = _mm256_setzero_ps();
    __m256 v0 = _mm256_loadu_ps(data);
    __m256 v1 = _mm256_loadu_ps(data + 8);
    _mm256_storeu_ps(data, _mm256_max_ps(v0, zero));
    _mm256_storeu_ps(data + 8, _mm256_max_ps(v1, zero));
}

void GNNInferenceEngine::layernorm_avx2_16(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float eps
) {
    __m256 v0 = _mm256_loadu_ps(in);
    __m256 v1 = _mm256_loadu_ps(in + 8);
    __m256 sum_vec = _mm256_add_ps(v0, v1);

    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float mean = _mm_cvtss_f32(s128) * (1.0f / 16.0f);

    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 diff0 = _mm256_sub_ps(v0, mean_vec);
    __m256 diff1 = _mm256_sub_ps(v1, mean_vec);

    __m256 sq_vec = _mm256_mul_ps(diff0, diff0);
    sq_vec = _mm256_fmadd_ps(diff1, diff1, sq_vec);

    lo = _mm256_castps256_ps128(sq_vec);
    hi = _mm256_extractf128_ps(sq_vec, 1);
    s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float var = _mm_cvtss_f32(s128) * (1.0f / 16.0f);
    float inv_std = 1.0f / sqrtf(var + eps);

    __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    __m256 g0 = _mm256_loadu_ps(gamma);
    __m256 b0 = _mm256_loadu_ps(beta);
    __m256 g1 = _mm256_loadu_ps(gamma + 8);
    __m256 b1 = _mm256_loadu_ps(beta + 8);

    _mm256_storeu_ps(out, _mm256_fmadd_ps(_mm256_mul_ps(diff0, inv_std_vec), g0, b0));
    _mm256_storeu_ps(out + 8, _mm256_fmadd_ps(_mm256_mul_ps(diff1, inv_std_vec), g1, b1));
}

// ---------------------------------------------------------------------------
// Specialized 64x64 AVX2 Primitives
// ---------------------------------------------------------------------------
void GNNInferenceEngine::gemv_64x64_avx2(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out
) {
    for (int j = 0; j < 64; ++j) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        const float* w_row = w + j * 64;
        for (int k = 0; k < 64; k += 16) {
            __m256 v0 = _mm256_loadu_ps(in + k);
            __m256 w0 = _mm256_loadu_ps(w_row + k);
            acc0 = _mm256_fmadd_ps(v0, w0, acc0);

            __m256 v1 = _mm256_loadu_ps(in + k + 8);
            __m256 w1 = _mm256_loadu_ps(w_row + k + 8);
            acc1 = _mm256_fmadd_ps(v1, w1, acc1);
        }
        __m256 sum256 = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(sum256);
        __m128 hi = _mm256_extractf128_ps(sum256, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        s128 = _mm_hadd_ps(s128, s128);
        s128 = _mm_hadd_ps(s128, s128);
        float s = _mm_cvtss_f32(s128);
        if (bias) s += bias[j];
        out[j] = s;
    }
}

void GNNInferenceEngine::relu_avx2_64(float* data) {
    __m256 zero = _mm256_setzero_ps();
    for (int k = 0; k < 64; k += 8) {
        __m256 v = _mm256_loadu_ps(data + k);
        _mm256_storeu_ps(data + k, _mm256_max_ps(v, zero));
    }
}

void GNNInferenceEngine::layernorm_avx2_64(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float eps
) {
    __m256 sum_vec = _mm256_setzero_ps();
    for (int k = 0; k < 64; k += 8) {
        sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(in + k));
    }
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float mean = _mm_cvtss_f32(s128) * (1.0f / 64.0f);

    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 sq_vec = _mm256_setzero_ps();
    for (int k = 0; k < 64; k += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(in + k), mean_vec);
        sq_vec = _mm256_fmadd_ps(diff, diff, sq_vec);
    }
    lo = _mm256_castps256_ps128(sq_vec);
    hi = _mm256_extractf128_ps(sq_vec, 1);
    s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float var = _mm_cvtss_f32(s128) * (1.0f / 64.0f);
    float inv_std = 1.0f / sqrtf(var + eps);

    __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    for (int k = 0; k < 64; k += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(in + k), mean_vec);
        __m256 norm = _mm256_mul_ps(diff, inv_std_vec);
        __m256 g = _mm256_loadu_ps(gamma + k);
        __m256 b = _mm256_loadu_ps(beta + k);
        _mm256_storeu_ps(out + k, _mm256_fmadd_ps(norm, g, b));
    }
}

// ---------------------------------------------------------------------------
// Weight Binding & Precomputation
// ---------------------------------------------------------------------------
void GNNInferenceEngine::bind_pointers() {
    auto get_p = [this](const std::string& n) -> const float* {
        const GNNTensor* t = find_tensor(n);
        return t ? t->ptr() : nullptr;
    };

    // Encoder
    enc_edge_0_w  = get_p("encoder.edge_mlp.0.weight");
    enc_edge_0_b  = get_p("encoder.edge_mlp.0.bias");
    enc_edge_ln_w = get_p("encoder.edge_mlp.2.weight");
    enc_edge_ln_b = get_p("encoder.edge_mlp.2.bias");

    enc_node_0_w  = get_p("encoder.node_mlp.0.weight");
    enc_node_0_b  = get_p("encoder.node_mlp.0.bias");
    enc_node_ln_w = get_p("encoder.node_mlp.2.weight");
    enc_node_ln_b = get_p("encoder.node_mlp.2.bias");

    enc_glob_0_w  = get_p("encoder.global_mlp.0.weight");
    enc_glob_0_b  = get_p("encoder.global_mlp.0.bias");
    enc_glob_ln_w = get_p("encoder.global_mlp.2.weight");
    enc_glob_ln_b = get_p("encoder.global_mlp.2.bias");

    // Core
    core_edge_0_w  = get_p("core.edge_mlp.0.weight");
    core_edge_0_b  = get_p("core.edge_mlp.0.bias");
    core_edge_2_w  = get_p("core.edge_mlp.2.weight");
    core_edge_2_b  = get_p("core.edge_mlp.2.bias");
    core_edge_ln_w = get_p("core.edge_mlp.4.weight");
    core_edge_ln_b = get_p("core.edge_mlp.4.bias");

    core_node_0_w  = get_p("core.node_mlp.0.weight");
    core_node_0_b  = get_p("core.node_mlp.0.bias");
    core_node_2_w  = get_p("core.node_mlp.2.weight");
    core_node_2_b  = get_p("core.node_mlp.2.bias");
    core_node_ln_w = get_p("core.node_mlp.4.weight");
    core_node_ln_b = get_p("core.node_mlp.4.bias");

    core_glob_0_w  = get_p("core.global_mlp.0.weight");
    core_glob_0_b  = get_p("core.global_mlp.0.bias");
    core_glob_2_w  = get_p("core.global_mlp.2.weight");
    core_glob_2_b  = get_p("core.global_mlp.2.bias");
    core_glob_ln_w = get_p("core.global_mlp.4.weight");
    core_glob_ln_b = get_p("core.global_mlp.4.bias");

    // Decoder
    dec_node_0_w  = get_p("decoder.node_mlp.0.weight");
    dec_node_0_b  = get_p("decoder.node_mlp.0.bias");
    dec_node_ln_w = get_p("decoder.node_mlp.2.weight");
    dec_node_ln_b = get_p("decoder.node_mlp.2.bias");

    // Weight Head
    wh_0_w  = get_p("weight_head.0.weight");
    wh_0_b  = get_p("weight_head.0.bias");
    wh_ln_w = get_p("weight_head.2.weight");
    wh_ln_b = get_p("weight_head.2.bias");
    wh_3_w  = get_p("weight_head.3.weight");
    wh_3_b  = get_p("weight_head.3.bias");

    // Polarity Head
    ph_0_w  = get_p("polarity_head.0.weight");
    ph_0_b  = get_p("polarity_head.0.bias");
    ph_ln_w = get_p("polarity_head.2.weight");
    ph_ln_b = get_p("polarity_head.2.bias");
    ph_3_w  = get_p("polarity_head.3.weight");
    ph_3_b  = get_p("polarity_head.3.bias");

    // Resize vectors according to d_model and enc_d
    enc_var_tpl.resize(enc_d);
    enc_clause_tpl.resize(enc_d);
    enc_pos_edge_tpl.resize(enc_d);
    enc_neg_edge_tpl.resize(enc_d);
    enc_glob_tpl.resize(enc_d);

    proj_src_tpl_var.resize(d_model);
    proj_src_tpl_clause.resize(d_model);
    proj_dest_tpl_var.resize(d_model);
    proj_dest_tpl_clause.resize(d_model);
    proj_edge_tpl_pos.resize(d_model);
    proj_edge_tpl_neg.resize(d_model);
    proj_glob_tpl.resize(d_model);

    proj_node_tpl_var.resize(d_model);
    proj_node_tpl_clause.resize(d_model);
    proj_node_glob_tpl.resize(d_model);
    if (d_model == 16) {
        w_core_edge_src_tpl.resize(16 * 16);
        w_core_edge_src_lat.resize(16 * 16);
        w_core_edge_dest_tpl.resize(16 * 16);
        w_core_edge_dest_lat.resize(16 * 16);
        w_core_edge_e_tpl.resize(16 * 16);
        w_core_edge_e_lat.resize(16 * 16);
        w_core_edge_glob_tpl.resize(16 * 16);
        w_core_edge_glob_lat.resize(16 * 16);

        w_core_node_tpl.resize(16 * 16);
        w_core_node_lat.resize(16 * 16);
        w_core_node_edgesum.resize(16 * 16);
        w_core_node_glob_tpl.resize(16 * 16);
        w_core_node_glob_lat.resize(16 * 16);

        // Slicing core_edge_0_w [16, 128]
        for (int j = 0; j < 16; ++j) {
            const float* row = core_edge_0_w + j * 128;
            memcpy(&w_core_edge_src_tpl[j * 16], row + 0, 16 * sizeof(float));
            memcpy(&w_core_edge_src_lat[j * 16], row + 16, 16 * sizeof(float));
            memcpy(&w_core_edge_dest_tpl[j * 16], row + 32, 16 * sizeof(float));
            memcpy(&w_core_edge_dest_lat[j * 16], row + 48, 16 * sizeof(float));
            memcpy(&w_core_edge_e_tpl[j * 16], row + 64, 16 * sizeof(float));
            memcpy(&w_core_edge_e_lat[j * 16], row + 80, 16 * sizeof(float));
            memcpy(&w_core_edge_glob_tpl[j * 16], row + 96, 16 * sizeof(float));
            memcpy(&w_core_edge_glob_lat[j * 16], row + 112, 16 * sizeof(float));
        }

        // Slicing core_node_0_w [16, 80]
        for (int j = 0; j < 16; ++j) {
            const float* row = core_node_0_w + j * 80;
            memcpy(&w_core_node_tpl[j * 16], row + 0, 16 * sizeof(float));
            memcpy(&w_core_node_lat[j * 16], row + 16, 16 * sizeof(float));
            memcpy(&w_core_node_edgesum[j * 16], row + 32, 16 * sizeof(float));
            memcpy(&w_core_node_glob_tpl[j * 16], row + 48, 16 * sizeof(float));
            memcpy(&w_core_node_glob_lat[j * 16], row + 64, 16 * sizeof(float));
        }

        // Template encodings
        float buf[16];
        // Var: [1, 0]
        {
            float x[2] = {1.0f, 0.0f};
            gemv_avx2(x, enc_node_0_w, enc_node_0_b, buf, 2, 16);
            relu_avx2_16(buf);
            layernorm_avx2_16(buf, enc_node_ln_w, enc_node_ln_b, enc_var_tpl.data());
        }
        // Clause: [0, 1]
        {
            float x[2] = {0.0f, 1.0f};
            gemv_avx2(x, enc_node_0_w, enc_node_0_b, buf, 2, 16);
            relu_avx2_16(buf);
            layernorm_avx2_16(buf, enc_node_ln_w, enc_node_ln_b, enc_clause_tpl.data());
        }
        // Neg edge: [1, 0]
        {
            float e[2] = {1.0f, 0.0f};
            gemv_avx2(e, enc_edge_0_w, enc_edge_0_b, buf, 2, 16);
            relu_avx2_16(buf);
            layernorm_avx2_16(buf, enc_edge_ln_w, enc_edge_ln_b, enc_neg_edge_tpl.data());
        }
        // Pos edge: [0, 1]
        {
            float e[2] = {0.0f, 1.0f};
            gemv_avx2(e, enc_edge_0_w, enc_edge_0_b, buf, 2, 16);
            relu_avx2_16(buf);
            layernorm_avx2_16(buf, enc_edge_ln_w, enc_edge_ln_b, enc_pos_edge_tpl.data());
        }
        // Glob: [0]
        {
            float u[1] = {0.0f};
            gemv_avx2(u, enc_glob_0_w, enc_glob_0_b, buf, 1, 16);
            relu_avx2_16(buf);
            layernorm_avx2_16(buf, enc_glob_ln_w, enc_glob_ln_b, enc_glob_tpl.data());
        }

        // Constant projections
        gemv_16x16_avx2(enc_var_tpl.data(), w_core_edge_src_tpl.data(), nullptr, proj_src_tpl_var.data());
        gemv_16x16_avx2(enc_clause_tpl.data(), w_core_edge_src_tpl.data(), nullptr, proj_src_tpl_clause.data());

        gemv_16x16_avx2(enc_var_tpl.data(), w_core_edge_dest_tpl.data(), nullptr, proj_dest_tpl_var.data());
        gemv_16x16_avx2(enc_clause_tpl.data(), w_core_edge_dest_tpl.data(), nullptr, proj_dest_tpl_clause.data());

        gemv_16x16_avx2(enc_pos_edge_tpl.data(), w_core_edge_e_tpl.data(), nullptr, proj_edge_tpl_pos.data());
        gemv_16x16_avx2(enc_neg_edge_tpl.data(), w_core_edge_e_tpl.data(), nullptr, proj_edge_tpl_neg.data());

        gemv_16x16_avx2(enc_glob_tpl.data(), w_core_edge_glob_tpl.data(), nullptr, proj_glob_tpl.data());

        gemv_16x16_avx2(enc_var_tpl.data(), w_core_node_tpl.data(), nullptr, proj_node_tpl_var.data());
        gemv_16x16_avx2(enc_clause_tpl.data(), w_core_node_tpl.data(), nullptr, proj_node_tpl_clause.data());

        gemv_16x16_avx2(enc_glob_tpl.data(), w_core_node_glob_tpl.data(), nullptr, proj_node_glob_tpl.data());

    } else if (d_model == 64) {
        w_core_edge_src_tpl.resize(64 * 32);
        w_core_edge_src_lat.resize(64 * 64);
        w_core_edge_dest_tpl.resize(64 * 32);
        w_core_edge_dest_lat.resize(64 * 64);
        w_core_edge_e_tpl.resize(64 * 32);
        w_core_edge_e_lat.resize(64 * 64);
        w_core_edge_glob_tpl.resize(64 * 32);
        w_core_edge_glob_lat.resize(64 * 32);

        w_core_node_tpl.resize(64 * 32);
        w_core_node_lat.resize(64 * 64);
        w_core_node_edgesum.resize(64 * 64);
        w_core_node_glob_tpl.resize(64 * 32);
        w_core_node_glob_lat.resize(64 * 32);

        // Slicing core_edge_0_w [64, 352]
        for (int j = 0; j < 64; ++j) {
            const float* row = core_edge_0_w + j * 352;
            memcpy(&w_core_edge_src_tpl[j * 32], row + 0, 32 * sizeof(float));
            memcpy(&w_core_edge_src_lat[j * 64], row + 32, 64 * sizeof(float));
            memcpy(&w_core_edge_dest_tpl[j * 32], row + 96, 32 * sizeof(float));
            memcpy(&w_core_edge_dest_lat[j * 64], row + 128, 64 * sizeof(float));
            memcpy(&w_core_edge_e_tpl[j * 32], row + 192, 32 * sizeof(float));
            memcpy(&w_core_edge_e_lat[j * 64], row + 224, 64 * sizeof(float));
            memcpy(&w_core_edge_glob_tpl[j * 32], row + 288, 32 * sizeof(float));
            memcpy(&w_core_edge_glob_lat[j * 32], row + 320, 32 * sizeof(float));
        }

        // Slicing core_node_0_w [64, 224]
        for (int j = 0; j < 64; ++j) {
            const float* row = core_node_0_w + j * 224;
            memcpy(&w_core_node_tpl[j * 32], row + 0, 32 * sizeof(float));
            memcpy(&w_core_node_lat[j * 64], row + 32, 64 * sizeof(float));
            memcpy(&w_core_node_edgesum[j * 64], row + 96, 64 * sizeof(float));
            memcpy(&w_core_node_glob_tpl[j * 32], row + 160, 32 * sizeof(float));
            memcpy(&w_core_node_glob_lat[j * 32], row + 192, 32 * sizeof(float));
        }

        // Compute template encodings
        float buf[32];
        // Var: [1, 0]
        {
            float x[2] = {1.0f, 0.0f};
            gemv_avx2(x, enc_node_0_w, enc_node_0_b, buf, 2, 32);
            relu_avx2(buf, 32);
            layernorm_avx2(buf, enc_node_ln_w, enc_node_ln_b, enc_var_tpl.data(), 32);
        }
        // Clause: [0, 1]
        {
            float x[2] = {0.0f, 1.0f};
            gemv_avx2(x, enc_node_0_w, enc_node_0_b, buf, 2, 32);
            relu_avx2(buf, 32);
            layernorm_avx2(buf, enc_node_ln_w, enc_node_ln_b, enc_clause_tpl.data(), 32);
        }
        // Neg edge: [1, 0]
        {
            float e[2] = {1.0f, 0.0f};
            gemv_avx2(e, enc_edge_0_w, enc_edge_0_b, buf, 2, 32);
            relu_avx2(buf, 32);
            layernorm_avx2(buf, enc_edge_ln_w, enc_edge_ln_b, enc_neg_edge_tpl.data(), 32);
        }
        // Pos edge: [0, 1]
        {
            float e[2] = {0.0f, 1.0f};
            gemv_avx2(e, enc_edge_0_w, enc_edge_0_b, buf, 2, 32);
            relu_avx2(buf, 32);
            layernorm_avx2(buf, enc_edge_ln_w, enc_edge_ln_b, enc_pos_edge_tpl.data(), 32);
        }
        // Glob: [0]
        {
            float u[1] = {0.0f};
            gemv_avx2(u, enc_glob_0_w, enc_glob_0_b, buf, 1, 32);
            relu_avx2(buf, 32);
            layernorm_avx2(buf, enc_glob_ln_w, enc_glob_ln_b, enc_glob_tpl.data(), 32);
        }

        // Precompute constant projections
        gemv_avx2(enc_var_tpl.data(), w_core_edge_src_tpl.data(), nullptr, proj_src_tpl_var.data(), 32, 64);
        gemv_avx2(enc_clause_tpl.data(), w_core_edge_src_tpl.data(), nullptr, proj_src_tpl_clause.data(), 32, 64);

        gemv_avx2(enc_var_tpl.data(), w_core_edge_dest_tpl.data(), nullptr, proj_dest_tpl_var.data(), 32, 64);
        gemv_avx2(enc_clause_tpl.data(), w_core_edge_dest_tpl.data(), nullptr, proj_dest_tpl_clause.data(), 32, 64);

        gemv_avx2(enc_pos_edge_tpl.data(), w_core_edge_e_tpl.data(), nullptr, proj_edge_tpl_pos.data(), 32, 64);
        gemv_avx2(enc_neg_edge_tpl.data(), w_core_edge_e_tpl.data(), nullptr, proj_edge_tpl_neg.data(), 32, 64);

        gemv_avx2(enc_glob_tpl.data(), w_core_edge_glob_tpl.data(), nullptr, proj_glob_tpl.data(), 32, 64);

        gemv_avx2(enc_var_tpl.data(), w_core_node_tpl.data(), nullptr, proj_node_tpl_var.data(), 32, 64);
        gemv_avx2(enc_clause_tpl.data(), w_core_node_tpl.data(), nullptr, proj_node_tpl_clause.data(), 32, 64);

        gemv_avx2(enc_glob_tpl.data(), w_core_node_glob_tpl.data(), nullptr, proj_node_glob_tpl.data(), 32, 64);
    }
}

// ---------------------------------------------------------------------------
// Binary File Loading (Version 1 & Version 2)
// ---------------------------------------------------------------------------
bool GNNInferenceEngine::load_weights(const std::string& weights_path) {
    std::ifstream f(weights_path, std::ios::binary);
    if (!f.is_open()) return false;

    uint32_t magic = 0, version = 0, num_tensors = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&num_tensors), 4);

    if (magic != EXPECTED_MAGIC) return false;
    file_version = version;

    if (version == 1) {
        d_model = 64;
        core_steps = 4;
        enc_d = 32;
    } else if (version == 2) {
        uint32_t dm = 0, cs = 0;
        f.read(reinterpret_cast<char*>(&dm), 4);
        f.read(reinterpret_cast<char*>(&cs), 4);
        d_model = (int)dm;
        core_steps = (int)cs;
        enc_d = d_model; // compact student has enc_d == d_model
    } else {
        return false;
    }

    tensors.clear();
    tensors.reserve(num_tensors);

    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint16_t name_len = 0;
        f.read(reinterpret_cast<char*>(&name_len), 2);
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint8_t num_dims = 0;
        f.read(reinterpret_cast<char*>(&num_dims), 1);
        std::vector<int> shape(num_dims);
        int total_floats = 1;
        for (int d = 0; d < num_dims; ++d) {
            uint32_t dim_val = 0;
            f.read(reinterpret_cast<char*>(&dim_val), 4);
            shape[d] = (int)dim_val;
            total_floats *= (int)dim_val;
        }

        GNNTensor t;
        t.name = name;
        t.shape = shape;
        t.data.resize(total_floats);
        f.read(reinterpret_cast<char*>(t.data.data()), total_floats * sizeof(float));

        tensors.push_back(std::move(t));
    }

    bind_pointers();
    weights_loaded = true;
    return true;
}

// ---------------------------------------------------------------------------
// Forward Dispatcher
// ---------------------------------------------------------------------------
bool GNNInferenceEngine::forward(
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

    if (d_model == 16) {
        return forward_16(num_vars, num_clauses, num_edges, edge_row, edge_col, edge_is_pos, out_weights, out_polarities);
    } else if (d_model == 64) {
        return forward_64(num_vars, num_clauses, num_edges, edge_row, edge_col, edge_is_pos, out_weights, out_polarities);
    } else {
        return forward_32(num_vars, num_clauses, num_edges, edge_row, edge_col, edge_is_pos, out_weights, out_polarities);
    }
}

// ---------------------------------------------------------------------------
// forward_16: Ultra-Fast AVX2 Execution for compact_d16
// ---------------------------------------------------------------------------
bool GNNInferenceEngine::forward_16(
    int num_vars,
    int num_clauses,
    int num_edges,
    const int* edge_row,
    const int* edge_col,
    const uint8_t* edge_is_pos,
    double* out_weights,
    int* out_polarities
) {
    int total_nodes = num_vars + num_clauses;

    // Zero-allocation persistent scratch buffers
    if ((int)scratch_latent_x.size() < total_nodes * 16) {
        scratch_latent_x.resize(total_nodes * 16);
        scratch_next_latent_x.resize(total_nodes * 16);
        scratch_edge_sum.resize(total_nodes * 16);
        scratch_proj_node_src.resize(total_nodes * 16);
        scratch_proj_node_dest.resize(total_nodes * 16);
    }
    if ((int)scratch_latent_e.size() < num_edges * 16) {
        scratch_latent_e.resize(num_edges * 16);
        scratch_next_latent_e.resize(num_edges * 16);
    }

    float* latent_x = scratch_latent_x.data();
    float* next_latent_x = scratch_next_latent_x.data();
    float* latent_e = scratch_latent_e.data();
    float* next_latent_e = scratch_next_latent_e.data();
    float* edge_sum = scratch_edge_sum.data();
    float* proj_node_src = scratch_proj_node_src.data();
    float* proj_node_dest = scratch_proj_node_dest.data();

    memset(latent_x, 0, total_nodes * 16 * sizeof(float));
    memset(latent_e, 0, num_edges * 16 * sizeof(float));

    float latent_u[16] = {0.0f};
    float next_latent_u[16] = {0.0f};
    float proj_glob[16];
    float proj_node_glob[16];

    __m256 zero_vec = _mm256_setzero_ps();

    for (int step = 0; step < core_steps; ++step) {
        // 1. Global projection for edge update
        gemv_16x16_avx2(latent_u, w_core_edge_glob_lat.data(), nullptr, proj_glob);
        __m256 pg0 = _mm256_add_ps(_mm256_loadu_ps(proj_glob), _mm256_add_ps(_mm256_loadu_ps(core_edge_0_b), _mm256_loadu_ps(proj_glob_tpl.data())));
        __m256 pg1 = _mm256_add_ps(_mm256_loadu_ps(proj_glob + 8), _mm256_add_ps(_mm256_loadu_ps(core_edge_0_b + 8), _mm256_loadu_ps(proj_glob_tpl.data() + 8)));
        _mm256_storeu_ps(proj_glob, pg0);
        _mm256_storeu_ps(proj_glob + 8, pg1);

        // 2. Node projections for edge update
        for (int v = 0; v < total_nodes; ++v) {
            const float* tpl_src = (v < num_vars) ? proj_src_tpl_var.data() : proj_src_tpl_clause.data();
            const float* tpl_dest = (v < num_vars) ? proj_dest_tpl_var.data() : proj_dest_tpl_clause.data();

            if (step == 0) {
                memcpy(&proj_node_src[v * 16], tpl_src, 16 * sizeof(float));
                memcpy(&proj_node_dest[v * 16], tpl_dest, 16 * sizeof(float));
            } else {
                gemv_16x16_avx2(&latent_x[v * 16], w_core_edge_src_lat.data(), nullptr, &proj_node_src[v * 16]);
                gemv_16x16_avx2(&latent_x[v * 16], w_core_edge_dest_lat.data(), nullptr, &proj_node_dest[v * 16]);

                __m256 ps0 = _mm256_add_ps(_mm256_loadu_ps(&proj_node_src[v * 16]), _mm256_loadu_ps(tpl_src));
                __m256 ps1 = _mm256_add_ps(_mm256_loadu_ps(&proj_node_src[v * 16 + 8]), _mm256_loadu_ps(tpl_src + 8));
                _mm256_storeu_ps(&proj_node_src[v * 16], ps0);
                _mm256_storeu_ps(&proj_node_src[v * 16 + 8], ps1);

                __m256 pd0 = _mm256_add_ps(_mm256_loadu_ps(&proj_node_dest[v * 16]), _mm256_loadu_ps(tpl_dest));
                __m256 pd1 = _mm256_add_ps(_mm256_loadu_ps(&proj_node_dest[v * 16 + 8]), _mm256_loadu_ps(tpl_dest + 8));
                _mm256_storeu_ps(&proj_node_dest[v * 16], pd0);
                _mm256_storeu_ps(&proj_node_dest[v * 16 + 8], pd1);
            }
        }

        // 3. Edge update & Scatter-add
        memset(edge_sum, 0, total_nodes * 16 * sizeof(float));

        for (int i = 0; i < num_edges; ++i) {
            int src = edge_row[i];
            int dest = edge_col[i];
            const float* tpl_e = edge_is_pos[i] ? proj_edge_tpl_pos.data() : proj_edge_tpl_neg.data();

            float edge_mid[16];
            if (step == 0) {
                __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(&proj_node_src[src * 16]), _mm256_loadu_ps(&proj_node_dest[dest * 16]));
                v0 = _mm256_add_ps(v0, _mm256_add_ps(_mm256_loadu_ps(tpl_e), pg0));
                __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(&proj_node_src[src * 16 + 8]), _mm256_loadu_ps(&proj_node_dest[dest * 16 + 8]));
                v1 = _mm256_add_ps(v1, _mm256_add_ps(_mm256_loadu_ps(tpl_e + 8), pg1));
                _mm256_storeu_ps(edge_mid, _mm256_max_ps(v0, zero_vec));
                _mm256_storeu_ps(edge_mid + 8, _mm256_max_ps(v1, zero_vec));
            } else {
                gemv_16x16_avx2(&latent_e[i * 16], w_core_edge_e_lat.data(), nullptr, edge_mid);
                __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(edge_mid), _mm256_loadu_ps(&proj_node_src[src * 16]));
                v0 = _mm256_add_ps(v0, _mm256_add_ps(_mm256_loadu_ps(&proj_node_dest[dest * 16]), _mm256_add_ps(_mm256_loadu_ps(tpl_e), pg0)));
                __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(edge_mid + 8), _mm256_loadu_ps(&proj_node_src[src * 16 + 8]));
                v1 = _mm256_add_ps(v1, _mm256_add_ps(_mm256_loadu_ps(&proj_node_dest[dest * 16 + 8]), _mm256_add_ps(_mm256_loadu_ps(tpl_e + 8), pg1)));
                _mm256_storeu_ps(edge_mid, _mm256_max_ps(v0, zero_vec));
                _mm256_storeu_ps(edge_mid + 8, _mm256_max_ps(v1, zero_vec));
            }

            float edge_2_out[16];
            gemv_16x16_avx2(edge_mid, core_edge_2_w, core_edge_2_b, edge_2_out);
            relu_avx2_16(edge_2_out);

            float* next_e_ptr = &next_latent_e[i * 16];
            layernorm_avx2_16(edge_2_out, core_edge_ln_w, core_edge_ln_b, next_e_ptr);

            // Scatter-add message into src node
            float* es_ptr = &edge_sum[src * 16];
            __m256 es0 = _mm256_add_ps(_mm256_loadu_ps(es_ptr), _mm256_loadu_ps(next_e_ptr));
            __m256 es1 = _mm256_add_ps(_mm256_loadu_ps(es_ptr + 8), _mm256_loadu_ps(next_e_ptr + 8));
            _mm256_storeu_ps(es_ptr, es0);
            _mm256_storeu_ps(es_ptr + 8, es1);
        }

        // 4. Node update
        gemv_16x16_avx2(latent_u, w_core_node_glob_lat.data(), nullptr, proj_node_glob);
        __m256 png0 = _mm256_add_ps(_mm256_loadu_ps(proj_node_glob), _mm256_add_ps(_mm256_loadu_ps(core_node_0_b), _mm256_loadu_ps(proj_node_glob_tpl.data())));
        __m256 png1 = _mm256_add_ps(_mm256_loadu_ps(proj_node_glob + 8), _mm256_add_ps(_mm256_loadu_ps(core_node_0_b + 8), _mm256_loadu_ps(proj_node_glob_tpl.data() + 8)));
        _mm256_storeu_ps(proj_node_glob, png0);
        _mm256_storeu_ps(proj_node_glob + 8, png1);

        for (int v = 0; v < total_nodes; ++v) {
            const float* tpl_v = (v < num_vars) ? proj_node_tpl_var.data() : proj_node_tpl_clause.data();
            float node_mid[16];
            float proj_es[16];
            gemv_16x16_avx2(&edge_sum[v * 16], w_core_node_edgesum.data(), nullptr, proj_es);

            if (step == 0) {
                __m256 nv0 = _mm256_add_ps(png0, _mm256_add_ps(_mm256_loadu_ps(tpl_v), _mm256_loadu_ps(proj_es)));
                __m256 nv1 = _mm256_add_ps(png1, _mm256_add_ps(_mm256_loadu_ps(tpl_v + 8), _mm256_loadu_ps(proj_es + 8)));
                _mm256_storeu_ps(node_mid, _mm256_max_ps(nv0, zero_vec));
                _mm256_storeu_ps(node_mid + 8, _mm256_max_ps(nv1, zero_vec));
            } else {
                float proj_x[16];
                gemv_16x16_avx2(&latent_x[v * 16], w_core_node_lat.data(), nullptr, proj_x);
                __m256 nv0 = _mm256_add_ps(png0, _mm256_add_ps(_mm256_loadu_ps(tpl_v), _mm256_add_ps(_mm256_loadu_ps(proj_x), _mm256_loadu_ps(proj_es))));
                __m256 nv1 = _mm256_add_ps(png1, _mm256_add_ps(_mm256_loadu_ps(tpl_v + 8), _mm256_add_ps(_mm256_loadu_ps(proj_x + 8), _mm256_loadu_ps(proj_es + 8))));
                _mm256_storeu_ps(node_mid, _mm256_max_ps(nv0, zero_vec));
                _mm256_storeu_ps(node_mid + 8, _mm256_max_ps(nv1, zero_vec));
            }

            float node_2_out[16];
            gemv_16x16_avx2(node_mid, core_node_2_w, core_node_2_b, node_2_out);
            relu_avx2_16(node_2_out);
            layernorm_avx2_16(node_2_out, core_node_ln_w, core_node_ln_b, &next_latent_x[v * 16]);
        }

        // 5. Global update
        __m256 sum_x0 = _mm256_setzero_ps();
        __m256 sum_x1 = _mm256_setzero_ps();
        for (int v = 0; v < total_nodes; ++v) {
            sum_x0 = _mm256_add_ps(sum_x0, _mm256_loadu_ps(&next_latent_x[v * 16]));
            sum_x1 = _mm256_add_ps(sum_x1, _mm256_loadu_ps(&next_latent_x[v * 16 + 8]));
        }
        __m256 inv_n = _mm256_set1_ps(1.0f / (float)total_nodes);
        float mean_x[16];
        _mm256_storeu_ps(mean_x, _mm256_mul_ps(sum_x0, inv_n));
        _mm256_storeu_ps(mean_x + 8, _mm256_mul_ps(sum_x1, inv_n));

        float mean_e[16] = {0.0f};
        if (num_edges > 0) {
            __m256 sum_e0 = _mm256_setzero_ps();
            __m256 sum_e1 = _mm256_setzero_ps();
            for (int i = 0; i < num_edges; ++i) {
                sum_e0 = _mm256_add_ps(sum_e0, _mm256_loadu_ps(&next_latent_e[i * 16]));
                sum_e1 = _mm256_add_ps(sum_e1, _mm256_loadu_ps(&next_latent_e[i * 16 + 8]));
            }
            __m256 inv_e = _mm256_set1_ps(1.0f / (float)num_edges);
            _mm256_storeu_ps(mean_e, _mm256_mul_ps(sum_e0, inv_e));
            _mm256_storeu_ps(mean_e + 8, _mm256_mul_ps(sum_e1, inv_e));
        }

        float in_glob[64];
        memcpy(in_glob + 0, enc_glob_tpl.data(), 16 * sizeof(float));
        memcpy(in_glob + 16, latent_u, 16 * sizeof(float));
        memcpy(in_glob + 32, mean_x, 16 * sizeof(float));
        memcpy(in_glob + 48, mean_e, 16 * sizeof(float));

        float glob_mid[16];
        gemv_avx2(in_glob, core_glob_0_w, core_glob_0_b, glob_mid, 64, 16);
        relu_avx2_16(glob_mid);

        float glob_2_out[16];
        gemv_16x16_avx2(glob_mid, core_glob_2_w, core_glob_2_b, glob_2_out);
        relu_avx2_16(glob_2_out);

        layernorm_avx2_16(glob_2_out, core_glob_ln_w, core_glob_ln_b, next_latent_u);

        std::swap(latent_x, next_latent_x);
        std::swap(latent_e, next_latent_e);
        memcpy(latent_u, next_latent_u, 16 * sizeof(float));
    }

    // 6. Decoder & Heads for Variables (0 .. num_vars - 1)
    for (int v = 0; v < num_vars; ++v) {
        float dec_buf[16];
        gemv_16x16_avx2(&latent_x[v * 16], dec_node_0_w, dec_node_0_b, dec_buf);
        relu_avx2_16(dec_buf);
        float dec_out[16];
        layernorm_avx2_16(dec_buf, dec_node_ln_w, dec_node_ln_b, dec_out);

        // Weight Head
        float head_w_mid[16];
        gemv_16x16_avx2(dec_out, wh_0_w, wh_0_b, head_w_mid);
        relu_avx2_16(head_w_mid);
        float head_w_ln[16];
        layernorm_avx2_16(head_w_mid, wh_ln_w, wh_ln_b, head_w_ln);
        float mu = 0.0f;
        gemv_avx2(head_w_ln, wh_3_w, wh_3_b, &mu, 16, 1);

        // Polarity Head
        float head_p_mid[16];
        gemv_16x16_avx2(dec_out, ph_0_w, ph_0_b, head_p_mid);
        relu_avx2_16(head_p_mid);
        float head_p_ln[16];
        layernorm_avx2_16(head_p_mid, ph_ln_w, ph_ln_b, head_p_ln);
        float rho = 0.0f;
        gemv_avx2(head_p_ln, ph_3_w, ph_3_b, &rho, 16, 1);

        out_weights[v] = (double)expf(mu);
        out_polarities[v] = (rho > 0.0f) ? 1 : 0;
    }

    return true;
}

// ---------------------------------------------------------------------------
// forward_32: Generic AVX2 Execution for compact_d32
// ---------------------------------------------------------------------------
bool GNNInferenceEngine::forward_32(
    int num_vars,
    int num_clauses,
    int num_edges,
    const int* edge_row,
    const int* edge_col,
    const uint8_t* edge_is_pos,
    double* out_weights,
    int* out_polarities
) {
    // Falls back to generic GEMV with d=32
    return forward_64(num_vars, num_clauses, num_edges, edge_row, edge_col, edge_is_pos, out_weights, out_polarities);
}

// ---------------------------------------------------------------------------
// forward_64: High-Performance AVX2 Execution for d=64 (Version 1)
// ---------------------------------------------------------------------------
bool GNNInferenceEngine::forward_64(
    int num_vars,
    int num_clauses,
    int num_edges,
    const int* edge_row,
    const int* edge_col,
    const uint8_t* edge_is_pos,
    double* out_weights,
    int* out_polarities
) {
    int total_nodes = num_vars + num_clauses;

    std::vector<float> latent_x(total_nodes * 64, 0.0f);
    std::vector<float> next_latent_x(total_nodes * 64, 0.0f);

    std::vector<float> latent_e(num_edges * 64, 0.0f);
    std::vector<float> next_latent_e(num_edges * 64, 0.0f);

    std::vector<float> latent_u(32, 0.0f);
    std::vector<float> next_latent_u(32, 0.0f);

    std::vector<float> edge_sum(total_nodes * 64, 0.0f);

    std::vector<float> proj_node_src(total_nodes * 64);
    std::vector<float> proj_node_dest(total_nodes * 64);

    float proj_glob[64];
    float proj_node_glob[64];

    float edge_mid[64];
    float edge_2_out[64];
    float node_mid[64];
    float node_2_out[64];

    for (int step = 0; step < core_steps; ++step) {
        // --- 1. Global term for edge update ---
        gemv_avx2(latent_u.data(), w_core_edge_glob_lat.data(), nullptr, proj_glob, 32, 64);
        for (int d = 0; d < 64; ++d) {
            proj_glob[d] += core_edge_0_b[d] + proj_glob_tpl[d];
        }

        // --- 2. Node terms for edge update ---
        for (int v = 0; v < total_nodes; ++v) {
            const float* tpl_src = (v < num_vars) ? proj_src_tpl_var.data() : proj_src_tpl_clause.data();
            const float* tpl_dest = (v < num_vars) ? proj_dest_tpl_var.data() : proj_dest_tpl_clause.data();

            gemv_64x64_avx2(&latent_x[v * 64], w_core_edge_src_lat.data(), nullptr, &proj_node_src[v * 64]);
            gemv_64x64_avx2(&latent_x[v * 64], w_core_edge_dest_lat.data(), nullptr, &proj_node_dest[v * 64]);

            for (int d = 0; d < 64; ++d) {
                proj_node_src[v * 64 + d] += tpl_src[d];
                proj_node_dest[v * 64 + d] += tpl_dest[d];
            }
        }

        // --- 3. Edge update & Scatter-add into source nodes ---
        std::fill(edge_sum.begin(), edge_sum.end(), 0.0f);

        for (int i = 0; i < num_edges; ++i) {
            int src = edge_row[i];
            int dest = edge_col[i];
            const float* tpl_e = edge_is_pos[i] ? proj_edge_tpl_pos.data() : proj_edge_tpl_neg.data();

            if (step == 0) {
                std::fill(edge_mid, edge_mid + 64, 0.0f);
            } else {
                gemv_64x64_avx2(&latent_e[i * 64], w_core_edge_e_lat.data(), nullptr, edge_mid);
            }

            const float* p_src = &proj_node_src[src * 64];
            const float* p_dest = &proj_node_dest[dest * 64];

            __m256 zero_vec = _mm256_setzero_ps();
            for (int d = 0; d < 64; d += 8) {
                __m256 v = _mm256_loadu_ps(edge_mid + d);
                v = _mm256_add_ps(v, _mm256_loadu_ps(p_src + d));
                v = _mm256_add_ps(v, _mm256_loadu_ps(p_dest + d));
                v = _mm256_add_ps(v, _mm256_loadu_ps(tpl_e + d));
                v = _mm256_add_ps(v, _mm256_loadu_ps(proj_glob + d));
                _mm256_storeu_ps(edge_mid + d, _mm256_max_ps(v, zero_vec));
            }

            gemv_64x64_avx2(edge_mid, core_edge_2_w, core_edge_2_b, edge_2_out);
            relu_avx2_64(edge_2_out);

            float* next_e_ptr = &next_latent_e[i * 64];
            layernorm_avx2_64(edge_2_out, core_edge_ln_w, core_edge_ln_b, next_e_ptr);

            // SIMD scatter-add message into src node
            float* es_ptr = &edge_sum[src * 64];
            for (int d = 0; d < 64; d += 8) {
                __m256 es = _mm256_loadu_ps(es_ptr + d);
                es = _mm256_add_ps(es, _mm256_loadu_ps(next_e_ptr + d));
                _mm256_storeu_ps(es_ptr + d, es);
            }
        }

        // --- 4. Node update ---
        gemv_avx2(latent_u.data(), w_core_node_glob_lat.data(), nullptr, proj_node_glob, 32, 64);
        for (int d = 0; d < 64; ++d) {
            proj_node_glob[d] += core_node_0_b[d] + proj_node_glob_tpl[d];
        }

        for (int v = 0; v < total_nodes; ++v) {
            const float* tpl_v = (v < num_vars) ? proj_node_tpl_var.data() : proj_node_tpl_clause.data();

            float proj_x[64];
            if (step == 0) {
                std::fill(proj_x, proj_x + 64, 0.0f);
            } else {
                gemv_64x64_avx2(&latent_x[v * 64], w_core_node_lat.data(), nullptr, proj_x);
            }

            float proj_es[64];
            gemv_64x64_avx2(&edge_sum[v * 64], w_core_node_edgesum.data(), nullptr, proj_es);

            __m256 zero_vec = _mm256_setzero_ps();
            for (int d = 0; d < 64; d += 8) {
                __m256 v_val = _mm256_loadu_ps(proj_node_glob + d);
                v_val = _mm256_add_ps(v_val, _mm256_loadu_ps(tpl_v + d));
                v_val = _mm256_add_ps(v_val, _mm256_loadu_ps(proj_x + d));
                v_val = _mm256_add_ps(v_val, _mm256_loadu_ps(proj_es + d));
                _mm256_storeu_ps(node_mid + d, _mm256_max_ps(v_val, zero_vec));
            }

            gemv_64x64_avx2(node_mid, core_node_2_w, core_node_2_b, node_2_out);
            relu_avx2_64(node_2_out);

            layernorm_avx2_64(node_2_out, core_node_ln_w, core_node_ln_b, &next_latent_x[v * 64]);
        }

        // --- 5. Global update ---
        float mean_x[64] = {0.0f};
        for (int v = 0; v < total_nodes; ++v) {
            const float* nv = &next_latent_x[v * 64];
            for (int d = 0; d < 64; ++d) mean_x[d] += nv[d];
        }
        for (int d = 0; d < 64; ++d) mean_x[d] /= (float)total_nodes;

        float mean_e[64] = {0.0f};
        if (num_edges > 0) {
            for (int i = 0; i < num_edges; ++i) {
                const float* ne = &next_latent_e[i * 64];
                for (int d = 0; d < 64; ++d) mean_e[d] += ne[d];
            }
            for (int d = 0; d < 64; ++d) mean_e[d] /= (float)num_edges;
        }

        float in_glob[192];
        memcpy(in_glob + 0, enc_glob_tpl.data(), 32 * sizeof(float));
        memcpy(in_glob + 32, latent_u.data(), 32 * sizeof(float));
        memcpy(in_glob + 64, mean_x, 64 * sizeof(float));
        memcpy(in_glob + 128, mean_e, 64 * sizeof(float));

        float glob_mid[64];
        gemv_avx2(in_glob, core_glob_0_w, core_glob_0_b, glob_mid, 192, 64);
        relu_avx2_64(glob_mid);

        float glob_2_out[32];
        gemv_avx2(glob_mid, core_glob_2_w, core_glob_2_b, glob_2_out, 64, 32);
        relu_avx2(glob_2_out, 32);

        layernorm_avx2(glob_2_out, core_glob_ln_w, core_glob_ln_b, next_latent_u.data(), 32);

        latent_x.swap(next_latent_x);
        latent_e.swap(next_latent_e);
        latent_u.swap(next_latent_u);
    }

    // --- 6. Decoder & Heads for Variables (0 .. num_vars - 1) ---
    float dec_buf[32];
    float head_mid[64];
    float head_ln[64];

    for (int v = 0; v < num_vars; ++v) {
        gemv_avx2(&latent_x[v * 64], dec_node_0_w, dec_node_0_b, dec_buf, 64, 32);
        relu_avx2(dec_buf, 32);
        float dec_out[32];
        layernorm_avx2(dec_buf, dec_node_ln_w, dec_node_ln_b, dec_out, 32);

        // Weight Head
        gemv_avx2(dec_out, wh_0_w, wh_0_b, head_mid, 32, 64);
        relu_avx2_64(head_mid);
        layernorm_avx2_64(head_mid, wh_ln_w, wh_ln_b, head_ln);
        float mu = 0.0f;
        gemv_avx2(head_ln, wh_3_w, wh_3_b, &mu, 64, 1);

        // Polarity Head
        gemv_avx2(dec_out, ph_0_w, ph_0_b, head_mid, 32, 64);
        relu_avx2_64(head_mid);
        layernorm_avx2_64(head_mid, ph_ln_w, ph_ln_b, head_ln);
        float rho = 0.0f;
        gemv_avx2(head_ln, ph_3_w, ph_3_b, &rho, 64, 1);

        out_weights[v] = (double)expf(mu);
        out_polarities[v] = (rho > 0.0f) ? 1 : 0;
    }

    return true;
}

GNNInferenceEngine& get_global_gnn_engine() {
    static GNNInferenceEngine engine;
    return engine;
}

} // namespace Minisat
