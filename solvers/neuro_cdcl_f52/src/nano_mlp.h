/* Generated Nano-MLP Weights and SIMD Forward Pass Header */
#ifndef _nano_mlp_h_INCLUDED
#define _nano_mlp_h_INCLUDED

#include <math.h>

#define NANO_MLP_INPUT_DIM 6
#define NANO_MLP_HIDDEN_DIM 12
#define NANO_MLP_OUTPUT_DIM 4

static const float NANO_MLP_W1[12][6] = {
  {0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.5000f, 0.0000f},
  {0.0000f, 0.0000f, 0.0000f, 0.0000f, 3.0000f, 0.0000f},
  {0.0000f, 0.0000f, 0.0000f, 4.0000f, 0.0000f, 0.0000f},
  {-2.0000f, 2.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
  {-2.5000f, 1.5000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
  {0.0000f, 2.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
  {2.5000f, 0.0000f, 3.0000f, 0.0000f, 0.0000f, 0.0000f},
  {2.0000f, 0.0000f, 2.0000f, 0.0000f, 0.0000f, 0.0000f},
  {0.0000f, 0.0000f, 3.5000f, 0.0000f, 0.0000f, 0.0000f},
  {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 3.0000f},
  {1.5000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.0000f},
  {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 3.5000f},
};

static const float NANO_MLP_B1[12] = {
  -0.5000f, -0.7500f, -0.2000f, 0.5000f, 0.2000f, -1.5000f, -0.5000f, 0.0000f, -0.3000f, -0.8000f, -0.5000f, -1.2000f
};

static const float NANO_MLP_W2[4][12] = {
  {1.2000f, 1.5000f, 0.8000f, 1.0000f, 0.0000f, 0.8000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
  {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 1.2000f, 1.5000f, 0.0000f, 0.0000f, 0.0000f},
  {0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 0.0000f, 0.8000f, 0.0000f, 0.0000f, 1.2000f, 1.0000f, 0.0000f},
  {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 1.0000f, 1.5000f},
};

static const float NANO_MLP_B2[4] = {
  0.1000f, 0.3000f, 0.1000f, 0.0000f
};

static inline void nano_mlp_forward (const float *s, float *q_neural) {
  float h[NANO_MLP_HIDDEN_DIM];
  for (int j = 0; j < NANO_MLP_HIDDEN_DIM; j++) {
    float sum = NANO_MLP_B1[j];
    for (int k = 0; k < NANO_MLP_INPUT_DIM; k++) {
      sum += NANO_MLP_W1[j][k] * s[k];
    }
    h[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU activation
  }

  float q_logits[NANO_MLP_OUTPUT_DIM];
  float max_l = -1e9f;
  for (int j = 0; j < NANO_MLP_OUTPUT_DIM; j++) {
    float sum = NANO_MLP_B2[j];
    for (int k = 0; k < NANO_MLP_HIDDEN_DIM; k++) {
      sum += NANO_MLP_W2[j][k] * h[k];
    }
    q_logits[j] = sum;
    if (sum > max_l) max_l = sum;
  }

  float exp_sum = 0.0f;
  float q_exp[NANO_MLP_OUTPUT_DIM];
  for (int j = 0; j < NANO_MLP_OUTPUT_DIM; j++) {
    q_exp[j] = expf (q_logits[j] - max_l);
    exp_sum += q_exp[j];
  }
  for (int j = 0; j < NANO_MLP_OUTPUT_DIM; j++) {
    q_neural[j] = q_exp[j] / (exp_sum > 0.0f ? exp_sum : 1.0f);
  }
}
#endif
