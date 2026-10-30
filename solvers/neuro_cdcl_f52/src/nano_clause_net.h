/* Distilled Nano-MLP: clause */
/* Parameters: 65 floats | Sub-5ns latency */
#ifndef NANO_CLAUSE_NET_H
#define NANO_CLAUSE_NET_H

#include <math.h>

static const float clause_w1[8][6] = {
  { +0.09239823f, +0.15248166f, -0.20522539f, -0.15960732f, -0.33976147f, +0.14315988f },
  { +1.46576965f, +0.90634024f, -2.96947718f, +0.13470033f, -0.19896004f, +0.40010914f },
  { -0.36107129f, -0.23110990f, +0.48089126f, +0.76842070f, -0.08537748f, +0.43762296f },
  { -0.20257694f, -0.04487705f, +0.64357752f, -0.76808149f, +0.87549919f, +0.03677163f },
  { +1.74066031f, +1.08259404f, -0.03736860f, -0.22689457f, +0.36586681f, -0.05712420f },
  { +1.36844063f, -0.37539670f, +0.63437176f, -0.17597754f, -0.88348514f, +0.26182133f },
  { +4.35025263f, -0.27289811f, +0.66046947f, -0.09788972f, -0.02131313f, +0.20653796f },
  { -0.34667084f, -0.37541771f, +0.22236679f, +0.95047748f, +0.23730496f, +0.23361625f },
};

static const float clause_b1[8] = { -0.29227239f, +0.11774021f, +0.11501826f, -0.02135856f, +0.19107324f, +0.58287102f, -0.26447314f, +0.05810357f };

static const float clause_w2[8] = { +0.00702716f, -1.98207390f, +0.90384656f, +1.37213016f, -0.65570140f, +1.30523193f, -3.13724685f, +0.58256400f };
static const float clause_b2 = -0.46325111f;

static inline float nano_clause_net_forward (const float *in) {
  float hidden[8];
  for (unsigned j = 0; j < 8; j++) {
    float sum = clause_b1[j];
    for (unsigned i = 0; i < 6; i++) {
      sum += in[i] * clause_w1[j][i];
    }
    hidden[j] = sum > 0.0f ? sum : 0.0f;
  }
  float out = clause_b2;
  for (unsigned j = 0; j < 8; j++) {
    out += hidden[j] * clause_w2[j];
  }
  return 0.5f + 0.5f * (out / (1.0f + fabsf (out)));
}

static inline float evaluate_nano_clause_quality (const float *feats) {
  return nano_clause_net_forward (feats);
}

#endif // NANO_CLAUSE_NET_H