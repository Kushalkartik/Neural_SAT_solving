/* Distilled Nano-MLP: restart */
/* Parameters: 81 floats | Sub-5ns latency */
#ifndef NANO_RESTART_NET_H
#define NANO_RESTART_NET_H

#include <math.h>

static const float restart_w1[8][8] = {
  { -0.12856004f, +0.89555693f, -0.03637873f, -0.00687296f, -0.11743117f, -1.08402872f, +1.06775248f, -3.27377772f },
  { +0.02826460f, -0.25050849f, +0.01737823f, -0.26411521f, +0.16921481f, +0.18442562f, +0.12544957f, +0.12760571f },
  { -0.68716550f, -1.28659320f, -12.36258793f, +0.45424208f, -0.27042192f, +0.73198938f, +0.56339025f, -2.43056297f },
  { +0.07403849f, -0.03474989f, +0.03407104f, +0.01627821f, -0.00405745f, -0.11159936f, -0.20076436f, -25.66685295f },
  { -0.00076393f, -0.31376600f, +0.13420784f, -0.30130893f, -0.02617215f, -0.05582291f, +0.05270517f, -0.33055183f },
  { -0.13432308f, +0.88735873f, -0.00136086f, +0.26412347f, -0.01290294f, -0.25718573f, +1.38414192f, -1.10694277f },
  { -0.00990879f, +0.02292095f, -0.03620490f, -0.08767266f, -0.00658423f, -0.08241850f, -0.17395125f, -14.89207745f },
  { -1.65184283f, +0.16793202f, -16.44029617f, +0.18463941f, +0.05646801f, +0.30910727f, +2.73931599f, +0.12950480f },
};

static const float restart_b1[8] = { +0.37390265f, -0.30906597f, +0.06428659f, +0.25041074f, -0.12259945f, -1.19900870f, +0.40677354f, -0.53538162f };

static const float restart_w2[8] = { -0.47061130f, -0.08304125f, +0.85171521f, -7.09529257f, -0.26865426f, +0.73318231f, -2.39200330f, -2.03700662f };
static const float restart_b2 = -0.25472271f;

static inline float nano_restart_net_forward (const float *in) {
  float hidden[8];
  for (unsigned j = 0; j < 8; j++) {
    float sum = restart_b1[j];
    for (unsigned i = 0; i < 8; i++) {
      sum += in[i] * restart_w1[j][i];
    }
    hidden[j] = sum > 0.0f ? sum : 0.0f;
  }
  float out = restart_b2;
  for (unsigned j = 0; j < 8; j++) {
    out += hidden[j] * restart_w2[j];
  }
  return 0.5f + 0.5f * (out / (1.0f + fabsf (out)));
}

#endif // NANO_RESTART_NET_H