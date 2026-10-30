import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Kushal Agarwal\Desktop\DAC-203 PROJECT\28_NEURAL_CAUSAL_FLOW_KISSAT")
TRACE_FILE = BASE_DIR / "training_traces.bin"
MNCF_SRC_DIR = BASE_DIR / "kissat_mncf" / "src"

MAX_BOOST = 0.45

class Neural1DFlowPolicy(nn.Module):
    def __init__(self, h_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(1, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        phi = 1.0 + MAX_BOOST * torch.sigmoid(z)
        return phi, z

def main():
    print("Training 1D Micro-Neural Flow Policy on Industrial Conflict Traces...")
    raw = np.fromfile(str(TRACE_FILE), dtype=np.float32)
    valid_len = (len(raw) // 5) * 5
    data = raw[:valid_len].reshape(valid_len // 5, 5)

    x0 = np.clip(data[:, 0:1], 0.0, 5.0) # log(1+w)
    # Target: bounded structural reward in [0, 0.45]
    y_target = MAX_BOOST * np.tanh(0.7 * x0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Subsample 200k samples
    idx = np.random.randint(0, len(x0), size=200000)
    X = torch.tensor(x0[idx], dtype=torch.float32)
    Y = torch.tensor(y_target[idx], dtype=torch.float32)

    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=1024, shuffle=True)

    model = Neural1DFlowPolicy(h_dim=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            phi, z = model(bx)
            pred_boost = MAX_BOOST * torch.sigmoid(z)
            loss = F.mse_loss(pred_boost, by)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(bx)
        total_loss /= len(ds)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | Loss: {total_loss:.6f}")

    # Export weights
    w1 = model.fc1.weight.detach().cpu().numpy() # [4, 1]
    b1 = model.fc1.bias.detach().cpu().numpy()   # [4]
    w2 = model.fc2.weight.detach().cpu().numpy() # [1, 4]
    b2 = model.fc2.bias.detach().cpu().numpy()   # [1]

    # Generate micro_neural_net.h
    header_code = f'''#ifndef _micro_neural_net_h_INCLUDED
#define _micro_neural_net_h_INCLUDED

#include <math.h>

// 1D Micro-Neural Causal Flow Policy (Trained with PyTorch AdamW)
// Guaranteed bounded non-suppressive modulation: Phi in [1.00, 1.45]
static inline float evaluate_micro_neural_flow_1d (float log_w) {{
  if (log_w <= 0.0f) return 1.0f;

  const float w1[4] = {{ {w1[0][0]:.6f}f, {w1[1][0]:.6f}f, {w1[2][0]:.6f}f, {w1[3][0]:.6f}f }};
  const float b1[4] = {{ {b1[0]:.6f}f, {b1[1]:.6f}f, {b1[2]:.6f}f, {b1[3]:.6f}f }};
  const float w2[4] = {{ {w2[0][0]:.6f}f, {w2[0][1]:.6f}f, {w2[0][2]:.6f}f, {w2[0][3]:.6f}f }};
  const float b2 = {b2[0]:.6f}f;

  float h0 = w1[0] * log_w + b1[0]; if (h0 < 0.0f) h0 = 0.0f;
  float h1 = w1[1] * log_w + b1[1]; if (h1 < 0.0f) h1 = 0.0f;
  float h2 = w1[2] * log_w + b1[2]; if (h2 < 0.0f) h2 = 0.0f;
  float h3 = w1[3] * log_w + b1[3]; if (h3 < 0.0f) h3 = 0.0f;

  float z = b2 + w2[0] * h0 + w2[1] * h1 + w2[2] * h2 + w2[3] * h3;

  float sig;
  if (z > 10.0f) sig = 1.0f;
  else if (z < -10.0f) sig = 0.0f;
  else sig = 1.0f / (1.0f + expf (-z));

  return 1.0f + 0.45f * sig;
}}

#endif
'''
    with open(MNCF_SRC_DIR / "micro_neural_net.h", "w", encoding="utf-8") as f:
        f.write(header_code)
    print("Exported 1D micro-neural header to micro_neural_net.h")

    # Verification on test vectors
    print("\nVerifying 1D Neural Policy:")
    for w_val in [0.0, 0.5, 1.0, 2.0, 3.5, 5.0]:
        t_x = torch.tensor([[np.log1p(w_val)]], dtype=torch.float32).to(device)
        phi, z = model(t_x)
        print(f" w={w_val:4.1f} -> log(1+w)={np.log1p(w_val):.4f} -> Phi={phi.item():.4f} (Boost: +{(phi.item()-1.0)*100:.1f}%)")

if __name__ == "__main__":
    main()
