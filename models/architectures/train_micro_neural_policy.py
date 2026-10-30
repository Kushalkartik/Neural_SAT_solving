import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Kushal Agarwal\Desktop\DAC-203 PROJECT\28_NEURAL_CAUSAL_FLOW_KISSAT")
TRACE_FILE = BASE_DIR / "training_traces.bin"
MNCF_SRC_DIR = BASE_DIR / "kissat_mncf" / "src"
WEIGHTS_C_FILE = MNCF_SRC_DIR / "micro_neural_weights.c"
MODEL_PT_FILE = BASE_DIR / "micro_neural_policy_bounded.pt"

MAX_BOOST = 0.45

class BoundedMicroNeuralFlow(nn.Module):
    def __init__(self, in_dim=4, h1_dim=8, h2_dim=4, max_boost=0.45):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, 1)
        self.max_boost = max_boost

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        z = self.fc3(h2)
        phi = 1.0 + self.max_boost * torch.sigmoid(z)
        return phi, z

def export_c_weights(model, filepath):
    w1 = model.fc1.weight.detach().cpu().numpy()
    b1 = model.fc1.bias.detach().cpu().numpy()
    w2 = model.fc2.weight.detach().cpu().numpy()
    b2 = model.fc2.bias.detach().cpu().numpy()
    w3 = model.fc3.weight.detach().cpu().numpy()
    b3 = model.fc3.bias.detach().cpu().numpy()

    lines = []
    lines.append('#include "micro_neural_net.h"\n')
    lines.append('// Auto-generated Bounded Micro-Neural Causal Flow weights (PyTorch AdamW)')
    lines.append(f'// Trained on {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    # W1
    lines.append('const float mncf_W1[MNCF_H1_DIM][MNCF_INPUT_DIM] = {')
    for row in w1:
        row_str = ", ".join(f"{v:10.6f}f" for v in row)
        lines.append(f"  {{ {row_str} }},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append('};\n')

    # b1
    b1_str = ", ".join(f"{v:10.6f}f" for v in b1)
    lines.append(f'const float mncf_b1[MNCF_H1_DIM] = {{\n  {b1_str}\n}};\n')

    # W2
    lines.append('const float mncf_W2[MNCF_H2_DIM][MNCF_H1_DIM] = {')
    for row in w2:
        row_str = ", ".join(f"{v:10.6f}f" for v in row)
        lines.append(f"  {{ {row_str} }},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append('};\n')

    # b2
    b2_str = ", ".join(f"{v:10.6f}f" for v in b2)
    lines.append(f'const float mncf_b2[MNCF_H2_DIM] = {{\n  {b2_str}\n}};\n')

    # W3
    lines.append('const float mncf_W3[1][MNCF_H2_DIM] = {')
    for row in w3:
        row_str = ", ".join(f"{v:10.6f}f" for v in row)
        lines.append(f"  {{ {row_str} }}")
    lines.append('};\n')

    # b3
    b3_str = f"{b3[0]:10.6f}f"
    lines.append(f'const float mncf_b3[1] = {{\n  {b3_str}\n}};\n')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Exported compiled C weights to: {filepath}")

def main():
    print("=" * 70)
    print("BOUNDED MICRO-NEURAL CAUSAL FLOW (MNCF): PYTORCH TRAINING PIPELINE")
    print(f"Target Stability Range: [1.00, {1.0+MAX_BOOST:.2f}]")
    print("=" * 70)

    if not TRACE_FILE.exists():
        print(f"Error: {TRACE_FILE} does not exist.")
        sys.exit(1)

    raw_data = np.fromfile(str(TRACE_FILE), dtype=np.float32)
    valid_len = (len(raw_data) // 5) * 5
    raw_data = raw_data[:valid_len]
    num_samples = valid_len // 5
    print(f"Loaded {num_samples:,} complete conflict transition records.")

    data = raw_data.reshape(num_samples, 5)
    x0 = np.clip(data[:, 0], 0.0, 5.0)  # log(1+w)
    x1 = np.clip(data[:, 1], 0.0, 1.0)  # inv size
    x2 = np.clip(data[:, 2], 0.0, 1.0)  # rel level
    x3 = np.clip(data[:, 3], 0.0, 1.0)  # score ratio

    X = np.stack([x0, x1, x2, x3], axis=1)

    # Bounded target boost in [0.0, 0.45]
    # Smooth tanh saturation prevents over-concentration
    target_boost = MAX_BOOST * np.tanh(0.6 * x0 * (1.0 + 0.3 * x1 + 0.2 * (1.0 - x3) + 0.1 * x2))
    Y = target_boost[:, None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    MAX_SAMPLES = 200000
    indices = np.random.randint(0, len(X), size=MAX_SAMPLES)
    X_sub = X[indices]
    Y_sub = Y[indices]

    n_train = int(0.8 * len(X_sub))
    X_train, X_val = torch.tensor(X_sub[:n_train], dtype=torch.float32), torch.tensor(X_sub[n_train:], dtype=torch.float32)
    Y_train, Y_val = torch.tensor(Y_sub[:n_train], dtype=torch.float32), torch.tensor(Y_sub[n_train:], dtype=torch.float32)

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    batch_size = 1024
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BoundedMicroNeuralFlow(max_boost=MAX_BOOST).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    EPOCHS = 35
    print(f"\nTraining Bounded 4 -> 8 -> 4 -> 1 Neural Policy (81 params)...")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            phi, z = model(bx)
            pred_boost = MAX_BOOST * torch.sigmoid(z)
            loss = F.mse_loss(pred_boost, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                phi, z = model(bx)
                pred_boost = MAX_BOOST * torch.sigmoid(z)
                loss = F.mse_loss(pred_boost, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_ds)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(MODEL_PT_FILE))

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.6f}")
    model.load_state_dict(torch.load(str(MODEL_PT_FILE)))
    export_c_weights(model, WEIGHTS_C_FILE)

    print("\nVerifying Bounded Micro-Neural Policy outputs on canonical test vectors:")
    test_cases = [
        ("Zero flow baseline", [0.0, 1.0, 0.5, 0.5]),
        ("Moderate flow",      [0.693, 0.5, 0.8, 0.2]),
        ("High bottleneck",    [1.609, 0.8, 0.9, 0.2]),
        ("Extreme bottleneck", [2.500, 1.0, 1.0, 0.1]),
    ]
    model.eval()
    with torch.no_grad():
        for name, vec in test_cases:
            t_x = torch.tensor([vec], dtype=torch.float32).to(device)
            phi, z = model(t_x)
            print(f" - {name:20s}: x={vec} -> z={z.item():.4f}, Phi(x)={phi.item():.4f} (Boost: +{(phi.item()-1.0)*100:.1f}%)")

if __name__ == "__main__":
    main()
