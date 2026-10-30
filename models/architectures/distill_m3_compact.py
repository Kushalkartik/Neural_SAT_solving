"""Milestone M3: High-Speed Precomputed Distillation of Ultra-Fast Compact GNNs.

Precomputes teacher targets (mu, rho) once from checkpoint_seed43.pt on 05_DATA/M2B_MULTISCALE/m2b_feedback_dataset.pt.
Trains lightweight student networks (d=16/32, T=1/2) with AdamW in seconds.
Evaluates all students on the 100 held-out SATLIB uf50-218 benchmark in MiniSat.
"""

import sys
import os
import time
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_src_dir = Path(__file__).resolve().parents[2]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from models.m2c_rlaf.rlaf_policy_network import RLAFPolicyNetwork
from sat.solver_adapter.minisat_adapter import MiniSatAdapter
from graph.construction.graph_builder import build_graph_state_from_adapter

def create_compact_network(d: int = 16, core_steps: int = 1, hidden_size: int = 16) -> RLAFPolicyNetwork:
    return RLAFPolicyNetwork(
        in_dims=(2, 2, 1),
        encoder_out_dims=(d, d, d),
        core_out_dims=(d, d, d),
        decoder_out_dims=(d, 1, 1),
        core_steps=core_steps,
        hidden_size=hidden_size,
        zero_init=False,
    )

def evaluate_on_satlib(policy: nn.Module, device: torch.device, num_instances: int = 100):
    policy.eval()
    test_manifest_path = Path("05_DATA/SPLITS/M1_REPRODUCTION/test_manifest.csv")
    import csv
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:num_instances]
    
    decisions_list = []
    conflicts_list = []
    
    for row in rows:
        fpath = row["relative_path"]
        with MiniSatAdapter() as adapter:
            adapter.load_cnf(fpath, simplify=False)
            graph_state = build_graph_state_from_adapter(adapter)
            v = torch.from_numpy(graph_state.vertex_data).to(device)
            e = torch.from_numpy(graph_state.edge_data).to(device)
            c = torch.from_numpy(graph_state.connectivity).to(device)
            u = torch.from_numpy(graph_state.global_attr).to(device)
            
            with torch.no_grad():
                out = policy(v, c, e, u)
                w = torch.exp(out["mu"]).view(-1)
                p = (out["rho"].view(-1) > 0).int()
                
            w_list = w.cpu().numpy().tolist()
            p_list = p.cpu().numpy().tolist()
            
            adapter.set_rlaf_guidance(w_list, p_list)
            status, stats = adapter.solve_continuous()
            decisions_list.append(stats.decisions)
            conflicts_list.append(stats.conflicts)
            
    return {
        "mean_decisions": float(np.mean(decisions_list)),
        "median_decisions": float(np.median(decisions_list)),
        "mean_conflicts": float(np.mean(conflicts_list)),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M3 Distillation] Training device: {device}", flush=True)
    
    # Load Teacher
    teacher_path = Path("11_MODELS/M2C_RLAF/checkpoint_seed43.pt")
    teacher = RLAFPolicyNetwork(zero_init=False)
    ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    teacher.to(device)
    teacher.eval()
    print("[M3 Distillation] Teacher Model (seed43) loaded.", flush=True)
    
    # Load Dataset
    data_path = Path("05_DATA/M2B_MULTISCALE/m2b_feedback_dataset.pt")
    dataset = torch.load(data_path, map_location="cpu", weights_only=False)
    print(f"[M3 Distillation] Loaded {len(dataset)} training graphs.", flush=True)
    
    # Precompute Teacher Targets
    print("[M3 Distillation] Precomputing teacher targets across training graphs...", flush=True)
    t_start = time.time()
    cached_targets = []
    
    # Use 1500 diverse graphs for lightning training
    subset_size = min(1500, len(dataset))
    for i in range(subset_size):
        item = dataset[i]
        v = item["vertex_data"].to(device)
        e = item["edge_data"].to(device)
        c = item["connectivity"].to(device)
        u = torch.zeros((1, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            t_out = teacher(v, c, e, u)
            cached_targets.append({
                "v": v,
                "e": e,
                "c": c,
                "t_mu": t_out["mu"].detach(),
                "t_p_target": torch.sigmoid(t_out["rho"].detach()),
            })
    print(f"[M3 Distillation] Precomputed {len(cached_targets)} targets in {time.time() - t_start:.2f}s", flush=True)
    
    configs = [
        {"name": "compact_d16_T1", "d": 16, "core_steps": 1, "hidden": 16},
        {"name": "compact_d16_T2", "d": 16, "core_steps": 2, "hidden": 16},
        {"name": "compact_d32_T1", "d": 32, "core_steps": 1, "hidden": 32},
    ]
    
    results = {}
    save_dir = Path("11_MODELS/M3_COMPACT")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for cfg in configs:
        name = cfg["name"]
        d = cfg["d"]
        T = cfg["core_steps"]
        h = cfg["hidden"]
        
        print(f"\n=======================================================", flush=True)
        print(f"Training Student: {name} (d={d}, T={T}, hidden={h})", flush=True)
        student = create_compact_network(d=d, core_steps=T, hidden_size=h).to(device)
        param_count = sum(p.numel() for p in student.parameters())
        print(f"Parameter Count: {param_count:,} params", flush=True)
        
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
        
        epochs = 10
        batch_size = 32
        n_samples = len(cached_targets)
        
        t0 = time.time()
        for epoch in range(epochs):
            student.train()
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            total_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for idx in batch_indices:
                    item = cached_targets[idx]
                    v = item["v"]
                    e = item["e"]
                    c = item["c"]
                    u = torch.zeros((1, 1), dtype=torch.float32, device=device)
                    
                    s_out = student(v, c, e, u)
                    loss_mu = F.mse_loss(s_out["mu"], item["t_mu"])
                    loss_rho = F.binary_cross_entropy_with_logits(s_out["rho"], item["t_p_target"])
                    loss = loss_mu + 2.0 * loss_rho
                    batch_loss = batch_loss + loss
                    
                batch_loss = batch_loss / len(batch_indices)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.5f}", flush=True)
                
        print(f"Distillation completed in {time.time() - t0:.1f}s", flush=True)
        
        ckpt_path = save_dir / f"{name}.pt"
        torch.save({
            "config": cfg,
            "state_dict": student.state_dict(),
            "param_count": param_count,
        }, ckpt_path)
        print(f"Saved: {ckpt_path}", flush=True)
        
        print("Evaluating on 100 held-out SATLIB uf50-218 test instances...", flush=True)
        metrics = evaluate_on_satlib(student, device, num_instances=100)
        print(f"Results for {name}:", flush=True)
        print(f"  Mean Decisions:   {metrics['mean_decisions']:.2f} (M0: 39.08, Teacher: 26.20)", flush=True)
        print(f"  Median Decisions: {metrics['median_decisions']:.2f}", flush=True)
        print(f"  Mean Conflicts:   {metrics['mean_conflicts']:.2f}", flush=True)
        
        results[name] = {
            "config": cfg,
            "params": param_count,
            "metrics": metrics,
            "ckpt_path": str(ckpt_path),
        }
        
    with open("11_MODELS/M3_COMPACT/distillation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[M3 Distillation] All candidate models evaluated successfully!", flush=True)

if __name__ == "__main__":
    main()
