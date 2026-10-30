"""Graph-Q-SAT Deep Q-Network implementation in pure PyTorch.

Faithfully implements the Encode-Process-Decode Graph Network architecture
from Kurin et al. (NeurIPS 2020) and Doc 03 / Doc 05 specifications.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from models.graph_q_sat.meta_layer import (
    ModifiedMetaLayer,
    scatter_add_native,
    scatter_mean_native,
)
from graph.construction.graph_builder import VAR_ID_IDX


def get_mlp(
    in_size: int,
    out_size: int,
    n_hidden: int = 1,
    hidden_size: int = 64,
    activation=nn.ReLU,
    activate_last: bool = True,
    layer_norm: bool = True,
) -> nn.Sequential:
    """Build multi-layer perceptron with optional LayerNorm and final activation."""
    layers = []
    l_in = in_size
    for _ in range(n_hidden):
        layers.append(nn.Linear(l_in, hidden_size))
        layers.append(activation())
        l_in = hidden_size

    layers.append(nn.Linear(l_in, out_size))

    if activate_last:
        layers.append(activation())
        if layer_norm:
            layers.append(nn.LayerNorm(out_size))

    return nn.Sequential(*layers)


class GraphNet(nn.Module):
    """MetaLayer Graph Network module supporting independent or coupled updates."""

    def __init__(
        self,
        in_dims: Tuple[int, int, int],
        out_dims: Tuple[int, int, int],
        independent: bool = False,
        e2v_agg: str = "sum",
        n_hidden: int = 1,
        hidden_size: int = 64,
        activation=nn.ReLU,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.e2v_agg = e2v_agg
        self.independent = independent
        v_in, e_in, u_in = in_dims
        v_out, e_out, u_out = out_dims

        if independent:
            self.edge_mlp = get_mlp(e_in, e_out, n_hidden, hidden_size, activation, layer_norm=layer_norm)
            self.node_mlp = get_mlp(v_in, v_out, n_hidden, hidden_size, activation, layer_norm=layer_norm)
            self.global_mlp = get_mlp(u_in, u_out, n_hidden, hidden_size, activation, layer_norm=layer_norm)

            def edge_model(src, dest, edge_attr, u=None, e_indices=None):
                return self.edge_mlp(edge_attr)

            def node_model(x, edge_index, edge_attr, u=None, v_indices=None):
                return self.node_mlp(x)

            def global_model(x, edge_attr, u, v_indices=None, e_indices=None):
                return self.global_mlp(u)

        else:
            self.edge_mlp = get_mlp(e_in + 2 * v_in + u_in, e_out, n_hidden, hidden_size, activation, layer_norm=layer_norm)
            self.node_mlp = get_mlp(v_in + e_out + u_in, v_out, n_hidden, hidden_size, activation, layer_norm=layer_norm)
            self.global_mlp = get_mlp(u_in + v_out + e_out, u_out, n_hidden, hidden_size, activation, layer_norm=layer_norm)

            def edge_model(src, dest, edge_attr, u=None, e_indices=None):
                u_expanded = u[e_indices] if (u is not None and e_indices is not None and e_indices.numel() > 0) else torch.zeros((edge_attr.size(0), u_in), device=edge_attr.device)
                out = torch.cat([src, dest, edge_attr, u_expanded], dim=1)
                return self.edge_mlp(out)

            def node_model(x, edge_index, edge_attr, u=None, v_indices=None):
                row, col = edge_index
                if self.e2v_agg == "sum":
                    out = scatter_add_native(edge_attr, row, dim_size=x.size(0))
                elif self.e2v_agg == "mean":
                    out = scatter_mean_native(edge_attr, row, dim_size=x.size(0))
                else:
                    raise ValueError(f"Unknown e2v_agg: {self.e2v_agg}")
                u_expanded = u[v_indices] if (u is not None and v_indices is not None and v_indices.numel() > 0) else torch.zeros((x.size(0), u_in), device=x.device)
                out = torch.cat([x, out, u_expanded], dim=1)
                return self.node_mlp(out)

            def global_model(x, edge_attr, u, v_indices=None, e_indices=None):
                dim_size = u.size(0)
                v_agg = scatter_mean_native(x, v_indices, dim_size=dim_size) if (v_indices is not None and v_indices.numel() > 0) else torch.zeros((dim_size, x.size(1)), device=u.device)
                e_agg = scatter_mean_native(edge_attr, e_indices, dim_size=dim_size) if (e_indices is not None and e_indices.numel() > 0) else torch.zeros((dim_size, edge_attr.size(1)), device=u.device)
                out = torch.cat([u, v_agg, e_agg], dim=1)
                return self.global_mlp(out)

        self.op = ModifiedMetaLayer(edge_model, node_model, global_model)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        v_indices: Optional[torch.Tensor] = None,
        e_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.op(x, edge_index, edge_attr, u, v_indices, e_indices)


class GraphQSatModel(nn.Module):
    """Full Encode-Process-Decode Graph Network for learned SAT variable/polarity selection."""

    def __init__(
        self,
        in_dims: Tuple[int, int, int] = (2, 2, 1),
        encoder_out_dims: Tuple[int, int, int] = (32, 32, 32),
        core_out_dims: Tuple[int, int, int] = (64, 64, 32),
        decoder_out_dims: Tuple[int, int, int] = (32, 1, 1),
        out_dims: Tuple[int, Optional[int], Optional[int]] = (2, None, None),
        core_steps: int = 4,
        e2v_agg: str = "sum",
        n_hidden: int = 1,
        hidden_size: int = 64,
        activation=nn.ReLU,
        independent_block_layers: int = 0,
    ):
        super().__init__()
        self.core_steps = core_steps
        self.core_out_dims = core_out_dims
        self.in_dims = in_dims

        # Encoder: independent GraphNet mapping input features to initial latent space
        self.encoder = GraphNet(
            in_dims=in_dims,
            out_dims=encoder_out_dims,
            independent=True,
            n_hidden=independent_block_layers,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=True,
        )

        # Core: recurrent GraphNet running message-passing for core_steps
        core_in_dims = (
            encoder_out_dims[0] + core_out_dims[0],
            encoder_out_dims[1] + core_out_dims[1],
            encoder_out_dims[2] + core_out_dims[2],
        )
        self.core = GraphNet(
            in_dims=core_in_dims,
            out_dims=core_out_dims,
            independent=False,
            e2v_agg=e2v_agg,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=True,
        )

        # Decoder: independent GraphNet mapping core representations
        self.decoder = GraphNet(
            in_dims=core_out_dims,
            out_dims=decoder_out_dims,
            independent=True,
            n_hidden=independent_block_layers,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=True,
        )

        # Final linear vertex transform: 32 -> 2 outputs (Q(True), Q(False))
        self.vertex_out_transform = nn.Linear(decoder_out_dims[0], out_dims[0])

    def get_init_latent(self, n_v: int, n_e: int, n_u: int, device: torch.device):
        return (
            torch.zeros((n_v, self.core_out_dims[0]), device=device),
            torch.zeros((n_e, self.core_out_dims[1]), device=device),
            torch.zeros((n_u, self.core_out_dims[2]), device=device),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        v_indices: Optional[torch.Tensor] = None,
        e_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if v_indices is None and e_indices is None:
            v_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            e_indices = torch.zeros(edge_attr.shape[0], dtype=torch.long, device=edge_attr.device)

        # Encode
        enc_x, enc_e, enc_u = self.encoder(x, edge_index, edge_attr, u, v_indices, e_indices)

        # Process (recurrent core)
        latent0 = (enc_x, enc_e, enc_u)
        latent = self.get_init_latent(x.shape[0], edge_attr.shape[0], u.shape[0], x.device)

        for _ in range(self.core_steps):
            latent = self.core(
                torch.cat([latent0[0], latent[0]], dim=1),
                edge_index,
                torch.cat([latent0[1], latent[1]], dim=1),
                torch.cat([latent0[2], latent[2]], dim=1),
                v_indices,
                e_indices,
            )

        # Decode
        dec_x, dec_e, dec_u = self.decoder(latent[0], edge_index, latent[1], latent[2], v_indices, e_indices)

        # Final Q-head on vertex representations
        v_out = self.vertex_out_transform(dec_x)
        return v_out, dec_e, dec_u

    def predict_action(self, graph_state) -> Tuple[int, float]:
        """Inference helper: predict greedy action index for a single GraphState."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(graph_state.vertex_data, dtype=torch.float32, device=device)
        edge_attr = torch.as_tensor(graph_state.edge_data, dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(graph_state.connectivity, dtype=torch.int64, device=device)
        u = torch.as_tensor(graph_state.global_attr, dtype=torch.float32, device=device)

        with torch.no_grad():
            v_out, _, _ = self(x, edge_index, edge_attr, u)
            # Filter to variable nodes only: [N_v, 2]
            var_qs = v_out[x[:, VAR_ID_IDX] == 1.0]  # shape [N_v, 2]
            # Flatten to 2 * N_v action values
            flat_qs = var_qs.flatten()
            best_action_idx = int(flat_qs.argmax().item())
            best_q_val = float(flat_qs[best_action_idx].item())
            return best_action_idx, best_q_val
