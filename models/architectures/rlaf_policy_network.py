"""RLAF Policy Network for Milestone M2C.

Faithfully implements the one-shot neural policy architecture from
Tönshoff & Grohe, 'Reinforcement Learning from Algorithm Feedback for SAT Solving' (ICLR 2026).

Key Specifications:
1. Message passing over bipartite variable-clause graph.
2. Dual heads:
   - Weight head: outputs mu(x) for LogNormal(mu(x), sigma^2) variable importance weights.
   - Polarity head: outputs rho(x) logit for Bernoulli(sigma(rho(x))) initial branch polarities.
3. Zero-Initialization:
   - Final linear projection layers are initialized to zero.
   - Guarantees mu=0 (weight=1.0) and rho=0 (p=0.5), exactly recovering standard MiniSat at initialization.
4. Exact log-prob, entropy, and KL-divergence against reference policy.
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch.distributions import LogNormal, Bernoulli

from models.graph_q_sat.graph_network import GraphNet
from graph.construction.graph_builder import VAR_ID_IDX


class RLAFPolicyNetwork(nn.Module):
    """One-Shot Bipartite GNN Policy for RLAF."""

    def __init__(
        self,
        in_dims: Tuple[int, int, int] = (2, 2, 1),
        encoder_out_dims: Tuple[int, int, int] = (32, 32, 32),
        core_out_dims: Tuple[int, int, int] = (64, 64, 32),
        decoder_out_dims: Tuple[int, int, int] = (32, 1, 1),
        core_steps: int = 4,
        e2v_agg: str = "sum",
        hidden_size: int = 64,
        sigma: float = 0.5,
        activation=nn.ReLU,
        zero_init: bool = True,
    ):
        super().__init__()
        self.core_steps = core_steps
        self.core_out_dims = core_out_dims
        self.in_dims = in_dims
        self.sigma = sigma

        # 1. Independent Feature Encoder
        self.encoder = GraphNet(
            in_dims=in_dims,
            out_dims=encoder_out_dims,
            independent=True,
            n_hidden=0,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=True,
        )

        # 2. Recurrent Message Passing Core
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
            n_hidden=1,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=True,
        )

        # 3. Latent Decoder
        self.decoder = GraphNet(
            in_dims=core_out_dims,
            out_dims=decoder_out_dims,
            independent=True,
            n_hidden=0,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=True,
        )

        # 4. Weight Head: outputs mu(x) in R for LogNormal(mu, sigma^2)
        v_latent_dim = decoder_out_dims[0]
        self.weight_head = nn.Sequential(
            nn.Linear(v_latent_dim, hidden_size),
            activation(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

        # 5. Polarity Head: outputs logit rho(x) in R for Bernoulli(sigmoid(rho))
        self.polarity_head = nn.Sequential(
            nn.Linear(v_latent_dim, hidden_size),
            activation(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

        # Zero-initialization: ensure initial policy outputs mu=0 -> w=1.0, rho=0 -> p=0.5
        if zero_init:
            nn.init.zeros_(self.weight_head[-1].weight)
            nn.init.zeros_(self.weight_head[-1].bias)
            nn.init.zeros_(self.polarity_head[-1].weight)
            nn.init.zeros_(self.polarity_head[-1].bias)

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
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through GNN producing distribution parameters (mu, rho)."""
        if v_indices is None and e_indices is None:
            v_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            e_indices = torch.zeros(edge_attr.shape[0], dtype=torch.long, device=edge_attr.device)

        # 1. Encode
        enc_x, enc_e, enc_u = self.encoder(x, edge_index, edge_attr, u, v_indices, e_indices)

        # 2. Process
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

        # 3. Decode
        dec_x, _, _ = self.decoder(latent[0], edge_index, latent[1], latent[2], v_indices, e_indices)

        # 4. Extract variable nodes
        var_mask = (x[:, VAR_ID_IDX] == 1.0)
        var_embeddings = dec_x[var_mask]

        # 5. Output distribution parameters
        mu = self.weight_head(var_embeddings).squeeze(-1)
        rho = self.polarity_head(var_embeddings).squeeze(-1)

        return {
            "mu": mu,
            "rho": rho,
            "var_mask": var_mask,
        }

    def sample_action(
        self,
        mu: torch.Tensor,
        rho: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions (weights, polarities) and return their log-probabilities.
        
        Args:
            mu: [N_vars] location parameter for LogNormal
            rho: [N_vars] logit parameter for Bernoulli
            deterministic: if True, returns mode (exp(mu), I(rho > 0))
            
        Returns:
            weights: [N_vars] float32 > 0
            polarities: [N_vars] int64 in {0, 1}
            log_prob: scalar float32 total action log-probability
        """
        device = mu.device
        sigma_tensor = torch.tensor(self.sigma, device=device, dtype=mu.dtype)

        dist_w = LogNormal(loc=mu, scale=sigma_tensor)
        dist_p = Bernoulli(logits=rho)

        if deterministic:
            # Deterministic inference mode (RLAF paper Section 4.3)
            weights = torch.exp(mu)
            polarities = (rho > 0.0).long()
            log_prob = torch.tensor(0.0, device=device)
        else:
            # Stochastic training mode
            weights = dist_w.sample()
            polarities = dist_p.sample().long()
            # Joint log probability of full action vector
            log_p_w = dist_w.log_prob(weights).sum()
            log_p_p = dist_p.log_prob(polarities.float()).sum()
            log_prob = log_p_w + log_p_p

        return weights, polarities, log_prob

    def evaluate_log_prob(
        self,
        mu: torch.Tensor,
        rho: torch.Tensor,
        weights: torch.Tensor,
        polarities: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate log-probability of given actions under current distribution."""
        device = mu.device
        sigma_tensor = torch.tensor(self.sigma, device=device, dtype=mu.dtype)

        dist_w = LogNormal(loc=mu, scale=sigma_tensor)
        dist_p = Bernoulli(logits=rho)

        # Clamp weights slightly above zero to prevent log(0)
        safe_weights = torch.clamp(weights, min=1e-7)
        log_p_w = dist_w.log_prob(safe_weights).sum()
        log_p_p = dist_p.log_prob(polarities.float()).sum()

        return log_p_w + log_p_p

    def compute_entropy(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Compute distribution entropy for regularization bonus."""
        device = mu.device
        sigma_tensor = torch.tensor(self.sigma, device=device, dtype=mu.dtype)

        dist_w = LogNormal(loc=mu, scale=sigma_tensor)
        dist_p = Bernoulli(logits=rho)

        return dist_w.entropy().sum() + dist_p.entropy().sum()

    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        rho: torch.Tensor,
        ref_mu: Optional[torch.Tensor] = None,
        ref_rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence KL(pi_theta || pi_ref).
        
        If ref_mu and ref_rho are None, defaults to zero-prior (mu=0, rho=0).
        """
        device = mu.device
        if ref_mu is None:
            ref_mu = torch.zeros_like(mu)
        if ref_rho is None:
            ref_rho = torch.zeros_like(rho)

        # KL(LogNormal(mu, sigma) || LogNormal(ref_mu, sigma)) = (mu - ref_mu)^2 / (2 * sigma^2)
        kl_w = ((mu - ref_mu) ** 2) / (2.0 * (self.sigma ** 2))

        # KL for Bernoulli distributions
        p = torch.sigmoid(rho).clamp(1e-6, 1.0 - 1e-6)
        ref_p = torch.sigmoid(ref_rho).clamp(1e-6, 1.0 - 1e-6)
        kl_p = p * (torch.log(p) - torch.log(ref_p)) + (1.0 - p) * (
            torch.log(1.0 - p) - torch.log(1.0 - ref_p)
        )

        return kl_w.sum() + kl_p.sum()
