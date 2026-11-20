import torch
import torch.nn as nn
from scene.attention import Attention


class FactorizedGaussianModel(nn.Module):
    """
    Replaces 3D ellipsoids with N factorized primitives:
    p(x, y, z) = p_2D(x, y) * p_z(z | x, y)
    """
    def __init__(
        self,
        num_primitives: int,
        d_qk: int = 16,
        hidden_dim: int = 16,
        num_features: int = 32,
        attention_mode: str = "exact",
    ):
        super().__init__()
        self.num_primitives = num_primitives

        # 1. 2D Gaussian parameters: mean mu_i in R^2, [log-scales (s_x, s_y), angle theta] for covariance matrix Sigma
        self._xy = nn.Parameter(torch.empty(num_primitives, 2).uniform_(-1.5, 1.5)) # initial primitive spread
        self._scaling = nn.Parameter(torch.full((num_primitives, 2), -3.0))
        self._theta = nn.Parameter(torch.zeros(num_primitives, 1))

        # 2. Appearance parameters: opacity alpha_i and RGB color c_i
        self._opacity = nn.Parameter(torch.full((num_primitives, 1), 0.5))
        self._rgb = nn.Parameter(torch.zeros(num_primitives, 3))

        # 3. Per-primitive attention module p_z(z | x, y)
        self.depth_attention = Attention(
            num_primitives=num_primitives,
            d_qk=d_qk,
            hidden_dim=hidden_dim,
            num_features=num_features,
            mode=attention_mode,
        )

    @property
    def get_xy(self) -> torch.Tensor:
        return self._xy

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling) # ensures radius stays positive; Kerbl uses log logic in line 40 of gaussian_model.py

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_rgb(self) -> torch.Tensor:
        return torch.sigmoid(self._rgb)

    def evaluate_2d_gaussian(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Evaluate unnormalized 2D Gaussian
        p_2D^i(x, y) = exp(-0.5 * (u - mu_i)^T Sigma_i^{-1} (u - mu_i))
        xy: (B, 2) -> returns (N, B)
        """
        diff = xy.unsqueeze(0) - self._xy.unsqueeze(1) # (u - mu_i) ; (N, B, 2)

        # rotate into each primitive's local ellipse axes using R(theta)^T = R(-theta)
        cos_t = torch.cos(self._theta)  # (N, 1)
        sin_t = torch.sin(self._theta)  # (N, 1)

        dx = diff[..., 0]  # x-offsets from (N, B, 2); (N, B)
        dy = diff[..., 1]  # y-offsets from (N, B, 2); (N, B)

        # local_coord = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]] x [dx, dy]
        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy

        scales = self.get_scaling + 1e-6  # (N, 2)
        p2d_term = (local_x / scales[:, 0:1]) ** 2 + (local_y / scales[:, 1:2]) ** 2

        return torch.exp(-0.5 * p2d_term)  # (N, B)

    def forward(self, pts: torch.Tensor):
        """
        Evaluates total density and weighted RGB color at ray sample points.
        pts: (R, M, 3) world coordinates along R rays and M depth slices
        returns:
            sigma_total: (R, M, 1) total density at each (x, y, z_m)
            color_weighted: (R, M, 3) density-weighted RGB C(x, y, z_m)
        """
        R, M, _ = pts.shape
        xy_flat = pts[..., :2].reshape(R * M, 2)

        # 1. p_2D^i(x, y) -> shape (N, R, M)
        p_2d = self.evaluate_2d_gaussian(xy_flat).reshape(self.num_primitives, R, M)

        # 2. p_z^i(z_m | x, y) -> shape (N, R, M)
        p_z = self.depth_attention(pts)

        # 3. sigma_i(x, y, z_m) = alpha_i * p_2D^i * p_z^i -> (N, R, M)
        alpha = self.get_opacity.unsqueeze(-1)  # (N, 1, 1)
        sigma_i = alpha * p_2d * p_z

        # 4. Sum over all N primitives at each depth slice -> (R, M, 1)
        sigma_total = torch.sum(sigma_i, dim=0).unsqueeze(-1)

        # 5. Weighted average color C(x, y, z_m) -> (R, M, 3)
        rgb = self.get_rgb  # (N, 3)
        color_weighted = torch.einsum("nrm,nc->rmc", sigma_i, rgb) / (sigma_total + 1e-8)

        return sigma_total, color_weighted
    
    def get_stats(self):
        """Returns counts of total splats, non-transparent (active) splats, and total parameters."""
        active_splats = int((self.get_opacity > 0.01).sum().item())
        total_params = sum(p.numel() for p in self.parameters()) # e.g., (x, y), theta, w1, b1, etc. & number of elements inside trainable param tensors
        return {
            "total_splats": self._xy.shape[0], 
            "active_splats": active_splats, 
            "total_params": total_params
        }