import math
import torch
import torch.nn as nn

class BatchedMLP(nn.Module):
    """N independent lightweight MLPs evaluated in parallel"""
    def __init__(self, num_primitives: int, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        
        self.w1 = nn.Parameter(torch.randn(num_primitives, in_dim, hidden_dim) * 0.1) # weights; shape: (N, in_dim, hiden_dim)
        self.b1 = nn.Parameter(torch.zeros(num_primitives, 1, hidden_dim)) # biases
        self.w2 = nn.Parameter(torch.randn(num_primitives, hidden_dim, out_dim) * 0.1) # weights; shape: (N, hiden_dim, out_dim)
        self.b2 = nn.Parameter(torch.zeros(num_primitives, 1, out_dim)) # biases

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     x: (B, in_dim) where B is total queried coordinates (e.g., R * M)
    #     returns: (N, B, out_dim)
    #     """
    #     h = torch.einsum("bi,nih->nbh", x, self.w1) + self.b1 # (B, in_dim) x (N, in_dim, hidden_dim) -> (N, B, hidden_dim)
    #     h = torch.tanh(h)
    #     out = torch.einsum("nbh,nho->nbo", h, self.w2) + self.b2 # (N, B, hidden_dim) x (N, hidden_dim, out_dim) -> (N, B, out_dim)
    #     return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim) where B is total queried coordinates (e.g., R * M)
        returns: (N, B, out_dim)
        """
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h = torch.matmul(x.unsqueeze(0), self.w1) + self.b1
            h = torch.tanh(h)
            out = torch.matmul(h, self.w2) + self.b2
        return out.float()

class Attention(nn.Module):
    """
    Computes p_z(z_m | x, y) for N primitives across R rays and M depth slices.
    """
    def __init__(
        self,
        num_primitives: int,
        d_qk: int = 16,
        hidden_dim: int = 16,
        num_features: int = 32,
        mode: str = "exact",
        z_min: float = -1.5,
        z_max: float = 1.5,
        num_z_bins: int = 64,
    ):
        super().__init__()
        assert mode in ("exact", "performer")
        self.num_primitives = num_primitives
        self.d_qk = d_qk
        self.num_features = num_features
        self.mode = mode
        self.num_z_bins = num_z_bins

        self.q_net = BatchedMLP(num_primitives, in_dim=2, hidden_dim=hidden_dim, out_dim=d_qk) # q_i(x, y): R^2 -> R^{d_QK}
        self.k_net = BatchedMLP(num_primitives, in_dim=1, hidden_dim=hidden_dim, out_dim=d_qk) # k_i(z): R -> R^{d_QK}

        z_grid = torch.linspace(z_min, z_max, num_z_bins).unsqueeze(-1)  # (K, 1)
        self.register_buffer("z_grid", z_grid)
        
        self._cached_k_grid = None

        omega = self._create_projection_matrix(num_features, d_qk) # fixed projection matrix omega in R^{r x d_QK} for performer
        self.register_buffer("omega", omega)

    @staticmethod
    def _create_projection_matrix(r: int, d: int) -> torch.Tensor:
        """Draws orthogonal rows to create projection matrix (Choromanski et al., Section 2.4/Appendix B.2)."""
        blocks = []
        for _ in range(math.ceil(r / d)):
            q, _ = torch.linalg.qr(torch.randn(d, d)) # generates d x d orthogonal matrix
            norms = torch.linalg.norm(torch.randn(d, d), dim=1, keepdim=True)
            blocks.append(q * norms)
        return torch.cat(blocks, dim=0)[:r]  # (r, d)

    def _positive_random_features(self, v: torch.Tensor) -> torch.Tensor:
        """
        Feature map:
        phi(v) = exp(-||v||^2 / 2) / sqrt(r) * exp(Omega @ v)
        v: (N, B, d_qk) -> returns (N, B, r)
        """
        norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True)  # (N, B, 1)
        proj = torch.matmul(v, self.omega.T)  # (N, B, r)
        # proj = torch.einsum("rd,nbd->nbr", self.omega, v)  # (N, B, r)
        proj = torch.clamp(proj, max=20.0) # clamp to prevent overflow
        return torch.exp(proj - 0.5 * norm_sq) / math.sqrt(self.num_features)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (R, M, 3) world coordinates (x, y, z_m) along R rays and M depth slices
        returns p_z: (N, R, M) normalized depth attention weights
        """
        R, M, _ = pts.shape
        xy = pts[..., :2].reshape(R * M, 2)
        z = pts[..., 2:3].reshape(R * M, 1)

        # Query q_i(x_m, y_m) and sample key k_i(z_m): shape (N, R*M, d_qk)
        q = self.q_net(xy)
        k_sample = self.k_net(z)
        # k_grid = self.k_net(self.z_grid)  # (N, K, d_qk)
        if self.training:
            self._cached_k_grid = None
            k_grid = self.k_net(self.z_grid)
        else:
            if self._cached_k_grid is None:
                self._cached_k_grid = self.k_net(self.z_grid)
            k_grid = self._cached_k_grid

        if self.mode == "exact":
            num_logits = (q.float() * k_sample.float()).sum(dim=-1)          # (N, R*M)
            denom_logits = torch.bmm(q.float(), k_grid.float().transpose(1, 2))  # (N, R*M, K)

            max_logit = torch.maximum(
                num_logits.unsqueeze(-1),
                denom_logits.max(dim=-1, keepdim=True).values,
            )
            num = torch.exp(num_logits - max_logit.squeeze(-1))
            denom = torch.exp(denom_logits - max_logit).sum(dim=-1) + 1e-8
            p_z = (num / denom).reshape(self.num_primitives, R, M)

        elif self.mode == "performer":
            phi_q = self._positive_random_features(q.float())               # (N, R*M, F)
            phi_k_sample = self._positive_random_features(k_sample.float()) # (N, R*M, F)
            phi_k_grid = self._positive_random_features(k_grid.float())     # (N, K, F)

            num = (phi_q * phi_k_sample).sum(dim=-1)                   # (N, R*M)
            k_grid_sum = phi_k_grid.sum(dim=1)                         # (N, F)
            denom = (phi_q * k_grid_sum.unsqueeze(1)).sum(dim=-1) + 1e-8
            p_z = (num / denom).reshape(self.num_primitives, R, M)

        return p_z