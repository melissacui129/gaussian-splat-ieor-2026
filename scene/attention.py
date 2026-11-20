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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim) where B is total queried coordinates (e.g., R * M)
        returns: (N, B, out_dim)
        """
        h = torch.einsum("bi,nih->nbh", x, self.w1) + self.b1 # (B, in_dim) x (N, in_dim, hidden_dim) -> (N, B, hidden_dim)
        h = torch.tanh(h)
        out = torch.einsum("nbh,nho->nbo", h, self.w2) + self.b2 # (N, B, hidden_dim) x (N, hidden_dim, out_dim) -> (N, B, out_dim)
        return out


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
    ):
        super().__init__()
        assert mode in ("exact", "performer")
        self.num_primitives = num_primitives
        self.d_qk = d_qk
        self.num_features = num_features
        self.mode = mode

        self.q_net = BatchedMLP(num_primitives, in_dim=2, hidden_dim=hidden_dim, out_dim=d_qk) # q_i(x, y): R^2 -> R^{d_QK}
        self.k_net = BatchedMLP(num_primitives, in_dim=1, hidden_dim=hidden_dim, out_dim=d_qk) # k_i(z): R -> R^{d_QK}

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

    def _favor_plus_features(self, v: torch.Tensor) -> torch.Tensor:
        """
        Feature map:
        phi(v) = exp(-||v||^2 / 2) / sqrt(r) * exp(Omega @ v)
        v: (N, B, d_qk) -> returns (N, B, r)
        """
        norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True)  # (N, B, 1)
        proj = torch.einsum("rd,nbd->nbr", self.omega, v)  # (N, B, r)
        proj = torch.clamp(proj, max=20.0) # clamp to prevent overflow
        return torch.exp(proj - 0.5 * norm_sq) / math.sqrt(self.num_features)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (R, M, 3) world coordinates (x, y, z_m) along R rays and M depth slices
        returns p_z: (N, R, M) normalized depth attention weights summing to 1 over M
        """
        R, M, _ = pts.shape
        B = R * M
        flat_pts = pts.reshape(B, 3)

        xy = flat_pts[:, :2]  # (B, 2)
        z = flat_pts[:, 2:3]  # (B, 1)

        # Evaluate q_i(x, y) and k_i(z) for all N primitives: shape (N, R, M, d_qk)
        q = self.q_net(xy).reshape(self.num_primitives, R, M, self.d_qk)
        k = self.k_net(z).reshape(self.num_primitives, R, M, self.d_qk)

        if self.mode == "exact":
            # Dot product q(x,y)^T k(z_m) at each sample, softmax over M
            logits = torch.sum(q * k, dim=-1)  # (N, R, M)
            p_z = torch.softmax(logits, dim=-1)
        else:
            # Performer positive random features
            phi_q = self._favor_plus_features(q.reshape(self.num_primitives, B, self.d_qk))
            phi_k = self._favor_plus_features(k.reshape(self.num_primitives, B, self.d_qk))

            phi_q = phi_q.reshape(self.num_primitives, R, M, self.num_features)
            phi_k = phi_k.reshape(self.num_primitives, R, M, self.num_features)

            # Numerator: phi(q)^T phi(k_m) -> (N, R, M)
            numer = torch.sum(phi_q * phi_k, dim=-1)

            # Denominator: phi(q)^T * sum_j phi(k_j) -> (N, R, M)
            s_i = torch.sum(phi_k, dim=2, keepdim=True)  # (N, R, 1, r)
            denom = torch.sum(phi_q * s_i, dim=-1) + 1e-8  # (N, R, M)
            p_z = numer / denom

        return p_z