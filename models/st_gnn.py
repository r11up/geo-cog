"""Dense PyTorch GCN encoder for the environmental grid, with an optional temporal GRU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=use_bias)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        HW = self.W(H)

        if H.dim() == 2:
            if A_hat.dim() == 3:
                A_hat = A_hat[0]
            return F.relu(A_hat @ HW)

        B = H.shape[0]
        if A_hat.dim() == 2:
            A_batch = A_hat.unsqueeze(0).expand(B, -1, -1)
        else:
            A_batch = A_hat
        return F.relu(torch.bmm(A_batch, HW))


class SpatialGCN(nn.Module):
    def __init__(self, in_dim: int, intermediate: int = 64, out_dim: int = 128):
        super().__init__()
        self.layer1 = DenseGCNLayer(in_dim,       intermediate)
        self.layer2 = DenseGCNLayer(intermediate, out_dim)
        self.out_dim = out_dim

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H = self.layer1(H, A_hat)
        H = self.layer2(H, A_hat)
        return H


class TemporalGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim

    def forward(self, H_seq: torch.Tensor) -> torch.Tensor:
        B, T, N, D = H_seq.shape
        H_flat = H_seq.permute(0, 2, 1, 3).reshape(B * N, T, D)
        _, h_n = self.gru(H_flat)
        h_n = h_n[-1]
        H_out = h_n.reshape(B, N, self.hidden_dim)
        return H_out


class STGNN(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 6,
        gcn_intermediate: int = 64,
        gcn_hidden: int = 128,
        gru_hidden: int = 128,
        gcn_layers: int = 2,
        gru_layers: int = 1,
        embed_dim: int = 256,
        use_temporal: bool = False,
    ):
        super().__init__()

        self.gcn = SpatialGCN(
            in_dim=node_feat_dim,
            intermediate=gcn_intermediate,
            out_dim=gcn_hidden,
        )
        self.use_temporal = use_temporal
        if use_temporal:
            self.gru = TemporalGRU(
                input_dim=gcn_hidden,
                hidden_dim=gru_hidden,
                n_layers=gru_layers,
            )
        else:
            self.temporal_proj = nn.Linear(gcn_hidden, gru_hidden)
        self.pool_proj = nn.Linear(gru_hidden, embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self,
        F_graph: torch.Tensor,
        A_hat: torch.Tensor,
    ) -> torch.Tensor:
        if F_graph.dim() == 3:
            F_graph = F_graph.unsqueeze(1)

        B, T, N, D = F_graph.shape

        H_gcn_seq = []
        for t in range(T):
            H_t = self.gcn(F_graph[:, t, :, :], A_hat)
            H_gcn_seq.append(H_t)
        H_gcn_seq = torch.stack(H_gcn_seq, dim=1)

        if self.use_temporal:
            H_st = self.gru(H_gcn_seq)
        else:
            H_st = self.temporal_proj(H_gcn_seq)
            H_st = H_st.squeeze(1) if T == 1 else H_st.mean(dim=1)

        h_graph = H_st.mean(dim=1)

        v_env = self.pool_proj(h_graph)
        return v_env
