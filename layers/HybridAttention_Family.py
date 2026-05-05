import torch
import torch.nn as nn

from layers.SWTAttention_Family import WaveletEmbedding
from layers.FFTAttention_Family import FFTEmbedding
from layers.ConvAttention_Family import ConvEmbedding, ScaleMixerReconstruction


class BranchFusionGate(nn.Module):
    """
    Softmax-normalized fusion over SWT, FFT, and Conv tokenizers.

    The gate is computed per sample, variate, and scale, which matches the
    project note's goal of adaptive branch weights instead of one global weight.
    """

    def __init__(
        self,
        d_model,
        num_branches=3,
        gate_mode='soft',
        branch_dropout=0.0,
        gate_temperature=0.5,
        residual_branch='none',
    ):
        super().__init__()
        if gate_mode not in {'soft', 'sparse', 'one_hot'}:
            raise ValueError("gate_mode must be one of: soft, sparse, one_hot")
        if residual_branch not in {'none', 'swt', 'fft', 'conv'}:
            raise ValueError("residual_branch must be one of: none, swt, fft, conv")

        self.num_branches = num_branches
        self.gate_mode = gate_mode
        self.branch_dropout = float(branch_dropout)
        self.gate_temperature = max(float(gate_temperature), 1e-4)
        self.residual_branch = residual_branch
        self.branch_to_index = {'swt': 0, 'fft': 1, 'conv': 2}
        self.gate = nn.Sequential(
            nn.Linear(num_branches * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_branches),
        )
        if self.residual_branch != 'none':
            self.residual_logit = nn.Parameter(torch.tensor(-2.0))

    def _make_weights(self, logits):
        if self.gate_mode == 'soft':
            return torch.softmax(logits, dim=-1)

        soft_weights = torch.softmax(logits / self.gate_temperature, dim=-1)
        if self.gate_mode == 'sparse':
            return soft_weights

        hard_indices = soft_weights.argmax(dim=-1)
        hard_weights = torch.zeros_like(soft_weights).scatter_(
            -1, hard_indices.unsqueeze(-1), 1.0
        )
        if self.training:
            return hard_weights - soft_weights.detach() + soft_weights
        return hard_weights

    def _apply_branch_dropout(self, weights):
        if not self.training or self.branch_dropout <= 0:
            return weights

        batch_size = weights.shape[0]
        keep = torch.rand(
            batch_size,
            self.num_branches,
            device=weights.device,
        ) >= self.branch_dropout

        empty_rows = keep.sum(dim=-1) == 0
        if empty_rows.any():
            fallback_indices = weights.detach().mean(dim=(1, 2)).argmax(dim=-1)
            keep[empty_rows, fallback_indices[empty_rows]] = True

        keep = keep[:, None, None, :].to(dtype=weights.dtype)
        dropped_weights = weights * keep
        denominator = dropped_weights.sum(dim=-1, keepdim=True)
        fallback_weights = keep / keep.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.where(
            denominator > 1e-8,
            dropped_weights / denominator.clamp_min(1e-8),
            fallback_weights,
        )

    def forward(self, branches):
        fused_input = torch.cat(branches, dim=-1)
        logits = self.gate(fused_input)
        weights = self._make_weights(logits)
        weights = self._apply_branch_dropout(weights)
        stacked = torch.stack(branches, dim=-2)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)

        if self.residual_branch != 'none':
            base = branches[self.branch_to_index[self.residual_branch]]
            residual_strength = torch.sigmoid(self.residual_logit)
            fused = base + residual_strength * (fused - base)

        return fused, weights


class HybridGeomAttentionLayer(nn.Module):
    """
    Three-branch hybrid tokenization followed by the original geometric attention.

    Branches:
    - SWT for multi-scale, non-stationary structure
    - FFT for compact spectral structure
    - Conv for local motifs and sharp short-range changes
    """

    def __init__(
        self,
        attention,
        d_model,
        requires_grad=True,
        wv='db2',
        m=2,
        kernel_size=None,
        d_channel=None,
        geomattn_dropout=0.5,
        conv_kernel_sizes=None,
        hybrid_gate_mode='soft',
        hybrid_branch_dropout=0.0,
        hybrid_gate_temperature=0.5,
        hybrid_residual_branch='none',
    ):
        super().__init__()
        self.inner_attention = attention
        self.swt = WaveletEmbedding(
            d_channel=d_channel,
            swt=True,
            requires_grad=requires_grad,
            wv=wv,
            m=m,
            kernel_size=kernel_size,
        )
        self.fft = FFTEmbedding(
            d_channel=d_channel,
            decompose=True,
            m=m,
        )
        self.conv = ConvEmbedding(
            d_channel=d_channel,
            m=m,
            kernel_sizes=conv_kernel_sizes,
        )
        self.fusion_gate = BranchFusionGate(
            d_model=d_model,
            num_branches=3,
            gate_mode=hybrid_gate_mode,
            branch_dropout=hybrid_branch_dropout,
            gate_temperature=hybrid_gate_temperature,
            residual_branch=hybrid_residual_branch,
        )

        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            ScaleMixerReconstruction(d_model=d_model, m=m),
        )

    def _tokenize_and_fuse(self, x):
        swt_tokens = self.swt(x)
        fft_tokens = self.fft(x)
        conv_tokens = self.conv(x)
        fused, weights = self.fusion_gate([swt_tokens, fft_tokens, conv_tokens])
        return fused, weights

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        queries, _ = self._tokenize_and_fuse(queries)
        keys, _ = self._tokenize_and_fuse(keys)
        values, _ = self._tokenize_and_fuse(values)

        queries = self.query_projection(queries).permute(0, 3, 2, 1)
        keys = self.key_projection(keys).permute(0, 3, 2, 1)
        values = self.value_projection(values).permute(0, 3, 2, 1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = self.out_projection(out.permute(0, 3, 2, 1))
        return out, attn
