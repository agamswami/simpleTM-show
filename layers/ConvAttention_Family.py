import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_conv_kernel_sizes(m, kernel_sizes=None):
    target = m + 1
    if kernel_sizes is None or str(kernel_sizes).lower() == 'none':
        sizes = [3, 5, 7, 11, 15, 21, 31]
    elif isinstance(kernel_sizes, str):
        sizes = [int(part.strip()) for part in kernel_sizes.split(',') if part.strip()]
    else:
        sizes = [int(size) for size in kernel_sizes]

    if not sizes:
        raise ValueError('conv_kernel_sizes must provide at least one positive kernel size')

    normalized = []
    for size in sizes:
        size = max(1, int(size))
        if size % 2 == 0:
            size += 1
        normalized.append(size)

    while len(normalized) < target:
        normalized.append(normalized[-1] + 2)

    return normalized[:target]


class DepthwiseCircularConv1d(nn.Module):
    def __init__(self, d_channel, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_channel,
            out_channels=d_channel,
            kernel_size=kernel_size,
            groups=d_channel,
            bias=False,
        )

    def forward(self, x):
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size // 2
        x = F.pad(x, (pad_left, pad_right), mode='circular')
        return self.conv(x)


class ScaleMixerReconstruction(nn.Module):
    def __init__(self, d_model, m):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear((m + 1) * d_model, d_model),
        )

    def forward(self, coeffs):
        return self.projection(coeffs)


class ConvEmbedding(nn.Module):
    """
    Multi-scale depthwise convolutional tokenization.

    Input:  (B, N, L)
    Output: (B, N, m+1, L)
    """

    def __init__(self, d_channel=16, m=2, kernel_sizes=None):
        super().__init__()
        self.kernel_sizes = resolve_conv_kernel_sizes(m, kernel_sizes)
        self.scale_convs = nn.ModuleList(
            [DepthwiseCircularConv1d(d_channel=d_channel, kernel_size=size) for size in self.kernel_sizes]
        )

    def forward(self, x):
        coeffs = [scale_conv(x) for scale_conv in self.scale_convs]
        return torch.stack(coeffs, dim=-2)


class ConvGeomAttentionLayer(nn.Module):
    """
    Geometric attention over convolutionally tokenized multi-scale features.
    """

    def __init__(
        self,
        attention,
        d_model,
        m=2,
        d_channel=None,
        geomattn_dropout=0.5,
        conv_kernel_sizes=None,
    ):
        super().__init__()
        self.inner_attention = attention
        self.conv_decompose = ConvEmbedding(
            d_channel=d_channel,
            m=m,
            kernel_sizes=conv_kernel_sizes,
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

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        queries = self.conv_decompose(queries)
        keys = self.conv_decompose(keys)
        values = self.conv_decompose(values)

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
