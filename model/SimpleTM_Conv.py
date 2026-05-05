"""
SimpleTM_Conv: Convolutional tokenization variant.

This replaces the wavelet/FFT tokenization stage with a multi-scale depthwise
convolutional tokenizer while keeping the same geometric attention encoder and
forecast head.
"""

#added comletely
import torch
import torch.nn as nn

from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.ConvAttention_Family import ConvGeomAttentionLayer
from layers.ParallelAttention_Family import ParallelConvGeomAttentionLayer
from layers.SWTAttention_Family import GeomAttention
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.attention_mode = getattr(configs, 'attention_mode', 'original')
        self.conv_kernel_sizes = getattr(configs, 'conv_kernel_sizes', None)

        attention_layer_cls = ConvGeomAttentionLayer
        if self.attention_mode == 'dual':
            attention_layer_cls = ParallelConvGeomAttentionLayer

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention_layer_cls(
                        GeomAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            alpha=self.alpha,
                            learnable_alpha=getattr(configs, 'learnable_alpha', False),
                        ),
                        configs.d_model,
                        m=configs.m,
                        d_channel=configs.dec_in,
                        geomattn_dropout=self.geomattn_dropout,
                        conv_kernel_sizes=self.conv_kernel_sizes,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.projector = nn.Linear(configs.d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, n_variates = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :n_variates]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, None, None, None)
        return dec_out, attns
