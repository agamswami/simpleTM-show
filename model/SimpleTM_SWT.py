"""
SimpleTM_SWT: SWT-Only Tokenization Model

This is the original SimpleTM model documented as the SWT-only tokenization baseline.
It uses Stationary Wavelet Transform (SWT) for tokenization and Geometric Attention
(wedge product + dot product) for inter-variate attention.

Pipeline:
    1. Instance Normalization (RevIN)
    2. Inverted Embedding: (B, L, N) → (B, N, d_model) — channel-independent temporal tokens
    3. SWT Tokenization: Decomposes each token via SWT into multi-scale coefficients
    4. Geometric Attention: Wedge product scoring across variates at each scale
    5. SWT Reconstruction: Inverse SWT to reconstruct from multi-scale representation
    6. Output Projection: (B, N, d_model) → (B, H, N) — forecast horizon
    7. De-normalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.SWTAttention_Family import GeomAttentionLayer, GeomAttention
from layers.ParallelAttention_Family import ParallelSWTGeomAttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    SimpleTM with SWT-Only Tokenization (Original Baseline).
    
    This is functionally identical to the original SimpleTM model.
    Kept as a separate file for clean ablation study comparison.
    
    Args:
        configs: Configuration namespace containing:
            - seq_len (int): Input sequence length
            - pred_len (int): Prediction horizon length
            - output_attention (bool): Whether to output attention weights
            - use_norm (bool): Whether to apply instance normalization
            - geomattn_dropout (float): Dropout rate in attention projections
            - alpha (float): Weight balancing dot product vs wedge product in GeomAttention
            - kernel_size (int|None): Wavelet kernel size (None = use wavelet default)
            - d_model (int): Pseudo token dimensionality
            - d_ff (int): Feed-forward network dimensionality
            - e_layers (int): Number of encoder layers
            - enc_in (int): Number of input variates
            - dec_in (int): Number of channels for wavelet embedding
            - embed (str): Time embedding type
            - freq (str): Data frequency
            - dropout (float): General dropout
            - requires_grad (bool): Whether wavelet filters are learnable
            - wv (str): Wavelet family (e.g., 'db1', 'db2')
            - m (int): Number of SWT decomposition levels
            - factor (int): Attention factor
            - activation (str): Activation function ('relu' or 'gelu')
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.kernel_size = configs.kernel_size
        self.attention_mode = getattr(configs, 'attention_mode', 'original')

        attention_layer_cls = GeomAttentionLayer
        if self.attention_mode == 'dual':
            attention_layer_cls = ParallelSWTGeomAttentionLayer

        # Step 1: Inverted Embedding — project temporal dim to d_model
        enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, 
                                               configs.embed, configs.freq, configs.dropout)
        self.enc_embedding = enc_embedding

        # Step 2: Encoder with SWT tokenization + Geometric Attention
        encoder = Encoder(
            [  
                EncoderLayer(
                    attention_layer_cls(
                        GeomAttention(
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention, alpha=self.alpha,
                            learnable_alpha=getattr(configs, 'learnable_alpha', False)
                        ),
                        configs.d_model, 
                        requires_grad=configs.requires_grad, 
                        wv=configs.wv, 
                        m=configs.m, 
                        d_channel=configs.dec_in, 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder = encoder

        # Step 3: Output Projection — project d_model to prediction horizon
        projector = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.projector = projector


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for forecasting.
        
        Args:
            x_enc: (B, L, N) input time series
            x_mark_enc: Time features (unused, passed as None)
            x_dec: Decoder input (unused)
            x_mark_dec: Decoder time features (unused)
            
        Returns:
            dec_out: (B, H, N) forecasted values
            attns: List of attention weights from each encoder layer
        """
        # Instance Normalization (RevIN-style)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector

        # Inverted Embedding:       B L N -> B N d_model (pseudo temporal tokens)
        enc_out = enc_embedding(x_enc, x_mark_enc) 

        # Encoder (SWT + GeomAttn): B N d_model -> B N d_model
        enc_out, attns = encoder(enc_out, attn_mask=None)

        # Output Projection:        B N d_model -> B H N
        dec_out = projector(enc_out).permute(0, 2, 1)[:, :, :N] 

        # De-normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Main forward method.
        
        Args:
            x_enc: (B, L, N) input time series
            x_mark_enc: Time features
            x_dec: Decoder input
            x_mark_dec: Decoder time features
            mask: Optional mask
            
        Returns:
            dec_out: (B, H, N) forecasted values
            attns: Attention weights
        """
        dec_out, attns = self.forecast(x_enc, None, None, None)
        return dec_out, attns 
