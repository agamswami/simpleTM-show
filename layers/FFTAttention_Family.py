import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from layers.SWTAttention_Family import GeomAttention


class FFTEmbedding(nn.Module):
    """
    FFT-based tokenization to replace SWT (Stationary Wavelet Transform).
    
    Decomposes the input signal into spectral components using FFT,
    keeping the top-m frequency bands. Produces a multi-scale representation
    analogous to SWT's (m+1) level decomposition.
    
    Forward (decompose=True):
        Input:  (B, N, L)  — batch, channels, sequence length
        Output: (B, N, m+1, L) — multi-scale spectral tokens
        
    Inverse (decompose=False):
        Input:  (B, N, m+1, L)
        Output: (B, N, L)
    """
    def __init__(self, d_channel=16, decompose=True, m=2, seq_len=None):
        super().__init__()
        self.decompose = decompose
        self.d_channel = d_channel
        self.m = m  # Number of frequency band levels (analogous to SWT levels)
        # Learnable weights for frequency band importance
        self.band_weights = nn.Parameter(torch.ones(m + 1), requires_grad=True)

    def forward(self, x):
        if self.decompose:
            return self.fft_decomposition(x)
        else:
            return self.fft_reconstruction(x)

    def fft_decomposition(self, x):
        """
        Decompose signal into m+1 frequency bands using FFT.
        
        The frequency spectrum is split into m+1 bands:
          - Band 0: lowest frequencies (approximation, like SWT approx coefficients)
          - Band 1..m: progressively higher frequencies (like SWT detail coefficients)
        
        Args:
            x: (B, N, L) input signal
        Returns:
            coeffs: (B, N, m+1, L) multi-scale frequency band coefficients
        """
        B, N, L = x.shape
        
        # Compute FFT
        X_fft = torch.fft.rfft(x, dim=-1)  # (B, N, L//2+1) complex
        n_freqs = X_fft.shape[-1]
        
        # Split frequency spectrum into m+1 bands
        band_size = max(1, n_freqs // (self.m + 1))
        coeffs = []
        
        for i in range(self.m + 1):
            # Create a mask for this frequency band
            mask = torch.zeros(n_freqs, device=x.device, dtype=torch.float32)
            start = i * band_size
            end = min((i + 1) * band_size, n_freqs) if i < self.m else n_freqs
            mask[start:end] = 1.0
            
            # Apply mask and inverse FFT to get time-domain representation
            X_band = X_fft * mask.unsqueeze(0).unsqueeze(0)  # (B, N, n_freqs)
            band_signal = torch.fft.irfft(X_band, n=L, dim=-1)  # (B, N, L)
            
            # Apply learnable weight for this band
            band_signal = band_signal * self.band_weights[i]
            coeffs.append(band_signal)
        
        # Stack: (B, N, m+1, L) — same shape convention as SWT output
        # Reverse so lowest freq (approx) is first, matching SWT convention
        return torch.stack(coeffs, dim=-2)

    def fft_reconstruction(self, coeffs):
        """
        Reconstruct signal from frequency band coefficients.
        
        Simply sums all frequency bands to reconstruct the original signal,
        analogous to SWT inverse reconstruction.
        
        Args:
            coeffs: (B, N, m+1, L) multi-scale coefficients
        Returns:
            reconstructed: (B, N, L)
        """
        # Sum all frequency bands (inverse of band decomposition)
        # Apply inverse band weights for reconstruction
        inv_weights = 1.0 / (self.band_weights + 1e-8)
        weighted_coeffs = coeffs * inv_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        reconstructed = weighted_coeffs.sum(dim=-2)  # (B, N, L)
        return reconstructed


class FFTGeomAttentionLayer(nn.Module):
    """
    Attention layer using FFT-based tokenization with Geometric Attention.
    
    This is a drop-in replacement for GeomAttentionLayer that uses FFT
    spectral decomposition instead of SWT wavelet decomposition, while
    keeping the geometric (wedge product) attention mechanism unchanged.
    
    Pipeline:
        1. FFT decomposition: (B, N, L') → (B, N, m+1, L')
        2. Q/K/V projections on spectral tokens
        3. Geometric attention (wedge product + dot product scoring)
        4. FFT reconstruction: (B, N, m+1, L') → (B, N, L')
    """
    def __init__(self, attention, d_model,
                 m=2, d_channel=None, geomattn_dropout=0.5):
        super(FFTGeomAttentionLayer, self).__init__()

        self.d_channel = d_channel
        self.inner_attention = attention
        
        # FFT decomposition (forward)
        self.fft_decompose = FFTEmbedding(
            d_channel=self.d_channel, decompose=True, m=m
        )
        
        # Q, K, V projections
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        
        # Output projection + FFT reconstruction (inverse)
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            FFTEmbedding(d_channel=self.d_channel, decompose=False, m=m),
        )
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # FFT decomposition: (B, N, d_model) → (B, N, m+1, d_model)
        queries = self.fft_decompose(queries)
        keys = self.fft_decompose(keys)
        values = self.fft_decompose(values)

        # Project Q, K, V and permute for attention
        # After permute: (B, d_model, m+1, N)
        queries = self.query_projection(queries).permute(0, 3, 2, 1)
        keys = self.key_projection(keys).permute(0, 3, 2, 1)
        values = self.value_projection(values).permute(0, 3, 2, 1)

        # Geometric attention (unchanged — wedge product scoring)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        # Inverse permute + output projection + FFT reconstruction
        out = self.out_projection(out.permute(0, 3, 2, 1))

        return out, attn
