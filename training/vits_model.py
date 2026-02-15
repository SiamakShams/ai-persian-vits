#!/usr/bin/env python3
"""
Production VITS model for Persian TTS
Based on: https://arxiv.org/abs/2106.06103
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextEncoder(nn.Module):
    """
    Text encoder using Transformer architecture
    """
    def __init__(self, n_vocab, hidden_channels=256, filter_channels=1024, 
                 n_heads=4, n_layers=6, kernel_size=3, p_dropout=0.1):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        # Embedding layer
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        
        # Positional encoding
        self.pos_emb = PositionalEncoding(hidden_channels, p_dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=filter_channels,
            dropout=p_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Pre-attention projection
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        
    def forward(self, x, x_lengths):
        """
        Args:
            x: [batch_size, seq_len] - input text sequence
            x_lengths: [batch_size] - length of each sequence
        Returns:
            x: [batch_size, hidden_channels, seq_len] - encoded text
            x_mask: [batch_size, 1, seq_len] - attention mask
        """
        # Embed text
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, C]
        
        # Add positional encoding
        x = self.pos_emb(x)
        
        # Create attention mask
        x_mask = self.sequence_mask(x_lengths, x.size(1)).unsqueeze(1)  # [B, 1, T]
        
        # Transform
        x = self.encoder(x)  # [B, T, C]
        
        # Transpose for conv layers
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.proj(x)
        
        return x, x_mask
    
    @staticmethod
    def sequence_mask(length, max_length=None):
        """Generate sequence mask"""
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PosteriorEncoder(nn.Module):
    """
    Posterior encoder for audio (mel spectrogram)
    """
    def __init__(self, in_channels=80, out_channels=192, hidden_channels=192, 
                 kernel_size=5, dilation_rate=1, n_layers=16):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        
        # Input projection
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # WaveNet-style residual blocks
        self.enc = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** (i % 3)
            self.enc.append(
                ResidualBlock(hidden_channels, kernel_size, dilation=dilation)
            )
        
        # Output projection to mean and log variance
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        
    def forward(self, x, x_mask):
        """
        Args:
            x: [batch_size, in_channels, time] - input mel spectrogram
            x_mask: [batch_size, 1, time] - attention mask
        Returns:
            z: [batch_size, out_channels, time] - latent representation (sampled)
            m: [batch_size, out_channels, time] - mean
            logs: [batch_size, out_channels, time] - log variance
        """
        x = self.pre(x) * x_mask
        
        for layer in self.enc:
            x = layer(x, x_mask)
        
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        # Sample from distribution using reparameterization trick
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        
        return z, m, logs


class ResidualBlock(nn.Module):
    """Residual block with dilated convolution"""
    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=(kernel_size * dilation - dilation) // 2,
            dilation=dilation
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, x_mask):
        residual = x
        x = self.conv(x * x_mask)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        return (x + residual) * x_mask


class FlowDecoder(nn.Module):
    """
    Flow-based decoder for converting latent to mel spectrogram
    """
    def __init__(self, channels=192, hidden_channels=192, kernel_size=5, 
                 n_layers=4, n_flows=4):
        super().__init__()
        
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        
        # Flow layers
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                CouplingLayer(channels, hidden_channels, kernel_size, n_layers)
            )
    
    def forward(self, x, x_mask, reverse=False):
        """
        Args:
            x: [batch_size, channels, time] - input
            x_mask: [batch_size, 1, time] - attention mask
            reverse: bool - forward or inverse flow
        Returns:
            x: [batch_size, channels, time] - output
            logdet: scalar - log determinant of Jacobian
        """
        logdet_tot = 0
        
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, x_mask, reverse=False)
                logdet_tot += logdet
        else:
            for flow in reversed(self.flows):
                x, logdet = flow(x, x_mask, reverse=True)
                logdet_tot += logdet
        
        return x, logdet_tot


class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow"""
    def __init__(self, channels, hidden_channels, kernel_size, n_layers):
        super().__init__()
        
        self.channels = channels
        self.half_channels = channels // 2
        
        # WaveNet for transformation
        self.net = WaveNet(self.half_channels, hidden_channels, 
                          self.half_channels * 2, kernel_size, n_layers)
        
    def forward(self, x, x_mask, reverse=False):
        """Affine coupling transformation"""
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)
        
        h = self.net(x0, x_mask)
        m, logs = torch.split(h, [self.half_channels, self.half_channels], dim=1)
        logs = torch.clamp(logs, -10, 2)
        
        if not reverse:
            x1 = m + x1 * torch.exp(logs)
            logdet = torch.sum(logs * x_mask, dim=[1, 2])
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            logdet = -torch.sum(logs * x_mask, dim=[1, 2])
        
        x = torch.cat([x0, x1], dim=1) * x_mask
        return x, logdet


class WaveNet(nn.Module):
    """WaveNet-style network"""
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 kernel_size, n_layers):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(
                ResidualBlock(hidden_channels, kernel_size, dilation=dilation)
            )
        
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()
        
    def forward(self, x, x_mask):
        x = self.pre(x) * x_mask
        for layer in self.layers:
            x = layer(x, x_mask)
        x = self.proj(x) * x_mask
        return x


class DurationPredictor(nn.Module):
    """Predict duration for each phoneme"""
    def __init__(self, in_channels=192, filter_channels=256, kernel_size=3, 
                 p_dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, 
                              padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(filter_channels)
        self.conv2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, 
                              padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self, x, x_mask):
        """
        Args:
            x: [batch_size, in_channels, seq_len] - input from text encoder
            x_mask: [batch_size, 1, seq_len] - attention mask
        Returns:
            x: [batch_size, seq_len] - predicted duration (in log scale)
        """
        x = self.conv1(x * x_mask)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x * x_mask)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.proj(x * x_mask)
        return x.squeeze(1)


class VITS(nn.Module):
    """
    Complete VITS model for end-to-end TTS
    """
    def __init__(self, n_vocab=100, spec_channels=80, hidden_channels=192,
                 filter_channels=768, n_heads=2, n_layers=6, kernel_size=3,
                 p_dropout=0.1):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        
        # Text encoder
        self.text_encoder = TextEncoder(
            n_vocab, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout
        )
        
        # Posterior encoder (for training)
        self.posterior_encoder = PosteriorEncoder(
            spec_channels, hidden_channels, hidden_channels,
            kernel_size=5, n_layers=16
        )
        
        # Flow decoder
        self.flow = FlowDecoder(
            hidden_channels, hidden_channels, kernel_size=5,
            n_layers=4, n_flows=4
        )
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(
            hidden_channels, 256, kernel_size, p_dropout
        )
        
        # Projection from text encoder to prior
        self.proj_mean = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.proj_std = nn.Conv1d(hidden_channels, hidden_channels, 1)
        
    def forward(self, x, x_lengths, y, y_lengths):
        """
        Training forward pass
        
        Args:
            x: [batch_size, text_len] - input text
            x_lengths: [batch_size] - text lengths
            y: [batch_size, spec_channels, mel_len] - target mel spectrogram
            y_lengths: [batch_size] - mel lengths
            
        Returns:
            Dictionary with:
                - y_pred: predicted mel spectrogram
                - dur_pred: predicted durations
                - loss_dict: dictionary of losses
        """
        # Encode text
        x, x_mask = self.text_encoder(x, x_lengths)
        
        # Create mel mask
        y_mask = self.sequence_mask(y_lengths, y.size(2)).unsqueeze(1).to(x_mask.dtype)
        
        # Encode mel (posterior)
        z, m_q, logs_q = self.posterior_encoder(y, y_mask)
        logs_q = torch.clamp(logs_q, min=-10, max=2)
        
        # Prior from text encoder
        m_p = self.proj_mean(x)
        logs_p = torch.clamp(self.proj_std(x), min=-10, max=2)
        
        # Duration prediction
        logw = self.duration_predictor(x, x_mask)
        
        # Expand x to match y length (using ground truth durations during training)
        # For simplicity, we use attention-based alignment
        attn = self.maximum_path(z, x, y_mask, x_mask)
        # attn: [B, 1, x_len, z_len], m_p: [B, channels, x_len]
        # We want: [B, channels, z_len]
        m_p_expanded = torch.matmul(m_p, attn.squeeze(1))  # [B, channels, x_len] @ [B, x_len, z_len] -> [B, channels, z_len]
        logs_p_expanded = torch.matmul(logs_p, attn.squeeze(1))
        
        # Flow
        z_p, logdet = self.flow(z, y_mask, reverse=False)
        
        # Losses
        # 1. Reconstruction loss (KL divergence between posterior and prior)
        loss_kl = self.kl_loss(m_q, logs_q, m_p_expanded, logs_p_expanded, y_mask)
        
        # 2. Duration loss
        loss_dur = torch.sum((logw - torch.log(torch.sum(attn, dim=-1) + 1e-8)) ** 2) / torch.sum(x_mask)
        
        # For inference
        z_p_inv, _ = self.flow(z_p, y_mask, reverse=True)
        
        return {
            'z_p': z_p,
            'z': z,
            'z_p_inv': z_p_inv,
            'logdet': logdet,
            'loss_kl': loss_kl,
            'loss_dur': loss_dur,
            'm_p': m_p,
            'logs_p': logs_p,
            'attn': attn
        }
    
    def infer(self, x, x_lengths, length_scale=1.0):
        """
        Inference (synthesis)
        
        Args:
            x: [batch_size, text_len] - input text
            x_lengths: [batch_size] - text lengths
            length_scale: duration scale factor
            
        Returns:
            y: [batch_size, spec_channels, mel_len] - generated mel spectrogram
        """
        # Encode text
        x, x_mask = self.text_encoder(x, x_lengths)
        
        # Prior from text encoder
        m_p = self.proj_mean(x)
        logs_p = torch.clamp(self.proj_std(x), min=-10, max=2)
        
        # Predict durations
        logw = self.duration_predictor(x, x_mask)
        w = torch.exp(logw) * x_mask.squeeze(1) * length_scale
        w_ceil = torch.ceil(w)
        
        # Expand using predicted durations
        y_lengths = torch.clamp_min(torch.sum(w_ceil, dim=1), 1).long()
        y_max_length = int(y_lengths.max())
        
        # Expand features
        m_p_expanded = self.regulate_length(m_p, w_ceil, y_max_length)
        logs_p_expanded = self.regulate_length(logs_p, w_ceil, y_max_length)
        
        y_mask = self.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x.dtype)
        
        # Sample from prior
        z_p = m_p_expanded + torch.randn_like(m_p_expanded) * torch.exp(logs_p_expanded)
        z_p = z_p * y_mask
        
        # Decode through flow
        y, _ = self.flow(z_p, y_mask, reverse=True)
        
        return y, y_mask
    
    @staticmethod
    def kl_loss(m_q, logs_q, m_p, logs_p, mask):
        """Compute KL divergence loss with numerical stability"""
        # Clamp log values for numerical stability
        logs_q = torch.clamp(logs_q, min=-10, max=10)
        logs_p = torch.clamp(logs_p, min=-10, max=10)
        
        # KL divergence: KL(q||p) = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2)/(2*sigma_p^2) - 0.5
        kl = logs_p - logs_q + (torch.exp(2 * logs_q) + (m_q - m_p) ** 2) / (2 * torch.exp(2 * logs_p) + 1e-8) - 0.5
        kl = torch.sum(kl * mask)
        return kl / (torch.sum(mask) + 1e-8)
    
    @staticmethod
    def sequence_mask(length, max_length=None):
        """Generate sequence mask"""
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)
    
    @staticmethod
    def maximum_path(z, x, z_mask, x_mask):
        """Compute attention alignment using maximum path algorithm"""
        # Simplified: return uniform attention
        # Full implementation would use Monotonic Alignment Search
        batch_size, _, z_len = z.shape
        _, _, x_len = x.shape
        
        attn = torch.zeros(batch_size, x_len, z_len, device=z.device)
        
        for b in range(batch_size):
            x_len_b = int(x_mask[b].sum())
            z_len_b = int(z_mask[b].sum())
            
            # Simple uniform alignment
            step = z_len_b / x_len_b
            for i in range(x_len_b):
                start = int(i * step)
                end = int((i + 1) * step)
                attn[b, i, start:end] = 1.0 / max(1, end - start)
        
        return attn.unsqueeze(1)
    
    @staticmethod
    def regulate_length(x, durations, max_len):
        """Expand features according to durations"""
        batch_size, channels, _ = x.shape
        output = torch.zeros(batch_size, channels, max_len, device=x.device)
        
        for b in range(batch_size):
            pos = 0
            for i, dur in enumerate(durations[b]):
                dur = int(dur.item())
                if dur > 0 and pos < max_len:
                    end_pos = min(pos + dur, max_len)
                    output[b, :, pos:end_pos] = x[b, :, i].unsqueeze(1)
                    pos = end_pos
                if pos >= max_len:
                    break
        
        return output
