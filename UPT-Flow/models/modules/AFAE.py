import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet2D(nn.Module):
    """Differentiable Haar Wavelet Transform - splits features into
    LL (illumination) and LH/HL/HH (edges/texture) sub-bands."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1,  1], [ 1,  1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[1,  1], [-1, -1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[1, -1], [ 1, -1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1,  1]], dtype=torch.float32) * 0.5
        self.register_buffer('f_ll', ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('f_lh', lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('f_hl', hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('f_hh', hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape
        f_ll = self.f_ll.expand(C, 1, 2, 2)
        f_lh = self.f_lh.expand(C, 1, 2, 2)
        f_hl = self.f_hl.expand(C, 1, 2, 2)
        f_hh = self.f_hh.expand(C, 1, 2, 2)
        ll = F.conv2d(x, f_ll, stride=2, groups=C)
        lh = F.conv2d(x, f_lh, stride=2, groups=C)
        hl = F.conv2d(x, f_hl, stride=2, groups=C)
        hh = F.conv2d(x, f_hh, stride=2, groups=C)
        return ll, lh, hl, hh


class HaarWaveletInverse2D(nn.Module):
    """Inverse Haar Wavelet via transposed convolution."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1,  1], [ 1,  1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[1,  1], [-1, -1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[1, -1], [ 1, -1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1,  1]], dtype=torch.float32) * 0.5
        self.register_buffer('f_ll', ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('f_lh', lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('f_hl', hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('f_hh', hh.unsqueeze(0).unsqueeze(0))

    def forward(self, ll, lh, hl, hh):
        C = ll.shape[1]
        f_ll = self.f_ll.expand(C, 1, 2, 2)
        f_lh = self.f_lh.expand(C, 1, 2, 2)
        f_hl = self.f_hl.expand(C, 1, 2, 2)
        f_hh = self.f_hh.expand(C, 1, 2, 2)
        out  = F.conv_transpose2d(ll, f_ll, stride=2, groups=C)
        out += F.conv_transpose2d(lh, f_lh, stride=2, groups=C)
        out += F.conv_transpose2d(hl, f_hl, stride=2, groups=C)
        out += F.conv_transpose2d(hh, f_hh, stride=2, groups=C)
        return out


class IlluminationAttention(nn.Module):
    """Channel attention for LL sub-band — targets global brightness/color."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU()
        )

    def forward(self, ll):
        w   = self.fc(self.gap(ll))
        out = ll * w
        out = out + self.refine(out)
        return out


class EdgeDetailAttention(nn.Module):
    """Spatial + channel attention for HF sub-bands — sharpens edges."""

    def __init__(self, channels):
        super().__init__()
        hf_ch = channels * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(hf_ch, channels, 1, bias=False),
            nn.GELU()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1,
                      groups=max(channels // 2, 1), bias=False),
            nn.Conv2d(channels // 2, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm   = nn.GroupNorm(min(8, channels), channels)

    def forward(self, lh, hl, hh):
        hf  = self.fuse(torch.cat([lh, hl, hh], dim=1))
        sp  = self.spatial(hf)
        ch  = self.channel(hf)
        out = hf * sp * ch
        out = self.norm(self.refine(out))
        return out


class AdaptiveFrequencyGate(nn.Module):
    """Learns per-pixel weights to fuse illumination and edge branches.
    Uses the unbalanced point map so distorted pixels rely more on
    illumination correction."""

    def __init__(self, channels):
        super().__init__()
        in_ch = channels * 2 + 1
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, ll_feat, hf_feat, unbalanced_map=None):
        if unbalanced_map is None:
            unbalanced_map = torch.zeros(
                ll_feat.shape[0], 1,
                ll_feat.shape[2], ll_feat.shape[3],
                device=ll_feat.device
            )
        if unbalanced_map.shape[-2:] != ll_feat.shape[-2:]:
            unbalanced_map = F.interpolate(
                unbalanced_map.float(),
                size=ll_feat.shape[-2:],
                mode='nearest'
            )
        combined = torch.cat([ll_feat, hf_feat, unbalanced_map], dim=1)
        weights  = self.gate(combined)
        w_ll = weights[:, 0:1]
        w_hf = weights[:, 1:2]
        return w_ll * ll_feat + w_hf * hf_feat


class AFAE(nn.Module):
    """
    Adaptive Frequency-Aware Enhancement Module.
    Plugs into ONE feature map from MSFormer.
    Wavelet decompose → frequency-specific attention → adaptive fuse → reconstruct.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.wavelet     = HaarWavelet2D()
        self.inv_wavelet = HaarWaveletInverse2D()
        self.ill_attn    = IlluminationAttention(channels)
        self.edge_attn   = EdgeDetailAttention(channels)
        self.ll_proj     = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate        = AdaptiveFrequencyGate(channels)
        self.proj        = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False)
        )
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, unbalanced_map=None):
        B, C, H, W = x.shape
        identity = x

        # --- Wavelet decomposition ---
        ll, lh, hl, hh = self.wavelet(x)

        # --- Frequency-specific attention ---
        ll_enh = self.ill_attn(ll)
        hf_enh = self.edge_attn(lh, hl, hh)

        # --- Adaptive gate ---
        um_down = None
        if unbalanced_map is not None:
            um_down = F.interpolate(
                unbalanced_map.float(),
                size=(H // 2, W // 2),
                mode='nearest'
            )
        fused = self.gate(ll_enh, hf_enh, um_down)

        # --- Inverse wavelet to reconstruct ---
        ll_out = self.ll_proj(fused)
        out    = self.inv_wavelet(ll_out, lh, hl, hh)

        # --- Residual + final projection ---
        out = self.norm(self.proj(out))
        out = out + identity
        return out


class FrequencyAwareLoss(nn.Module):
    """
    Auxiliary loss supervising both frequency bands.
    Use alongside NLL loss: total = nll + lambda_freq * freq_loss
    Recommended lambda_freq = 0.02
    """

    def __init__(self, lambda_ll: float = 1.0, lambda_hf: float = 0.5):
        super().__init__()
        self.lambda_ll = lambda_ll
        self.lambda_hf = lambda_hf
        self.wavelet   = HaarWavelet2D()

    def forward(self, enhanced: torch.Tensor, ground_truth: torch.Tensor):
        ll_e, lh_e, hl_e, hh_e = self.wavelet(enhanced)
        ll_g, lh_g, hl_g, hh_g = self.wavelet(ground_truth)

        loss_ll = F.l1_loss(ll_e, ll_g)

        eps = 1e-6
        loss_hf = (
            torch.mean(torch.sqrt((lh_e - lh_g) ** 2 + eps)) +
            torch.mean(torch.sqrt((hl_e - hl_g) ** 2 + eps)) +
            torch.mean(torch.sqrt((hh_e - hh_g) ** 2 + eps))
        ) / 3.0

        return self.lambda_ll * loss_ll + self.lambda_hf * loss_hf