
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from segmentation_models_pytorch.losses import DiceLoss
from encoder import VSSEncoder

class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(DecoderBlock, self).__init__()

        self.identity = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        )

 
        self.decodeD = nn.Sequential(
            Upsample(2, mode="bilinear"),
            ResBlock_CBAM(input_channels, output_channels//4),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        residual = self.identity(x)
        out = self.decodeD(x)
        out = F.interpolate(out, size=residual.shape[2:], mode='bilinear', align_corners=False)
        out += residual

        return out


class FCM(nn.Module):
    """Three-way cross-modal interaction: Discard / Complement / Enhance.

    A learned softmax gate jointly partitions each spatial position into
    three mutually-exclusive roles:
      g_enh  — both modalities agree → enhance with each other's filtered features
      g_comp — one modality has exclusive info → cross-attention transfer
      g_disc — neither is reliable (noise/cloud/speckle) → suppress before transfer

    Gate network uses a 3×3 DWConv for spatial context (cloud/speckle are
    regional phenomena, not single-pixel events).

    All four γ parameters are decoupled: OPT and SAR have different noise
    profiles and complementarity needs, so they should scale independently.
    """

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # ── Three-way gate (1×1 → BN → DWConv 3×3 → GELU → 1×1 → softmax) ──
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, 3, 1, bias=False),
        )

        # ── Complement: bidirectional cross-attention with SR ────────────
        self.q_opt   = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_sar   = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_sar  = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.kv_opt  = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.pool    = nn.AvgPool2d(sr_ratio, sr_ratio, ceil_mode=True) \
                       if sr_ratio > 1 else nn.Identity()
        self.out_opt = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_sar = nn.Conv2d(dim, dim, 1, bias=False)

        # ── Decoupled γ (non-zero init so gate_net gets gradients from step 1) ──
        self.gamma_comp_opt = nn.Parameter(torch.full((1,), 0.1))
        self.gamma_comp_sar = nn.Parameter(torch.full((1,), 0.1))
        self.gamma_enh_opt  = nn.Parameter(torch.full((1,), 0.1))
        self.gamma_enh_sar  = nn.Parameter(torch.full((1,), 0.1))

    def _attn(self, q, k, v):
        B, C, H, W = q.shape
        nH, d = self.num_heads, self.head_dim
        L, S  = H * W, k.shape[2] * k.shape[3]
        q_ = q.flatten(2).view(B, nH, d, L).permute(0, 1, 3, 2)
        k_ = k.flatten(2).view(B, nH, d, S).permute(0, 1, 3, 2)
        v_ = v.flatten(2).view(B, nH, d, S).permute(0, 1, 3, 2)
        attn = F.softmax((q_ @ k_.transpose(-2, -1)) * self.scale, dim=-1)
        return (attn @ v_).permute(0, 1, 3, 2).contiguous().reshape(B, C, H, W)

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor):
        gates  = F.softmax(
            self.gate_net(torch.cat([x_opt, x_sar], dim=1)), dim=1
        )
        g_enh  = gates[:, 0:1]
        g_comp = gates[:, 1:2]

        keep    = g_enh + g_comp
        x_opt_f = x_opt * keep
        x_sar_f = x_sar * keep

        sar_r = self.pool(x_sar_f)
        opt_r = self.pool(x_opt_f)
        k_sar, v_sar = self.kv_sar(sar_r).chunk(2, dim=1)
        k_opt, v_opt = self.kv_opt(opt_r).chunk(2, dim=1)

        ctx_opt = self.out_opt(self._attn(self.q_opt(x_opt), k_sar, v_sar)) * g_comp
        ctx_sar = self.out_sar(self._attn(self.q_sar(x_sar), k_opt, v_opt)) * g_comp

        enh_opt = x_sar_f * g_enh
        enh_sar = x_opt_f * g_enh

        out_opt = x_opt + self.gamma_comp_opt * ctx_opt + self.gamma_enh_opt * enh_opt
        out_sar = x_sar + self.gamma_comp_sar * ctx_sar + self.gamma_enh_sar * enh_sar
        return out_opt, out_sar


class RFI(nn.Module):
    """Cross-modal fusion: spatial gate + global channel weighting + diff residual.

    Three complementary paths:
      base : pixel-wise interpolation via spatial gate (local decision)
      dyn  : channel-wise reweighting via GAP+FC      (global decision)
      diff : cross-modal disagreement residual         (conflict correction)
    """

    def __init__(self, dim: int):
        super().__init__()
        # Spatial gate (DW conv for local context)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

        # SE-style channel recalibration of base
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 4, 16), dim),
            nn.Sigmoid(),
        )

        # Cross-modal difference branch
        self.diff_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        # Output refinement
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        self.gamma_dyn  = nn.Parameter(torch.full((1,), 0.1))
        self.gamma_diff = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x_opt, x_sar], dim=1)

        # Spatial interpolation
        g    = self.gate(cat)
        base = g * x_opt + (1.0 - g) * x_sar

        # SE channel recalibration on base
        se  = self.se_fc(self.gap(base).flatten(1))[:, :, None, None]
        dyn = base * se

        # Cross-modal disagreement
        diff = self.diff_proj(x_opt - x_sar)

        out = self.out_proj(base + self.gamma_dyn * dyn + self.gamma_diff * diff)
        return out

# Define a simple learnable scale parameter (replacement for mmcv's Scale)
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

# ConvModule replacement
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, norm_cfg=True, act_cfg=True):
        super(ConvModule, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)]
        if norm_cfg:
            layers.append(nn.BatchNorm2d(out_channels))
        if act_cfg:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class CEB(nn.Module):
    """Channel Enhanced Block (from SFMFusion, TIP 2025)"""
    def __init__(self, num_feat):
        super(CEB, self).__init__()
        self.num_feat = num_feat
        self.conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        half = num_feat // 2
        self.mlp_avg = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, kernel_size=1)
        )
        self.mlp_max = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, kernel_size=1)
        )
        self.sigmoid_avg = nn.Sigmoid()
        self.sigmoid_max = nn.Sigmoid()
        idx = torch.arange(num_feat).reshape(2, num_feat // 2).t().reshape(-1)
        self.register_buffer('shuffle_idx', idx)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        C = self.num_feat
        x1, x2 = torch.split(x, C // 2, dim=1)
        y1 = self.sigmoid_avg(self.mlp_avg(self.avg_pool(x1)))
        y2 = self.sigmoid_max(self.mlp_max(self.max_pool(x2)))
        z = torch.cat((x1 * y1, x2 * y2), dim=1)
        z = z[:, self.shuffle_idx, :, :]
        z = z + skip
        return z


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D feature maps."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class FreMLP(nn.Module):
    def __init__(self, nc, expand=1):
        super(FreMLP, self).__init__()
        self.process_mag = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        mag = self.process_mag(mag)
        pha = self.process_pha(pha)

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        
        return x_out


class SRB(nn.Module):

    def __init__(self, in_channels):
        super(SRB, self).__init__()

        self.process_ll = FreMLP(in_channels)

        self.process_high = FreMLP(in_channels * 3)
        
        self.refine = nn.Conv2d(in_channels, in_channels, 1)
 
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def dwt_2d(self, x):
        # Haar DWT
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        
        x_ll = (x00 + x01 + x10 + x11) / 4.0
        x_lh = (x00 + x01 - x10 - x11) / 2.0
        x_hl = (x00 - x01 + x10 - x11) / 2.0
        x_hh = (x00 - x01 - x10 + x11) / 2.0
        
        return x_ll, x_lh, x_hl, x_hh

    def idwt_2d(self, ll, lh, hl, hh):
        # Haar IDWT
        x00 = ll + lh + hl + hh
        x01 = ll + lh - hl - hh
        x10 = ll - lh + hl - hh
        x11 = ll - lh - hl + hh
        
        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out

    def forward(self, x):
        ll, lh, hl, hh = self.dwt_2d(x)
        
        ll_enhanced = self.process_ll(ll) + ll

        highs = torch.cat([lh, hl, hh], dim=1)      # (B, 3C, H/2, W/2)
        highs_enhanced = self.process_high(highs) + highs
        
        lh_new, hl_new, hh_new = torch.chunk(highs_enhanced, 3, dim=1)
        
        out = self.idwt_2d(ll_enhanced, lh_new, hl_new, hh_new)
        
        if out.shape != x.shape:
             out = out[:, :, :x.shape[2], :x.shape[3]]
             
        return self.gamma * self.refine(out) + x

SFE = SRB


class ResBlock_CBAM(nn.Module):
    """Ours: CEB (channel enhancement) only, SFE applied only at final decoder output"""
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        self.ceb = CEB(places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.ceb(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out



from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


# AlignDecoderBlock
class AlignDecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AlignDecoderBlock, self).__init__()
        self.up = ConvModule(input_channels, output_channels, kernel_size=1)
        self.identity_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.decode = nn.Sequential(
            # nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(output_channels)
        )
        
        self.decodeD =ResBlock_CBAM(input_channels, output_channels//4)
        self.up1 = CAB(input_channels)
        
    def forward(self, low_feat, high_feat):
        f = self.up1(low_feat, high_feat)

        out = self.decodeD(f)

        return out



class CISRMambaBackbone(nn.Module):
    """
    Dual-branch encoder-decoder for CISRMamba.

    Encoder output: 4 scales at strides {4, 8, 16, 32}
      enc_dims = [96, 192, 384, 768]

    Resolution flow (input H x W):
      f1 @ H/4  -> f2 @ H/8  -> f3 @ H/16 -> f4 @ H/32
      decode4(f3, f4) -> H/16
      decode3(f2, ·)  -> H/8
      decode2(f1, ·)  -> H/4
      decode1(·)      -> H/2   (DecoderBlock x2 upsample)
      final bilinear  -> H     (x2, restores full resolution)
    """

    ENC_DIMS = [96, 192, 384, 768]   # MobileMamba dims

    def __init__(self, side_dim=64):
        super(CISRMambaBackbone, self).__init__()

        enc_dims = self.ENC_DIMS

        # ── Backbones ──────────────────────────────────────────────────
        self.backbone_opt = VSSEncoder(in_channels=4)
        self.backbone_sar = VSSEncoder(in_channels=1)

        # ── Side projections (4 scales) ────────────────────────────────
        self.side1_rgb = ConvModule(enc_dims[0], side_dim, kernel_size=3)
        self.side2_rgb = ConvModule(enc_dims[1], side_dim, kernel_size=3)
        self.side3_rgb = ConvModule(enc_dims[2], side_dim, kernel_size=3)
        self.side4_rgb = ConvModule(enc_dims[3], side_dim, kernel_size=3)

        self.side1_sar = ConvModule(enc_dims[0], side_dim, kernel_size=3)
        self.side2_sar = ConvModule(enc_dims[1], side_dim, kernel_size=3)
        self.side3_sar = ConvModule(enc_dims[2], side_dim, kernel_size=3)
        self.side4_sar = ConvModule(enc_dims[3], side_dim, kernel_size=3)

        # ── FCM + RFI (4 scales) ───────────────────────────────────────
        self.fcm1 = FCM(side_dim)
        self.fcm2 = FCM(side_dim)
        self.fcm3 = FCM(side_dim)
        self.fcm4 = FCM(side_dim)

        self.rfi1 = RFI(side_dim)
        self.rfi2 = RFI(side_dim)
        self.rfi3 = RFI(side_dim)
        self.rfi4 = RFI(side_dim)

        # ── Decoder (4 stages) ─────────────────────────────────────────
        self.decode1 = DecoderBlock(side_dim, side_dim)        # x2 upsample
        self.decode2 = AlignDecoderBlock(side_dim, side_dim)
        self.decode3 = AlignDecoderBlock(side_dim, side_dim)
        self.decode4 = AlignDecoderBlock(side_dim, side_dim)

        # SFE at final decoder output
        self.sfe_final = SFE(side_dim)

    def forward(self, x):
        # Split 5-channel input into optical (4ch) and SAR (1ch)
        x_img, x_aux = torch.split(x, (4, 1), dim=1)

        # ── Encode ────────────────────────────────────────────────────
        # Each backbone returns [f1, f2, f3, f4] at strides [4,8,16,32]
        x1, x2, x3, x4 = self.backbone_opt(x_img)
        y1, y2, y3, y4 = self.backbone_sar(x_aux)

        # ── Side projection ───────────────────────────────────────────
        x1s = self.side1_rgb(x1); x2s = self.side2_rgb(x2)
        x3s = self.side3_rgb(x3); x4s = self.side4_rgb(x4)

        y1s = self.side1_sar(y1); y2s = self.side2_sar(y2)
        y3s = self.side3_sar(y3); y4s = self.side4_sar(y4)

        # ── FCM + RFI fusion ──────────────────────────────────────────
        x4s, y4s = self.fcm4(x4s, y4s); f4 = self.rfi4(x4s, y4s)
        x3s, y3s = self.fcm3(x3s, y3s); f3 = self.rfi3(x3s, y3s)
        x2s, y2s = self.fcm2(x2s, y2s); f2 = self.rfi2(x2s, y2s)
        x1s, y1s = self.fcm1(x1s, y1s); f1 = self.rfi1(x1s, y1s)

        # ── Decode ────────────────────────────────────────────────────
        # f4 @ H/32, f3 @ H/16, f2 @ H/8, f1 @ H/4
        out = self.decode4(f3, f4)   # -> H/16
        out = self.decode3(f2, out)  # -> H/8
        out = self.decode2(f1, out)  # -> H/4
        out = self.decode1(out)      # -> H/2  (DecoderBlock x2)

        # SFE + final upsample to restore full resolution H x W
        out = self.sfe_final(out)
        out = F.interpolate(out, scale_factor=2,
                            mode='bilinear', align_corners=False)  # -> H

        return out


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
        


class CISRMamba(nn.Module):
    def __init__(self,num_classes=21843):
        super(CISRMamba, self).__init__()
        self.num_classes = num_classes
        self.network = CISRMambaBackbone()
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=3,
        )

    def forward(self, images, masks=None):  
        
        images = self.network(images)  # (B, n_patch, hidden)
        logits = self.segmentation_head(images) 
        if masks != None:  
            loss1 = DiceLoss(mode='binary')(logits, masks)  
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)  
            
            return logits, loss1, loss2   
        return logits


class CAB(nn.Module):
    def __init__(self, features, norm_cfg=dict(type='BN', requires_grad=True)):
        super(CAB, self).__init__()

        # Replacing build_norm_layer with nn.BatchNorm2d
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage



