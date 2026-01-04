import torch
import torch.nn as nn
import math
from .utils import FlopsUtils, get_safe_groups

# ==========================================
# Optimized expert modules
# ==========================================

class OptimizedSimpleExpert(nn.Module):
    """Use GroupNorm instead of BatchNorm to improve stability for small batches."""
    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x): 
        return self.conv(x)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.conv[3], (1, self.hidden_dim, H, W))
        return flops

class FusedGhostExpert(nn.Module):
    """Fused Ghost expert that reduces memory traffic by combining operations."""
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        # Use GroupNorm to improve stability
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.GroupNorm(min(num_groups, init_channels), init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.GroupNorm(min(num_groups, new_channels), new_channels),
            nn.SiLU(inplace=True)
        )
        self.init_channels = init_channels
        
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.cheap_operation[0], (1, self.init_channels, H, W))
        return flops

class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x): return self.conv(x)
    def compute_flops(self, input_shape): return FlopsUtils.count_conv2d(self.conv, input_shape)

class GhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv, input_shape)
        # Compute input shape to cheap op (output of primary conv)
        p_out = self.primary_conv[0].out_channels
        flops += FlopsUtils.count_conv2d(self.cheap_operation, (B, p_out, H, W))
        return flops

class InvertedResidualExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_expand = expand_ratio != 1
        layers = []
        if self.use_expand:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x): return self.conv(x)
    def compute_flops(self, input_shape): return FlopsUtils.count_conv2d(self.conv, input_shape)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(EfficientExpertGroup, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        if not hasattr(self, "conv"):
            out_c = x.shape[1]
            self.conv = DepthwiseSeparableConv(x.shape[1], out_c, 3, 1)
        return self.conv(x)
