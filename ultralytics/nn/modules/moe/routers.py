# 🐧Please note that this file has been modified by Tencent on 2026/01/18. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Efficient routers for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .utils import FlopsUtils, get_safe_groups


# ==========================================
# Ultra-lightweight Router (core optimization)
# ==========================================
class UltraEfficientRouter(nn.Module):
    """
    Ultra-efficient router:
    1) Depthwise-separable convolution instead of standard conv
    2) Aggressive downsampling (8x)
    3) Early channel compression
    4) Improved numerical stability

    Expected FLOPs reduction: ~95% vs a local router baseline.
    """

    def __init__(self, in_channels, num_experts, reduction=16, top_k=2,
                 noise_std=1.0, temperature: float = 1.0, pool_scale=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.temperature = max(float(temperature), 1e-3)
        self.pool_scale = pool_scale

        # More aggressive channel compression
        reduced_channels = max(in_channels // reduction, 4)

        # Depthwise-separable conv: compute ~ 1/(kernel_size^2) of standard conv
        self.router = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(get_safe_groups(in_channels, 8), in_channels),
            nn.SiLU(inplace=True),
            # Pointwise compression
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced_channels, 4), reduced_channels),
            nn.SiLU(inplace=True),
            # Expert projection
            nn.Conv2d(reduced_channels, num_experts, 1, bias=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, C, H, W = x.shape

        # 1) Aggressive downsampling (core optimization)
        if H > self.pool_scale and W > self.pool_scale:
            x_down = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_down = x

        # 2) Lightweight convolutional routing
        logits = self.router(x_down)

        # 3) Z-loss computation (numerical stability)
        z_loss_metric = None
        if self.training:
            # Use clamp instead of tanh for better performance
            logits_safe = logits.clamp(-10.0, 10.0)
            z_loss_metric = torch.logsumexp(logits_safe, dim=1).pow(2).mean()

        # 4) Noise injection
        if self.training and self.noise_std > 0:
            logits.add_(torch.randn_like(logits).mul_(self.noise_std))

        # 5) Softmax + TopK (fused operation)
        weights = self.softmax(logits / self.temperature)
        pooled_weights = weights.mean(dim=[2, 3], keepdim=True)

        topk_vals, topk_indices = torch.topk(pooled_weights, self.top_k, dim=1)

        # In-place normalization
        topk_vals.div_(topk_vals.sum(dim=1, keepdim=True).add_(1e-9))

        if self.training:
            importance = pooled_weights.sum(dim=0).view(self.num_experts)

            # Optimization: use one_hot instead of scatter
            topk_indices_flat = topk_indices.view(B, self.top_k, 1, 1)[:, :, 0, 0]
            mask = F.one_hot(topk_indices_flat, num_classes=self.num_experts).float()
            usage_frequency = mask.sum(dim=[0, 1]) / (B * self.top_k)

            return topk_vals, topk_indices, usage_frequency, importance, z_loss_metric
        else:
            return topk_vals, topk_indices, None, None, None

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_down = max(H // self.pool_scale, 1)
        w_down = max(W // self.pool_scale, 1)

        flops = B * C * H * W  # AvgPool

        input_down_shape = (B, C, h_down, w_down)

        # Depthwise conv
        flops += FlopsUtils.count_conv2d(self.router[0], input_down_shape)
        # Pointwise conv
        flops += FlopsUtils.count_conv2d(self.router[3], (B, self.router[0].out_channels, h_down, w_down))
        # Expert projection
        flops += FlopsUtils.count_conv2d(self.router[6], (B, self.router[3].out_channels, h_down, w_down))

        return flops


class BaseRouter(nn.Module):
    def __init__(self, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=1)

    def _process_logits(self, logits: torch.Tensor, noise_std: float, training: bool) -> Tuple[
        torch.Tensor, torch.Tensor, Dict]:
        """Unified logic to process logits into Top-K selection."""
        B = logits.shape[0]

        # 1) Add noise during training (simplified Gumbel-Softmax trick)
        if training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std

        # 2) Compute probabilities
        probs = self.softmax(logits)

        # 3) Select Top-K
        topk_vals, topk_indices = torch.topk(probs, self.top_k, dim=1)

        # 4) Normalize weights
        sum_vals = topk_vals.sum(dim=1, keepdim=True) + 1e-6
        topk_vals = topk_vals / sum_vals

        # 5) Collect loss-related info (train only)
        loss_dict = {}
        if training:
            loss_dict['router_logits'] = logits
            loss_dict['router_probs'] = probs
            loss_dict['topk_indices'] = topk_indices

        return topk_vals, topk_indices, loss_dict


class EfficientSpatialRouter(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0, pool_scale=4):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        self.pool_scale = pool_scale
        reduced_channels = max(in_channels // reduction, 8)

        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)  # numerical stability
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Pre-pooling optimization
        if H > self.pool_scale and W > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)  # [B, E, H', W']
        global_logits = torch.mean(out, dim=[2, 3])  # [B, E]

        return self._process_logits(global_logits, self.noise_std, self.training)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_down, w_down = max(H // self.pool_scale, 1), max(W // self.pool_scale, 1)
        return FlopsUtils.count_conv2d(self.router, (B, C, h_down, w_down))


class AdaptiveRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        reduced_channels = max(in_channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )

    def forward(self, x):
        pooled = self.avg_pool(x)
        logits = self.router(pooled).squeeze(-1).squeeze(-1)  # [B, E]
        return self._process_logits(logits, self.noise_std, self.training)

    def compute_flops(self, input_shape):
        # FLOPs here are minimal
        return FlopsUtils.count_conv2d(self.router, (input_shape[0], input_shape[1], 1, 1))


class LocalRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        # Even for local routing, default to 2x downsampling to save FLOPs with minimal texture loss
        self.pool_scale = 2

        reduced_channels = max(in_channels // reduction, 8)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )

    def forward(self, x):
        # Moderate downsampling to accelerate
        if x.shape[2] > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)
        global_logits = torch.mean(out, dim=[2, 3])
        return self._process_logits(global_logits, self.noise_std, self.training)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_d, w_d = max(H // self.pool_scale, 1), max(W // self.pool_scale, 1)
        return FlopsUtils.count_conv2d(self.router, (B, C, h_d, w_d))


class AdvancedRoutingLayer(nn.Module):
    """Compatibility router used by some legacy checkpoints; behaves like a global average-pooling router."""

    def __init__(self, in_channels=64, num_experts=3, top_k=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts if top_k is None else min(top_k, num_experts)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not hasattr(self, "router"):
            reduced = max(in_channels // 8, 8)
            self.router = nn.Sequential(
                nn.Conv2d(in_channels, reduced, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(reduced, num_experts, 1, bias=True),
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if not hasattr(self, "avg_pool"):
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not hasattr(self, "softmax"):
            self.softmax = nn.Softmax(dim=1)
        if not hasattr(self, "router"):
            reduced = max(C // 8, 8)
            self.router = nn.Sequential(
                nn.Conv2d(C, reduced, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(reduced, getattr(self, "num_experts", 3), 1, bias=True),
            )
        pooled = self.avg_pool(x)
        if hasattr(self, "router") and isinstance(self.router, nn.Sequential) and len(self.router) > 0 and isinstance(
                self.router[0], nn.Conv2d):
            expected_in = self.router[0].in_channels
            if expected_in != C:
                if not hasattr(self, "_proj") or not isinstance(self._proj,
                                                                nn.Conv2d) or self._proj.in_channels != C or self._proj.out_channels != expected_in:
                    self._proj = nn.Conv2d(C, expected_in, 1, bias=False)
                pooled = self._proj(pooled)
        logits = self.router(pooled)
        probs = self.softmax(logits)
        E = probs.shape[1]
        k = getattr(self, "top_k", E)
        k = max(1, min(k, E))
        if k < E:
            vals, idx = torch.topk(probs, k, dim=1)
            vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-6)
            weights = torch.zeros_like(probs)
            weights.scatter_(1, idx, vals)
        else:
            weights = probs
        return weights.repeat(1, 1, H, W)


class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=8, top_k=None):
        """
        Args:
            top_k: Number of active experts; if None uses all experts (Softmax)
        """
        super(DynamicRoutingLayer, self).__init__()
        reduced_channels = max(in_channels // reduction, 8)

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = (top_k is not None)  # whether to enable Top-K

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Remove Softmax and control manually
        self.routing_network = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, kernel_size=1),
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        routing_logits = self.routing_network(pooled)  # [B, num_experts, 1, 1]

        # Choose strategy based on Top-K enablement and train/infer mode
        if not self.use_top_k:
            # No Top-K: direct Softmax
            routing_weights = F.softmax(routing_logits, dim=1)
        elif self.training:
            # Training: soft Top-K (keeps gradients flowing)
            routing_weights = self._soft_top_k(routing_logits)
        else:
            # Inference: hard Top-K (truly sparse)
            routing_weights = self._hard_top_k(routing_logits)

        return routing_weights.repeat(1, 1, x.size(2), x.size(3))

    def _soft_top_k(self, logits):
        """Soft Top-K during training to maintain gradient flow."""
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)

        # Compute softmax
        weights = F.softmax(logits_flat, dim=1)

        # Find Top-K and build mask
        _, topk_indices = torch.topk(weights, self.top_k, dim=1)
        idx = topk_indices.permute(0, 2, 1).contiguous()
        mask_one_hot = F.one_hot(idx, num_classes=E).sum(dim=2)
        mask_one_hot = mask_one_hot.permute(0, 2, 1).contiguous().to(weights.dtype)

        # Apply mask and re-normalize
        weights = weights * mask_one_hot
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        return weights.view(B, E, H, W)

    def _hard_top_k(self, logits):
        """Hard Top-K during inference for true sparsity."""
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)

        # Find Top-K
        topk_values, topk_indices = torch.topk(logits_flat, self.top_k, dim=1)

        # Apply softmax to Top-K logits
        topk_weights = F.softmax(topk_values, dim=1)

        # Construct sparse weights
        idx = topk_indices.permute(0, 2, 1).contiguous()
        oh = F.one_hot(idx, num_classes=E)
        tw = topk_weights.permute(0, 2, 1).contiguous()
        weighted = (oh.to(tw.dtype) * tw.unsqueeze(-1)).sum(dim=2)
        weights = weighted.permute(0, 2, 1).contiguous()

        return weights.view(B, E, H, W)
