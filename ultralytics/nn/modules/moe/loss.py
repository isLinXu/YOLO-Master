# 🐧Please note that this file has been modified by Tencent on 2026/01/18. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Auxiliary losses for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoss(nn.Module):
    """
    Computes auxiliary losses for Mixture of Experts (MoE) models,
    specifically Load Balancing Loss and Z-Loss.
    """

    def __init__(self, balance_loss_coeff=0.01, z_loss_coeff=1e-3, num_experts=4, top_k=2):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, router_probs, router_logits, expert_indices):
        """
        Args:
            router_probs (torch.Tensor): Probability distribution predicted by router [B, E]
            router_logits (torch.Tensor): Logits from router [B, E]
            expert_indices (torch.Tensor): Selected expert indices [B, k]

        Returns:
            torch.Tensor: Combined auxiliary loss
        """
        # 1) Load Balancing Loss
        # importance: probability distribution predicted by router (differentiable)
        importance = router_probs.mean(dim=0)

        # usage: which experts were actually selected (non-differentiable, detached)
        usage_mask = torch.zeros_like(router_probs)
        for k in range(self.top_k):
            usage_mask.scatter_(1, expert_indices[:, k].unsqueeze(1), 1.0)
        usage = usage_mask.mean(dim=0)

        balance_loss = self.num_experts * torch.sum(importance * usage.detach())

        # 2) Z-Loss (numerical stability)
        # Penalize square of log(sum(exp(logits))) to prevent logits from exploding.
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=1) ** 2)

        return (self.balance_loss_coeff * balance_loss) + (self.z_loss_coeff * z_loss)
