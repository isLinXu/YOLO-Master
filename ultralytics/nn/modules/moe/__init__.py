"""
Mixture-of-Experts (MoE) modules, routing layers, and compatibility shims.

This module provides several MoE variants and routers optimized for inference efficiency,
plus backward-compatibility aliases so legacy checkpoints can be loaded without changes.
"""

from .modules import (
    UltraOptimizedMoE,
    AdaptiveCapacityMoE,
    ES_MOE,
    OptimizedMOE,
    OptimizedMOEImproved,
    MOE,
    EfficientSpatialRouterMoE,
    ModularRouterExpertMoE,
)

from .experts import (
    OptimizedSimpleExpert,
    FusedGhostExpert,
    SimpleExpert,
    GhostExpert,
    InvertedResidualExpert,
    EfficientExpertGroup,
    DepthwiseSeparableConv
)

from .routers import (
    UltraEfficientRouter,
    BaseRouter,
    EfficientSpatialRouter,
    AdaptiveRoutingLayer,
    LocalRoutingLayer,
    AdvancedRoutingLayer,
    DynamicRoutingLayer
)

from .utils import (
    FlopsUtils,
    get_safe_groups,
    BatchedExpertComputation
)

__all__ = [
    "UltraOptimizedMoE",
    "AdaptiveCapacityMoE",
    "ES_MOE",
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "MOE",
    "EfficientSpatialRouterMoE",
    "ModularRouterExpertMoE",
    "OptimizedSimpleExpert",
    "FusedGhostExpert",
    "SimpleExpert",
    "GhostExpert",
    "InvertedResidualExpert",
    "EfficientExpertGroup",
    "DepthwiseSeparableConv",
    "UltraEfficientRouter",
    "BaseRouter",
    "EfficientSpatialRouter",
    "AdaptiveRoutingLayer",
    "LocalRoutingLayer",
    "AdvancedRoutingLayer",
    "DynamicRoutingLayer",
    "FlopsUtils",
    "get_safe_groups",
    "BatchedExpertComputation"
]
