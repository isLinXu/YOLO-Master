---
Title: YOLO Master WebUI Demo
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# YOLO Master WebUI Demo

This Space runs a Gradio-based YOLO Master WebUI demo.

> import os
> import gc
> import warnings
> from pathlib import Path
> from typing import List, Dict, Optional, Tuple, Any
> 
> import gradio as gr
> import numpy as np
> import pandas as pd
> import cv2
> import torch
> from ultralytics import YOLO
> try:
>     from huggingface_hub import hf_hub_download
> except Exception:
>     hf_hub_download = None
> 
> # Ignore unnecessary warnings
> warnings.filterwarnings("ignore")
> 
> 
> class GlobalConfig:
>     """Global configuration parameters for easy modification."""
>     # Default model files mapping
>     DEFAULT_MODELS = {
>         "detect": "ckpts/yolo-master-v0.1-n.pt",
>         "seg": "ckpts/yolo-master-seg-n.pt",
>         "cls": "ckpts/yolo-master-cls-n.pt",
>         "pose": "yolov8n-pose.pt",
>         "obb": "yolov8n-obb.pt"
>     }
>     # Allowed image formats
>     IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
>     # UI Theme
>     THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
>     DEFAULT_IMAGE_DIR = "./image"
> 
> 
> class ModelManager:
>     """Handles model scanning, loading, and memory management."""
>     def __init__(self, ckpts_root: Path):
>         self.ckpts_root = ckpts_root
>         self.current_model: Optional[YOLO] = None
>         self.current_model_path: str = ""
>         self.current_task: str = "detect"
> 
>     def scan_checkpoints(self) -> Dict[str, List[str]]:
>         """
>         Scans the checkpoint directory and categorizes models by task.
>         """
>         model_map = {k: [] for k in GlobalConfig.DEFAULT_MODELS.keys()}
>         
>         if not self.ckpts_root.exists():
>             return model_map
> 
>         # Recursively find all .pt files
>         for p in self.ckpts_root.rglob("*.pt"):
>             if p.is_dir(): continue 
>             
>             path_str = str(p.absolute())
>             filename = p.name.lower()
>             parent = p.parent.name.lower()
>             
>             # Intelligent classification logic
>             if "seg" in filename or "seg" in parent:
>                 model_map["seg"].append(path_str)
>             elif "cls" in filename or "class" in filename or "cls" in parent:
>                 model_map["cls"].append(path_str)
>             elif "pose" in filename or "pose" in parent:
>                 model_map["pose"].append(path_str)
>             elif "obb" in filename or "obb" in parent:
>                 model_map["obb"].append(path_str)
>             else:
>                 model_map["detect"].append(path_str) # Default to detect
> 
>         # Deduplicate and sort
>         for k in model_map:
>             model_map[k] = sorted(list(set(model_map[k])))
>             
>         return model_map
