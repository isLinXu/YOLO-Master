import os
import gc
import json
import base64
import time
import re
import urllib.error
import urllib.request
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import gradio as gr
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")


class GlobalConfig:
    """Global configuration parameters for easy modification."""
    PROJECT_URL = "https://github.com/Tencent/YOLO-Master"
    MASCOT_IMAGE_URL = "https://github.com/user-attachments/assets/bbf751ea-af27-465d-a8a9-7822db343638"
    STREAM_DISABLED_OPTIONS = {"half", "show", "save", "save_txt", "save_crop"}
    AGENT_DISABLED_OPTIONS = {"half", "show", "save", "save_txt", "save_crop", "hide_labels", "hide_conf"}
    STREAM_PRESETS = {
        "Realtime": {"max_side": 416, "interval": 0.06, "stride": 2, "max_det": 60},
        "Fast": {"max_side": 480, "interval": 0.10, "stride": 3, "max_det": 80},
        "Balanced": {"max_side": 640, "interval": 0.15, "stride": 2, "max_det": 120},
        "Quality": {"max_side": 832, "interval": 0.20, "stride": 1, "max_det": 200},
    }
    STREAM_BOX_COLORS = (
        (56, 189, 248),
        (52, 211, 153),
        (251, 191, 36),
        (248, 113, 113),
        (167, 139, 250),
        (244, 114, 182),
        (45, 212, 191),
        (250, 204, 21),
    )
    AGENT_API_PROVIDERS = {
        "OpenAI": {
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "api_mode": "auto",
            "vlm_model": "gpt-4.1-mini",
            "llm_model": "",
        },
        "DashScope": {
            "provider": "dashscope",
            "api_key_env": "DASHSCOPE_API_KEY",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_mode": "chat.completions",
            "vlm_model": "qwen-vl-max",
            "llm_model": "",
        },
        "Custom": {
            "provider": "custom",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "api_mode": "chat.completions",
            "vlm_model": "gpt-4.1-mini",
            "llm_model": "",
        },
    }
    AGENT_PROMPT_TEMPLATES = [
        "vlm_coco_multitask",
        "vlm_open_world_detection",
        "vlm_open_world_detection_compact",
        "vlm_open_world_detect_classify_compact",
        "vlm_open_world_caption_misses_compact",
    ]
    # Default model files mapping
    DEFAULT_MODELS = {
        "detect": "ckpts/yolo-master-v0.1-n.pt",
        "seg": "ckpts/yolo-master-seg-n.pt",
        "cls": "ckpts/yolo-master-cls-n.pt",
        "pose": "yolov8n-pose.pt",
        "obb": "yolov8n-obb.pt"
    }
    # Allowed image formats
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # UI Theme
    THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    DEFAULT_IMAGE_DIR = "./image"


class ModelManager:
    """Handles model scanning, loading, and memory management."""
    def __init__(self, ckpts_root: Path):
        self.ckpts_root = ckpts_root
        self.current_model: Optional[YOLO] = None
        self.current_model_path: str = ""
        self.current_task: str = "detect"

    def scan_checkpoints(self) -> Dict[str, List[str]]:
        """
        Scans the checkpoint directory and categorizes models by task.
        """
        model_map = {k: [] for k in GlobalConfig.DEFAULT_MODELS.keys()}
        
        if not self.ckpts_root.exists():
            return model_map

        # Recursively find all .pt files
        for p in self.ckpts_root.rglob("*.pt"):
            if p.is_dir(): continue 
            
            path_str = str(p.absolute())
            filename = p.name.lower()
            parent = p.parent.name.lower()
            
            # Intelligent classification logic
            if "seg" in filename or "seg" in parent:
                model_map["seg"].append(path_str)
            elif "cls" in filename or "class" in filename or "cls" in parent:
                model_map["cls"].append(path_str)
            elif "pose" in filename or "pose" in parent:
                model_map["pose"].append(path_str)
            elif "obb" in filename or "obb" in parent:
                model_map["obb"].append(path_str)
            else:
                model_map["detect"].append(path_str) # Default to detect

        # Deduplicate and sort
        for k in model_map:
            model_map[k] = sorted(list(set(model_map[k])))
            
        return model_map

    def unload_model(self):
        """Force clear GPU memory."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("INFO: Memory cleared.")

    def load_model(self, model_path: str, task: str) -> YOLO:
        """Load model with caching and memory management."""
        target_path = model_path
        if not target_path or not os.path.exists(target_path):
            target_path = GlobalConfig.DEFAULT_MODELS.get(task, "yolov8n.pt")
            if not os.path.exists(target_path):
                repo_id = os.environ.get("YOLO_MASTER_WEIGHTS_REPO", "")
                if hf_hub_download and repo_id:
                    try:
                        fname = Path(target_path).name
                        local_dir = Path(__file__).parent / "ckpts"
                        local_dir.mkdir(parents=True, exist_ok=True)
                        dl = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="model", local_dir=str(local_dir))
                        target_path = dl
                    except Exception:
                        pass
                else:
                    if task == "detect":
                        target_path = "yolov8n.pt"
                    elif task == "seg":
                        target_path = "yolov8n-seg.pt"
                    elif task == "cls":
                        target_path = "yolov8n-cls.pt"
        else:
            # Support directory path, auto-resolve to weights file
            if os.path.isdir(target_path):
                candidates = [
                    os.path.join(target_path, "weights", "best.pt"),
                    os.path.join(target_path, "weights", "last.pt"),
                    os.path.join(target_path, "best.pt"),
                    os.path.join(target_path, "last.pt"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        target_path = c
                        break

        if self.current_model is not None and self.current_model_path == target_path:
            return self.current_model

        self.unload_model()

        print(f"INFO: Loading model from {target_path}...")
        try:
            model = YOLO(target_path)
            self.current_model = model
            self.current_model_path = target_path
            self.current_task = task
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def get_current_model_info(self):
        """Returns device info of the current loaded model."""
        try:
            if self.current_model:
                return str(next(self.current_model.model.parameters()).device)
        except Exception:
            pass
        return "unknown"


class YOLO_Master_WebUI:
    def __init__(self, ckpts_root: str):
        self.ckpts_root = Path(ckpts_root)
        self.model_manager = ModelManager(self.ckpts_root)
        self.model_map = self.model_manager.scan_checkpoints()

    def brand_header(self) -> str:
        return f"""
        <style>
            .ym-brand {{
                position: relative;
                overflow: hidden;
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: 22px;
                align-items: center;
                padding: 18px 22px;
                margin-bottom: 16px;
                border: 1px solid rgba(102, 153, 255, 0.26);
                border-radius: 18px;
                background:
                    linear-gradient(135deg, rgba(21, 32, 55, 0.98), rgba(10, 15, 30, 0.94)),
                    radial-gradient(circle at 80% 20%, rgba(45, 196, 255, 0.22), transparent 28%);
                box-shadow: 0 18px 46px rgba(5, 12, 28, 0.26);
            }}
            .ym-brand:before {{
                content: "";
                position: absolute;
                inset: 0;
                background-image:
                    linear-gradient(rgba(255,255,255,0.045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px);
                background-size: 28px 28px;
                mask-image: linear-gradient(90deg, rgba(0,0,0,0.7), transparent);
                pointer-events: none;
            }}
            .ym-brand-main {{
                position: relative;
                z-index: 1;
                min-width: 0;
            }}
            .ym-brand-kicker {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                align-items: center;
                margin-bottom: 8px;
            }}
            .ym-pill {{
                display: inline-flex;
                align-items: center;
                min-height: 26px;
                padding: 4px 10px;
                border-radius: 999px;
                color: #eaf4ff;
                background: rgba(55, 94, 160, 0.28);
                border: 1px solid rgba(140, 185, 255, 0.28);
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0;
                white-space: nowrap;
            }}
            .ym-pill-cvpr {{
                color: #102032;
                background: linear-gradient(135deg, #8be9ff, #75ffa8);
                border-color: rgba(255, 255, 255, 0.42);
            }}
            .ym-title {{
                margin: 0;
                color: #ffffff;
                font-size: clamp(30px, 4vw, 52px);
                line-height: 0.96;
                font-weight: 900;
                letter-spacing: 0;
            }}
            .ym-subtitle {{
                max-width: 850px;
                margin: 10px 0 14px;
                color: rgba(235, 245, 255, 0.82);
                font-size: 15px;
                line-height: 1.55;
            }}
            .ym-actions {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .ym-action {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 38px;
                padding: 8px 14px;
                border-radius: 10px;
                text-decoration: none !important;
                font-weight: 800;
                font-size: 14px;
                letter-spacing: 0;
                transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
            }}
            .ym-action:hover {{
                transform: translateY(-1px);
            }}
            .ym-action-primary {{
                color: #08111f !important;
                background: #ffffff;
                border: 1px solid rgba(255,255,255,0.72);
            }}
            .ym-action-secondary {{
                color: #f4fbff !important;
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.20);
            }}
            .ym-mascot-link {{
                position: relative;
                z-index: 1;
                display: block;
                width: clamp(116px, 12vw, 166px);
                aspect-ratio: 1;
                border-radius: 18px;
                overflow: hidden;
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.20);
                box-shadow: 0 16px 34px rgba(0, 0, 0, 0.22);
            }}
            .ym-mascot {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }}
            @media (max-width: 760px) {{
                .ym-brand {{
                    grid-template-columns: 1fr;
                    padding: 16px;
                }}
                .ym-mascot-link {{
                    width: 112px;
                    justify-self: start;
                }}
                .ym-title {{
                    font-size: 32px;
                }}
            }}
        </style>
        <section class="ym-brand" aria-label="YOLO-Master project banner">
            <div class="ym-brand-main">
                <div class="ym-brand-kicker">
                    <span class="ym-pill ym-pill-cvpr">CVPR 2026</span>
                    <span class="ym-pill">Tencent Youtu Lab</span>
                    <span class="ym-pill">ES-MoE RTOD</span>
                </div>
                <h1 class="ym-title">YOLO-Master WebUI</h1>
                <p class="ym-subtitle">
                    MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection.
                    Try the demo here, then visit the official Tencent/YOLO-Master repository for code,
                    models, citation, and updates.
                </p>
                <div class="ym-actions">
                    <a class="ym-action ym-action-primary" href="{GlobalConfig.PROJECT_URL}" target="_blank" rel="noopener noreferrer">
                        Star Tencent/YOLO-Master
                    </a>
                    <a class="ym-action ym-action-secondary" href="https://github.com/Tencent/YOLO-Master#-citation" target="_blank" rel="noopener noreferrer">
                        CVPR 2026 Citation
                    </a>
                </div>
            </div>
            <a class="ym-mascot-link" href="{GlobalConfig.PROJECT_URL}" target="_blank" rel="noopener noreferrer" aria-label="Open Tencent YOLO-Master on GitHub">
                <img class="ym-mascot" src="{GlobalConfig.MASCOT_IMAGE_URL}" alt="YOLO-Master Tencent mascot">
            </a>
        </section>
        """
    
    def load_default_image(self) -> Optional[np.ndarray]:
        p = Path(GlobalConfig.DEFAULT_IMAGE_DIR)
        if not p.exists() or not p.is_dir():
            return None
        files = []
        for ext in GlobalConfig.IMAGE_EXTENSIONS:
            files += sorted(p.glob(f"*{ext}"))
        if not files:
            return None
        img = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def resize_for_stream(image: np.ndarray, max_side: float) -> np.ndarray:
        if image is None or max_side is None or max_side <= 0:
            return image
        h, w = image.shape[:2]
        longest_side = max(h, w)
        if longest_side <= max_side:
            return image
        scale = float(max_side) / float(longest_side)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def selected_model_path(model_dropdown: str, custom_model_path: str) -> str:
        return (custom_model_path or "").strip() or (model_dropdown or "").strip()

    @staticmethod
    def detection_count(result: Any) -> int:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return 0
        try:
            return len(boxes)
        except Exception:
            return 0

    @staticmethod
    def class_name(names: Any, cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        try:
            return str(names[cls_id])
        except Exception:
            return str(cls_id)

    @staticmethod
    def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def stream_color(cls_id: int) -> Tuple[int, int, int]:
        colors = GlobalConfig.STREAM_BOX_COLORS
        return colors[cls_id % len(colors)]

    def extract_stream_detections(self, result: Any, names: Any) -> List[Dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            try:
                cls_id = int(box.cls[0]) if box.cls is not None and box.cls.numel() > 0 else 0
                conf_val = float(box.conf[0]) if box.conf is not None and box.conf.numel() > 0 else 0.0
                coords = box.xyxy[0].detach().cpu().tolist()
                detections.append({
                    "cls_id": cls_id,
                    "class_name": self.class_name(names, cls_id),
                    "confidence": conf_val,
                    "xyxy": [float(v) for v in coords],
                })
            except Exception:
                continue
        return detections

    @staticmethod
    def scale_stream_detections(
        detections: List[Dict[str, Any]],
        from_shape: Optional[Tuple[int, int]],
        to_shape: Optional[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        if not detections or not from_shape or not to_shape:
            return [dict(det) for det in detections or []]

        from_h, from_w = float(from_shape[0]), float(from_shape[1])
        to_h, to_w = float(to_shape[0]), float(to_shape[1])
        if from_h <= 0 or from_w <= 0 or to_h <= 0 or to_w <= 0:
            return [dict(det) for det in detections]

        sx, sy = to_w / from_w, to_h / from_h
        scaled = []
        for det in detections:
            det_copy = dict(det)
            x1, y1, x2, y2 = det_copy.get("xyxy", [0.0, 0.0, 0.0, 0.0])
            det_copy["xyxy"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            scaled.append(det_copy)
        return scaled

    def smooth_stream_detections(
        self,
        detections: List[Dict[str, Any]],
        previous_detections: List[Dict[str, Any]],
        enabled: bool,
    ) -> List[Dict[str, Any]]:
        if not enabled or not detections or not previous_detections:
            return detections

        smoothed = []
        used_previous = set()
        new_weight = 0.68
        for det in detections:
            best_idx = -1
            best_iou = 0.0
            for idx, prev in enumerate(previous_detections):
                if idx in used_previous or prev.get("cls_id") != det.get("cls_id"):
                    continue
                iou = self.bbox_iou(det.get("xyxy", []), prev.get("xyxy", []))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= 0.35:
                prev_box = previous_detections[best_idx].get("xyxy", det["xyxy"])
                new_box = det["xyxy"]
                det = dict(det)
                det["xyxy"] = [
                    prev_box[i] * (1.0 - new_weight) + new_box[i] * new_weight
                    for i in range(4)
                ]
                used_previous.add(best_idx)
            smoothed.append(det)
        return smoothed

    def draw_stream_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        line_width: float,
        hide_labels: bool,
        hide_conf: bool,
    ) -> np.ndarray:
        annotated = image.copy()
        h, w = annotated.shape[:2]
        lw = int(line_width) if line_width and line_width > 0 else max(2, round((h + w) / 640))
        font_scale = max(0.42, min(0.72, lw * 0.22))
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            xyxy = det.get("xyxy", [0.0, 0.0, 0.0, 0.0])
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            color = self.stream_color(int(det.get("cls_id", 0)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)

            if hide_labels and hide_conf:
                continue
            if hide_labels:
                label = f"{float(det.get('confidence', 0.0)):.2f}"
            elif hide_conf:
                label = str(det.get("class_name", det.get("cls_id", "")))
            else:
                label = f"{det.get('class_name', det.get('cls_id', ''))} {float(det.get('confidence', 0.0)):.2f}"

            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, max(1, lw - 1))
            label_y1 = max(0, y1 - label_h - baseline - 6)
            label_y2 = label_y1 + label_h + baseline + 6
            label_x2 = min(w - 1, x1 + label_w + 8)
            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y2 - baseline - 3),
                font,
                font_scale,
                (8, 16, 28),
                max(1, lw - 1),
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): YOLO_Master_WebUI.json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [YOLO_Master_WebUI.json_safe(v) for v in value]
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def listify_report_value(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def report_value_text(self, value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(self.json_safe(value), ensure_ascii=False)
        return str(value)

    @staticmethod
    def caption_text(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("short") or value.get("dense") or "")
        return str(value or "")

    @staticmethod
    def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        cleaned = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        def balanced_fragment(source: str) -> Optional[str]:
            start = None
            depth = 0
            in_string = False
            escape = False
            for idx, ch in enumerate(source):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    if start is None:
                        start = idx
                    depth += 1
                elif ch == "}":
                    if start is None:
                        continue
                    depth -= 1
                    if depth == 0:
                        return source[start:idx + 1]
            return None

        for candidate in (balanced_fragment(cleaned), cleaned):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass
        decoder = json.JSONDecoder()
        for match in re.finditer(r"{", cleaned):
            try:
                parsed, _ = decoder.raw_decode(cleaned[match.start():])
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def encode_image_data_url(image: np.ndarray, max_side: int = 1280, quality: int = 90) -> Optional[str]:
        if image is None:
            return None
        img = image
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = float(max_side) / float(longest)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    def mark_agent_image(self, image: np.ndarray, detections: List[Dict[str, Any]], line_width: float) -> np.ndarray:
        marked = image.copy()
        h, w = marked.shape[:2]
        lw = int(line_width) if line_width and line_width > 0 else max(2, round((h + w) / 650))
        font_scale = max(0.45, min(0.8, lw * 0.24))
        font = cv2.FONT_HERSHEY_SIMPLEX

        for idx, det in enumerate(detections[:40]):
            bbox = det.get("bbox_xyxy") or det.get("xyxy")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            color = self.stream_color(int(det.get("class_id", idx) or idx))
            label = f"#{idx} {det.get('label') or det.get('class_name') or 'object'}"
            if det.get("confidence") is not None:
                label += f" {float(det['confidence']):.2f}"
            cv2.rectangle(marked, (x1, y1), (x2, y2), color, lw)
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, max(1, lw - 1))
            label_y1 = max(0, y1 - label_h - baseline - 8)
            label_y2 = min(h - 1, label_y1 + label_h + baseline + 8)
            label_x2 = min(w - 1, x1 + label_w + 10)
            cv2.rectangle(marked, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                marked,
                label,
                (x1 + 5, label_y2 - baseline - 4),
                font,
                font_scale,
                (4, 10, 18),
                max(1, lw - 1),
                cv2.LINE_AA,
            )
        return marked

    def build_visual_evidence(
        self,
        result: Any,
        task: str,
        image_shape: Tuple[int, int],
        max_items: int = 40,
    ) -> Dict[str, Any]:
        h, w = image_shape[:2]
        names = getattr(result, "names", {}) or {}
        detections: List[Dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                cls = boxes.cls.detach().cpu().tolist()
                conf = boxes.conf.detach().cpu().tolist()
                for idx, coords in enumerate(xyxy[:max_items]):
                    class_id = int(cls[idx]) if idx < len(cls) else None
                    label = self.class_name(names, class_id) if class_id is not None else None
                    detections.append({
                        "index": idx,
                        "class_id": class_id,
                        "label": label,
                        "confidence": round(float(conf[idx]), 4) if idx < len(conf) else None,
                        "bbox_xyxy": [round(float(v), 2) for v in coords[:4]],
                    })
            except Exception:
                pass

        obb_items: List[Dict[str, Any]] = []
        obb = getattr(result, "obb", None)
        if obb is not None:
            try:
                obb_cls = obb.cls.detach().cpu().tolist()
                obb_conf = obb.conf.detach().cpu().tolist()
                xywhr = obb.xywhr.detach().cpu().tolist() if getattr(obb, "xywhr", None) is not None else []
                corners = obb.xyxyxyxy.detach().cpu().tolist() if getattr(obb, "xyxyxyxy", None) is not None else []
                for idx in range(min(max_items, len(obb_cls))):
                    class_id = int(obb_cls[idx])
                    obb_items.append({
                        "index": idx,
                        "class_id": class_id,
                        "label": self.class_name(names, class_id),
                        "confidence": round(float(obb_conf[idx]), 4) if idx < len(obb_conf) else None,
                        "xywhr": [round(float(v), 2) for v in xywhr[idx]] if idx < len(xywhr) else None,
                        "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in corners[idx]] if idx < len(corners) else None,
                    })
            except Exception:
                pass

        masks_summary: List[Dict[str, Any]] = []
        masks = getattr(result, "masks", None)
        if masks is not None:
            try:
                mask_data = masks.data.detach().cpu().numpy()
                polygons = getattr(masks, "xy", None) or []
                for idx, mask in enumerate(mask_data[:min(max_items, 12)]):
                    det = detections[idx] if idx < len(detections) else {}
                    area_px = int(np.count_nonzero(mask > 0.5))
                    area_ratio = float(area_px) / float(mask.size or 1)
                    polygon = []
                    if idx < len(polygons):
                        points = polygons[idx]
                        step = max(1, len(points) // 12)
                        polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in points[::step][:12]]
                    masks_summary.append({
                        "index": idx,
                        "linked_detection_index": det.get("index"),
                        "class_id": det.get("class_id"),
                        "label": det.get("label"),
                        "area_ratio": round(area_ratio, 5),
                        "bbox_xyxy": det.get("bbox_xyxy"),
                        "polygon_sample_xy": polygon,
                    })
            except Exception:
                pass

        classification: List[Dict[str, Any]] = []
        probs = getattr(result, "probs", None)
        if probs is not None:
            try:
                top_ids = list(getattr(probs, "top5", []) or [])[:5]
                top_conf = getattr(probs, "top5conf", None)
                top_conf = top_conf.detach().cpu().tolist() if top_conf is not None else []
                for rank, class_id in enumerate(top_ids):
                    classification.append({
                        "rank": rank + 1,
                        "class_id": int(class_id),
                        "label": self.class_name(names, int(class_id)),
                        "confidence": round(float(top_conf[rank]), 4) if rank < len(top_conf) else None,
                    })
            except Exception:
                pass

        pose: List[Dict[str, Any]] = []
        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None:
            try:
                xy = keypoints.xy.detach().cpu().tolist()
                conf = keypoints.conf.detach().cpu().tolist() if getattr(keypoints, "conf", None) is not None else []
                for idx, points in enumerate(xy[:10]):
                    visible = 0
                    sampled = []
                    for point_idx, point in enumerate(points):
                        kp_conf = conf[idx][point_idx] if idx < len(conf) and point_idx < len(conf[idx]) else None
                        if kp_conf is None or kp_conf > 0.2:
                            visible += 1
                        sampled.append([round(float(point[0]), 1), round(float(point[1]), 1), round(float(kp_conf), 3) if kp_conf is not None else None])
                    pose.append({"index": idx, "visible_keypoints": visible, "keypoints_xy_conf": sampled[:17]})
            except Exception:
                pass

        count_by_class = Counter(item.get("label") or str(item.get("class_id")) for item in detections)
        speed = getattr(result, "speed", {}) or {}
        return {
            "task": task,
            "image": {"width": int(w), "height": int(h)},
            "model": Path(self.model_manager.current_model_path).name if self.model_manager.current_model_path else "",
            "device": self.model_manager.get_current_model_info(),
            "speed_ms": self.json_safe(speed),
            "counts": {
                "boxes": len(detections),
                "masks": len(masks_summary),
                "obb": len(obb_items),
                "classifications": len(classification),
                "pose_instances": len(pose),
                "by_class": dict(count_by_class),
            },
            "detections": detections,
            "segmentation": masks_summary,
            "classification": classification,
            "pose": pose,
            "oriented_boxes": obb_items,
        }

    @staticmethod
    def evidence_to_dataframe(evidence: Dict[str, Any]) -> Optional[pd.DataFrame]:
        rows = []
        for det in evidence.get("detections", []):
            coords = det.get("bbox_xyxy") or [None, None, None, None]
            rows.append({
                "Class ID": det.get("class_id"),
                "Class Name": det.get("label"),
                "Confidence": det.get("confidence"),
                "x1": coords[0],
                "y1": coords[1],
                "x2": coords[2],
                "y2": coords[3],
            })
        return pd.DataFrame(rows) if rows else None

    @staticmethod
    def api_provider_defaults(provider: str) -> Dict[str, str]:
        return dict(GlobalConfig.AGENT_API_PROVIDERS.get(provider, GlobalConfig.AGENT_API_PROVIDERS["OpenAI"]))

    def update_agent_provider(self, provider: str):
        defaults = self.api_provider_defaults(provider)
        return (
            gr.update(value=defaults["base_url"]),
            gr.update(value=defaults["api_mode"]),
            gr.update(value=defaults["vlm_model"]),
            gr.update(value=defaults["llm_model"]),
        )

    def build_agent_prompt(
        self,
        evidence: Dict[str, Any],
        user_prompt: str,
        prompt_template: str,
        structured_output: bool,
        thinking_with_image: bool,
    ) -> str:
        image_instruction = (
            "Privately inspect the image or marked image, compare it with the YOLO-Master evidence, and resolve disagreements."
            if thinking_with_image
            else "Use only the YOLO-Master evidence and the user task; do not assume direct image access."
        )
        prompt = user_prompt.strip() or "请基于 YOLO-Master 的检测、分割、分类或姿态证据生成一份可执行的视觉推理报告。"
        evidence_text = json.dumps(self.json_safe(evidence), ensure_ascii=False, indent=2)
        if structured_output:
            schema = (
                "Return exactly one JSON object without Markdown fences. Use these keys: "
                "answer, caption, visual_evidence, yolo_cross_check, segmentation_analysis, "
                "possible_misses, false_positives, risk_notes, recommended_next_actions, report_markdown."
            )
        else:
            schema = "Return a concise Markdown report with evidence, uncertainty, and recommended next actions."
        template_note = (
            "COCO-oriented multitask report: caption, classification, object evidence, rough segmentation proxies, "
            "YOLO cross-check, and fusion-safe suggestions."
            if "coco" in prompt_template
            else "Open-world report: preserve novel visible categories, likely misses, and mapping uncertainty."
        )
        return (
            "You are a YOLO-Master multimodal perception agent.\n"
            f"Prompt template: {prompt_template}. {template_note}\n"
            f"{image_instruction}\n"
            "Do not reveal hidden chain-of-thought. Keep all claims grounded in the image or YOLO evidence.\n"
            f"{schema}\n\n"
            f"User task:\n{prompt}\n\n"
            f"YOLO-Master evidence envelope:\n{evidence_text}"
        )

    @staticmethod
    def agent_developer_prompt(prompt_template: str, structured_output: bool) -> str:
        if "open_world" in prompt_template:
            base = (
                "You are a careful open-world multimodal perception assistant for YOLO-Master. "
                "Preserve visible novel categories and include COCO mappings only when grounded."
            )
        else:
            base = (
                "You are a careful COCO-oriented visual reasoning assistant for YOLO-Master. "
                "Use the image and detector evidence conservatively."
            )
        if structured_output:
            base += " Return exactly one JSON object that includes every requested top-level key."
        return base

    def call_openai_responses_api(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        user_text: str,
        developer_text: str,
        image_url: Optional[str],
        max_output_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        content = [{"type": "input_text", "text": user_text}]
        if image_url:
            content.append({"type": "input_image", "image_url": image_url, "detail": "auto"})
        body = {
            "model": model,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": developer_text}]},
                {"role": "user", "content": content},
            ],
            "max_output_tokens": int(max_output_tokens),
        }
        if temperature is not None:
            body["temperature"] = float(temperature)
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {"status": "failed", "api_mode": "responses", "summary": f"Responses API HTTP {exc.code}", "error": detail[:1200]}
        except Exception as exc:
            return {"status": "failed", "api_mode": "responses", "summary": str(exc), "error_type": type(exc).__name__}

        chunks = []
        if isinstance(payload.get("output_text"), str):
            chunks.append(payload["output_text"])
        for output in payload.get("output", []) or []:
            for item in output.get("content", []) or []:
                if isinstance(item.get("text"), str):
                    chunks.append(item["text"])
        return {
            "status": "ok",
            "api_mode": "responses",
            "model": model,
            "text": "\n".join(chunks).strip(),
            "response_id": payload.get("id"),
            "usage": self.json_safe(payload.get("usage", {})),
        }

    def call_openai_chat_api(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        user_text: str,
        developer_text: str,
        image_url: Optional[str],
        max_output_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        content = [{"type": "text", "text": user_text}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": developer_text},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_output_tokens),
        }
        if temperature is not None:
            body["temperature"] = float(temperature)
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {"status": "failed", "api_mode": "chat.completions", "summary": f"Chat Completions API HTTP {exc.code}", "error": detail[:1200]}
        except Exception as exc:
            return {"status": "failed", "api_mode": "chat.completions", "summary": str(exc), "error_type": type(exc).__name__}

        chunks = []
        for choice in payload.get("choices", []) or []:
            message = choice.get("message") or {}
            content_value = message.get("content")
            if isinstance(content_value, str):
                chunks.append(content_value)
            elif isinstance(content_value, list):
                chunks += [item["text"] for item in content_value if isinstance(item, dict) and isinstance(item.get("text"), str)]
        return {
            "status": "ok",
            "api_mode": "chat.completions",
            "model": model,
            "text": "\n".join(chunks).strip(),
            "response_id": payload.get("id"),
            "usage": self.json_safe(payload.get("usage", {})),
        }

    def call_agent_api(
        self,
        *,
        provider: str,
        api_key_input: str,
        base_url: str,
        api_mode: str,
        model: str,
        user_text: str,
        developer_text: str,
        image_url: Optional[str],
        max_output_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        defaults = self.api_provider_defaults(provider)
        key_env = defaults.get("api_key_env", "OPENAI_API_KEY")
        api_key = (api_key_input or "").strip() or os.environ.get(key_env)
        if key_env != "OPENAI_API_KEY":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                "status": "blocked",
                "provider": defaults.get("provider", provider.lower()),
                "api_mode": api_mode,
                "summary": f"{key_env} is not set; API reasoning was skipped.",
                "api_key_env": key_env,
            }

        resolved_base = (base_url or defaults["base_url"]).rstrip("/")
        resolved_mode = (api_mode or defaults["api_mode"] or "auto").replace("_", ".").lower()
        resolved_model = (model or defaults["vlm_model"]).strip()
        if resolved_mode in {"chat", "chat.completion", "chat.completions"}:
            result = self.call_openai_chat_api(
                api_key=api_key,
                base_url=resolved_base,
                model=resolved_model,
                user_text=user_text,
                developer_text=developer_text,
                image_url=image_url,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        elif resolved_mode == "responses":
            result = self.call_openai_responses_api(
                api_key=api_key,
                base_url=resolved_base,
                model=resolved_model,
                user_text=user_text,
                developer_text=developer_text,
                image_url=image_url,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        else:
            result = self.call_openai_responses_api(
                api_key=api_key,
                base_url=resolved_base,
                model=resolved_model,
                user_text=user_text,
                developer_text=developer_text,
                image_url=image_url,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            if result.get("status") != "ok":
                fallback = self.call_openai_chat_api(
                    api_key=api_key,
                    base_url=resolved_base,
                    model=resolved_model,
                    user_text=user_text,
                    developer_text=developer_text,
                    image_url=image_url,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
                if fallback.get("status") == "ok":
                    fallback["fallback_from"] = result
                    result = fallback
        result["provider"] = defaults.get("provider", provider.lower())
        return result

    def compose_agent_report(
        self,
        envelope: Dict[str, Any],
        api_result: Dict[str, Any],
        verdict: Optional[Dict[str, Any]],
    ) -> str:
        evidence = envelope.get("yolo_evidence", {})
        counts = evidence.get("counts", {})
        by_class = counts.get("by_class", {})
        class_line = ", ".join(f"{name}: {count}" for name, count in by_class.items()) or "No boxes"
        report = [
            "### YOLO-Master Agent Visual Report",
            f"- **Task:** `{evidence.get('task', '-')}`",
            f"- **Model:** `{evidence.get('model', '-')}`",
            f"- **Image:** `{evidence.get('image', {}).get('width', 0)} x {evidence.get('image', {}).get('height', 0)}`",
            f"- **YOLO Evidence:** boxes `{counts.get('boxes', 0)}`, masks `{counts.get('masks', 0)}`, OBB `{counts.get('obb', 0)}`, pose `{counts.get('pose_instances', 0)}`",
            f"- **Class Counts:** {class_line}",
            "",
        ]

        if verdict:
            if verdict.get("report_markdown"):
                report.append(str(verdict["report_markdown"]))
            else:
                report += [
                    "**Agent Answer**",
                    str(verdict.get("answer") or self.caption_text(verdict.get("caption")) or "API returned a structured verdict."),
                    "",
                    "**Visual Evidence**",
                ]
                for item in self.listify_report_value(verdict.get("visual_evidence"))[:8]:
                    report.append(f"- {self.report_value_text(item)}")
                yolo_cross_check = verdict.get("yolo_cross_check")
                if yolo_cross_check:
                    report += ["", "**YOLO Cross-check**", f"```json\n{json.dumps(self.json_safe(yolo_cross_check), ensure_ascii=False, indent=2)}\n```"]
                next_actions = self.listify_report_value(verdict.get("recommended_next_actions"))
                if next_actions:
                    report += ["", "**Recommended Next Actions**"]
                    report += [f"- {self.report_value_text(item)}" for item in next_actions[:8]]
        elif api_result.get("status") == "ok":
            report += ["**API Report**", str(api_result.get("text") or "API returned no text.")]
        elif api_result.get("status") == "blocked":
            report += [
                "**API Reasoning**",
                f"- {api_result.get('summary')}",
                "- 已输出本地 YOLO-only 报告；在 HF Secrets 或界面里配置 API key 后会自动生成 VLM/LLM 视觉推理报告。",
            ]
        else:
            report += [
                "**API Reasoning**",
                f"- API call status: `{api_result.get('status')}`",
                f"- {api_result.get('summary', 'Unknown API error')}",
            ]

        if evidence.get("segmentation"):
            report += ["", "**Segmentation Evidence**"]
            for mask in evidence["segmentation"][:6]:
                report.append(f"- #{mask.get('index')} `{mask.get('label')}` area_ratio `{mask.get('area_ratio')}`")

        if evidence.get("classification"):
            report += ["", "**Classification Top-k**"]
            for item in evidence["classification"][:5]:
                report.append(f"- {item.get('rank')}. `{item.get('label')}` confidence `{item.get('confidence')}`")

        return "\n".join(report)

    def update_stream_preset(self, preset: str):
        config = GlobalConfig.STREAM_PRESETS.get(preset, GlobalConfig.STREAM_PRESETS["Balanced"])
        return (
            gr.update(value=config["max_side"]),
            gr.update(value=config["interval"]),
            gr.update(value=config["stride"]),
            gr.update(value=config["max_det"]),
        )

    def inference(self, 
                  task: str, 
                  image: np.ndarray, 
                  model_dropdown: str,
                  custom_model_path: str,
                  conf: float, 
                  iou: float, 
                  device: str, 
                  max_det: float, 
                  line_width: float, 
                  cpu: bool,
                  checkboxes: List[str],
                  stream_mode: bool = False,
                  stream_max_side: Optional[float] = None):
        """
        Core inference function.
        Returns: (Annotated Image, Results DataFrame, Summary Text)
        """
        if image is None:
            return None, None, "⚠️ Please upload an image first."

        # 1. Parameter Sanitization
        device_opt = "cpu" if cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width > 0 else None
        max_det_opt = int(max_det)
        enabled_options = set(checkboxes or [])
        if stream_mode:
            enabled_options -= GlobalConfig.STREAM_DISABLED_OPTIONS
        options = {k: True for k in enabled_options}
        if stream_mode and stream_max_side:
            options["imgsz"] = int(stream_max_side)
        options["verbose"] = False
        
        # Optimization for segmentation task
        if task == "seg" and not stream_mode and "retina_masks" not in options:
            options["retina_masks"] = True

        # 2. Model Loading
        # Prioritize custom path, then dropdown
        model_path = (custom_model_path or "").strip() or (model_dropdown or "").strip()
        try:
            model = self.model_manager.load_model(model_path, task)
        except Exception as e:
            return image, None, f"❌ Error loading model: {str(e)}"

        # 3. Execution
        try:
            # Gradio input is RGB, but Ultralytics expects BGR for numpy arrays
            # We convert to BGR to ensure correct inference and plotting colors
            if stream_mode:
                image = self.resize_for_stream(image, stream_max_side or 0)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            with torch.inference_mode():
                results = model(image_bgr,
                                conf=conf,
                                iou=iou,
                                device=device_opt,
                                max_det=max_det_opt,
                                line_width=line_width_opt,
                                **options)
        except Exception as e:
            return image, None, f"❌ Inference Error: {str(e)}"

        # 4. Result Parsing
        res = results[0]
        
        # 4.1 Image Processing
        res_img = res.plot() 
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB) # Convert back to RGB
        
        # 4.2 Data Extraction (Build DataFrame)
        data_list = []
        if res.boxes:
            for box in res.boxes:
                try:
                    # Compatibility handling: box.cls might be tensor or float
                    cls_id = int(box.cls[0]) if box.cls.numel() > 0 else 0
                    cls_name = model.names[cls_id]
                    conf_val = float(box.conf[0]) if box.conf.numel() > 0 else 0.0
                    coords = box.xyxy[0].tolist()
                    
                    row = {
                        "Class ID": cls_id,
                        "Class Name": cls_name,
                        "Confidence": round(conf_val, 3),
                        "x1": round(coords[0], 1),
                        "y1": round(coords[1], 1),
                        "x2": round(coords[2], 1),
                        "y2": round(coords[3], 1)
                    }
                    data_list.append(row)
                except Exception:
                    pass
        
        df = None if stream_mode else pd.DataFrame(data_list)
        
        # 4.3 Summary Info
        speed = res.speed
        infer_time = speed.get('inference', 0.0)
        model_device = self.model_manager.get_current_model_info()
        
        title = "Live Stream Frame" if stream_mode else "Inference Done"
        summary = (
            f"### ✅ {title}\n"
            f"- **Model:** `{Path(self.model_manager.current_model_path).name}`\n"
            f"- **Time:** `{infer_time:.1f}ms`\n"
            f"- **Objects:** {len(data_list)}\n"
            f"- **Device:** `{model_device}`"
        )
        
        return res_img, df, summary

    def generate_agent_report(
        self,
        task: str,
        image: np.ndarray,
        model_dropdown: str,
        custom_model_path: str,
        conf: float,
        iou: float,
        device: str,
        max_det: float,
        line_width: float,
        cpu: bool,
        checkboxes: List[str],
        api_provider: str,
        api_base_url: str,
        api_mode: str,
        api_key: str,
        vlm_model: str,
        llm_model: str,
        prompt_template: str,
        report_prompt: str,
        structured_output: bool,
        thinking_with_image: bool,
        use_marked_image: bool,
        enable_llm_refine: bool,
        max_output_tokens: float,
        temperature: float,
    ):
        if image is None:
            empty = {"status": "blocked", "summary": "No input image."}
            return None, "⚠️ Please upload an image first.", empty, None, None, "⚠️ Please upload an image first."

        device_opt = "cpu" if cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width and line_width > 0 else None
        selected_options = set(checkboxes or [])
        enabled_options = selected_options - GlobalConfig.AGENT_DISABLED_OPTIONS
        disabled_options = sorted(selected_options - enabled_options)
        options = {k: True for k in enabled_options}
        options["verbose"] = False
        if task == "seg" and "retina_masks" not in options:
            options["retina_masks"] = True

        model_path = self.selected_model_path(model_dropdown, custom_model_path)
        try:
            model = self.model_manager.load_model(model_path, task)
        except Exception as e:
            error = {"status": "failed", "summary": f"Error loading model: {str(e)}"}
            return image, f"❌ Error loading model: {str(e)}", error, image, None, f"❌ Error loading model: {str(e)}"

        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            with torch.inference_mode():
                results = model(
                    image_bgr,
                    conf=conf,
                    iou=iou,
                    device=device_opt,
                    max_det=int(max_det),
                    line_width=line_width_opt,
                    **options,
                )
        except Exception as e:
            error = {"status": "failed", "summary": f"Inference Error: {str(e)}"}
            return image, f"❌ Inference Error: {str(e)}", error, image, None, f"❌ Inference Error: {str(e)}"

        result = results[0]
        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        evidence = self.build_visual_evidence(result, task, image.shape[:2])
        report_image = self.mark_agent_image(image, evidence.get("detections", []), line_width) if use_marked_image and evidence.get("detections") else annotated
        report_df = self.evidence_to_dataframe(evidence)

        prompt_text = self.build_agent_prompt(
            evidence,
            report_prompt,
            prompt_template,
            structured_output,
            thinking_with_image,
        )
        developer_text = self.agent_developer_prompt(prompt_template, structured_output)
        image_url = self.encode_image_data_url(report_image if use_marked_image else annotated) if thinking_with_image else None

        agent_request = {
            "skill": "yolo.multimodal.infer",
            "inputs": {
                "model": Path(self.model_manager.current_model_path).name,
                "source": "gradio:image",
                "prompt": report_prompt or "Generate a visual inference report.",
            },
            "params": {
                "thinking_with_image": bool(thinking_with_image),
                "structured_output": bool(structured_output),
                "prompt_template": prompt_template,
                "use_marked_image": bool(use_marked_image),
                "fusion_mode": "preview",
                "provider": api_provider,
                "openai_base_url": api_base_url,
                "openai_api_mode": api_mode,
                "vlm_model": vlm_model,
                "llm_model": llm_model,
                "enable_llm_refine": bool(enable_llm_refine),
                "max_output_tokens": int(max_output_tokens),
                "temperature": float(temperature),
                "disabled_output_options": disabled_options,
            },
            "policy": {"dry_run": False},
        }

        api_result = self.call_agent_api(
            provider=api_provider,
            api_key_input=api_key,
            base_url=api_base_url,
            api_mode=api_mode,
            model=vlm_model,
            user_text=prompt_text,
            developer_text=developer_text,
            image_url=image_url,
            max_output_tokens=int(max_output_tokens),
            temperature=float(temperature),
        )
        verdict = self.extract_json_object(str(api_result.get("text") or "")) if api_result.get("status") == "ok" else None
        if api_result.get("status") == "ok":
            api_result["verdict_parse_status"] = "parsed" if verdict else "unparsed"

        if enable_llm_refine and llm_model and api_result.get("status") == "ok":
            refine_prompt = (
                "Refine the following YOLO-Master multimodal report into a concise Chinese visual inference report. "
                "Preserve JSON facts and uncertainty. Do not invent new evidence.\n\n"
                f"YOLO evidence:\n{json.dumps(self.json_safe(evidence), ensure_ascii=False, indent=2)}\n\n"
                f"VLM result:\n{api_result.get('text', '')}"
            )
            refine_result = self.call_agent_api(
                provider=api_provider,
                api_key_input=api_key,
                base_url=api_base_url,
                api_mode="chat.completions",
                model=llm_model,
                user_text=refine_prompt,
                developer_text="You are a concise verifier for YOLO-Master visual reports.",
                image_url=None,
                max_output_tokens=int(max_output_tokens),
                temperature=float(temperature),
            )
            api_result["llm_refine"] = refine_result
            if refine_result.get("status") == "ok" and refine_result.get("text"):
                refined_verdict = self.extract_json_object(str(refine_result.get("text") or ""))
                verdict = refined_verdict or verdict
                if not refined_verdict:
                    api_result["text"] = refine_result["text"]

        envelope = {
            "status": "ok" if api_result.get("status") in {"ok", "blocked"} else "partial",
            "agent_request": agent_request,
            "yolo_evidence": evidence,
            "api_result": {k: v for k, v in api_result.items() if k != "raw_payload"},
            "verdict": verdict,
        }
        report_md = self.compose_agent_report(envelope, api_result, verdict)
        summary = (
            f"### ✅ Agent Report Ready\n"
            f"- **YOLO task:** `{task}`\n"
            f"- **API status:** `{api_result.get('status')}`\n"
            f"- **Evidence boxes:** `{evidence.get('counts', {}).get('boxes', 0)}`\n"
            f"- **Masks:** `{evidence.get('counts', {}).get('masks', 0)}`"
        )
        if disabled_options:
            summary += f"\n- **Report-safe options disabled:** `{', '.join(disabled_options)}`"
        return report_image, report_md, envelope, annotated, report_df, summary

    def run_stream_inference(
        self,
        task: str,
        frame: np.ndarray,
        model_dropdown: str,
        custom_model_path: str,
        conf: float,
        iou: float,
        device: str,
        line_width: float,
        cpu: bool,
        checkboxes: List[str],
        stream_max_side: float,
        stream_max_det: float,
        previous_detections: List[Dict[str, Any]],
        previous_shape: Optional[Tuple[int, int]],
        smooth_boxes: bool,
    ) -> Tuple[Optional[np.ndarray], str, List[Dict[str, Any]], Optional[Tuple[int, int]], float]:
        """Stream-only inference path: no dataframe work, optional box smoothing."""
        device_opt = "cpu" if cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width and line_width > 0 else None
        max_det_opt = int(stream_max_det)
        enabled_options = set(checkboxes or []) - GlobalConfig.STREAM_DISABLED_OPTIONS
        options = {k: True for k in enabled_options}
        if stream_max_side:
            options["imgsz"] = int(stream_max_side)
        options["verbose"] = False

        model_path = self.selected_model_path(model_dropdown, custom_model_path)
        try:
            model = self.model_manager.load_model(model_path, task)
        except Exception as e:
            return frame, f"❌ Error loading model: {str(e)}", [], None, 0.0

        resized_frame = self.resize_for_stream(frame, stream_max_side or 0)
        frame_shape = resized_frame.shape[:2]
        previous_scaled = self.scale_stream_detections(previous_detections, previous_shape, frame_shape)

        try:
            image_bgr = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            with torch.inference_mode():
                results = model(
                    image_bgr,
                    conf=conf,
                    iou=iou,
                    device=device_opt,
                    max_det=max_det_opt,
                    line_width=line_width_opt,
                    **options,
                )
        except Exception as e:
            return resized_frame, f"❌ Inference Error: {str(e)}", previous_scaled, frame_shape, 0.0

        result = results[0]
        detections = self.extract_stream_detections(result, model.names)
        detections = self.smooth_stream_detections(detections, previous_scaled, smooth_boxes)

        if task == "detect":
            res_img = self.draw_stream_detections(
                resized_frame,
                detections,
                line_width,
                "hide_labels" in enabled_options,
                "hide_conf" in enabled_options,
            )
        else:
            res_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

        speed = getattr(result, "speed", {})
        infer_time = float(speed.get("inference", 0.0))
        model_device = self.model_manager.get_current_model_info()
        summary = (
            f"### ✅ Live Stream Frame\n"
            f"- **Model:** `{Path(self.model_manager.current_model_path).name}`\n"
            f"- **Inference:** `{infer_time:.1f}ms`\n"
            f"- **Objects:** {self.detection_count(result)}\n"
            f"- **Device:** `{model_device}`"
        )
        return res_img, summary, detections, frame_shape, infer_time

    def inference_stream(
        self,
        task: str,
        frame: np.ndarray,
        model_dropdown: str,
        custom_model_path: str,
        conf: float,
        iou: float,
        device: str,
        line_width: float,
        cpu: bool,
        checkboxes: List[str],
        stream_max_side: float,
        min_frame_interval: float,
        frame_stride: float,
        stream_max_det: float,
        smooth_preview: bool,
        smooth_boxes: bool,
        auto_throttle: bool,
        stream_state: Optional[Dict[str, Any]],
    ):
        """Run inference for a webcam frame without refreshing the detections table."""
        stream_state = stream_state or {}
        if frame is None:
            return None, "Waiting for webcam stream...", stream_state

        selected_model = self.selected_model_path(model_dropdown, custom_model_path)
        stream_key = f"{task}|{selected_model}|{int(stream_max_side or 0)}"
        if stream_state.get("stream_key") != stream_key:
            stream_state = {"stream_key": stream_key}

        now = time.monotonic()
        frame_index = int(stream_state.get("frame_index", 0)) + 1
        processed_frames = int(stream_state.get("processed_frames", 0))
        skipped_frames = int(stream_state.get("skipped_frames", 0))
        last_time = float(stream_state.get("last_time", 0.0))
        last_summary = stream_state.get("last_summary")
        last_latency_ms = float(stream_state.get("last_latency_ms", 0.0))
        last_detections = stream_state.get("last_detections", [])
        last_detection_shape = stream_state.get("last_detection_shape")
        stream_state["frame_index"] = frame_index

        frame_stride = max(1, int(frame_stride or 1))
        base_interval = float(min_frame_interval or 0.0)
        adaptive_interval = min(0.75, (last_latency_ms / 1000.0) * 0.60) if auto_throttle else 0.0
        effective_interval = max(base_interval, adaptive_interval)
        should_skip_stride = last_summary is not None and (frame_index - 1) % frame_stride != 0
        should_skip_time = (
            effective_interval
            and last_summary is not None
            and now - last_time < effective_interval
        )
        if (
            should_skip_stride
            or should_skip_time
        ):
            skipped_frames += 1
            stream_state["skipped_frames"] = skipped_frames
            if smooth_preview and task == "detect":
                preview_frame = self.resize_for_stream(frame, stream_max_side or 0)
                preview_shape = preview_frame.shape[:2]
                preview_detections = self.scale_stream_detections(
                    last_detections,
                    last_detection_shape,
                    preview_shape,
                )
                preview_img = self.draw_stream_detections(
                    preview_frame,
                    preview_detections,
                    line_width,
                    "hide_labels" in set(checkboxes or []),
                    "hide_conf" in set(checkboxes or []),
                )
                return preview_img, last_summary, stream_state
            return gr.update(), last_summary, stream_state

        process_start = time.monotonic()
        out_img, summary, detections, detection_shape, infer_time = self.run_stream_inference(
            task,
            frame,
            model_dropdown,
            custom_model_path,
            conf,
            iou,
            device,
            line_width,
            cpu,
            checkboxes,
            stream_max_side,
            stream_max_det,
            last_detections,
            last_detection_shape,
            smooth_boxes,
        )
        process_end = time.monotonic()
        if out_img is not None:
            processed_frames += 1
            elapsed_since_last = process_end - last_time if last_time else 0.0
            fps = 1.0 / elapsed_since_last if elapsed_since_last > 0 else 0.0
            latency_ms = (process_end - process_start) * 1000.0
            previous_ema = float(stream_state.get("ema_fps", 0.0))
            ema_fps = fps if previous_ema <= 0 else (previous_ema * 0.75 + fps * 0.25)
            summary = (
                f"{summary}\n"
                f"- **Stream FPS:** `{ema_fps:.1f}`\n"
                f"- **End-to-end:** `{latency_ms:.1f}ms`\n"
                f"- **Throttle:** `{effective_interval:.2f}s`"
                f" / **Model:** `{infer_time:.1f}ms`\n"
                f"- **Frame Size:** `{int(stream_max_side)}px max side`\n"
                f"- **Processed / Skipped:** `{processed_frames}` / `{skipped_frames}`"
            )
            stream_state = {
                "stream_key": stream_key,
                "frame_index": frame_index,
                "processed_frames": processed_frames,
                "skipped_frames": skipped_frames,
                "last_time": process_end,
                "last_summary": summary,
                "ema_fps": ema_fps,
                "last_latency_ms": latency_ms,
                "last_detections": detections,
                "last_detection_shape": detection_shape,
            }
        return out_img, summary, stream_state

    def describe_model(self, task: str, model_path: str) -> str:
        """Validate and describe the model."""
        if not model_path or not model_path.strip():
            return "⚠️ Please enter a model path."
        
        path = Path(model_path.strip())
        if not path.exists():
            return f"❌ Path does not exist: `{model_path}`"
            
        try:
            # Check if it's a directory, try to find pt file
            if path.is_dir():
                candidates = [
                    path / "weights" / "best.pt",
                    path / "weights" / "last.pt",
                    path / "best.pt",
                    path / "last.pt",
                ]
                found = False
                for c in candidates:
                    if c.exists():
                        path = c
                        found = True
                        break
                if not found:
                    return f"❌ No model file (.pt) found in directory: `{model_path}`"
            
            # Load model to get info (temporary load, no caching here to avoid polluting main state)
            model = YOLO(str(path))
            names = model.names
            nc = len(names)
            model_task = model.task
            
            return (
                f"### ✅ Model Validated\n"
                f"- **Path:** `{path}`\n"
                f"- **Task:** `{model_task}` (Expected: `{task}`)\n"
                f"- **Classes:** {nc}\n"
                f"- **Names:** {list(names.values())[:5]}..."
            )
        except Exception as e:
            return f"❌ Invalid Model: {str(e)}"

    def update_model_dropdown(self, task: str):
        """UI Event: Update model list when task changes."""
        choices = self.model_map.get(task, [])
        if not choices:
            choices = [GlobalConfig.DEFAULT_MODELS.get(task, "yolov8n.pt")]
        return gr.update(choices=choices, value=choices[0])

    def refresh_models(self, task: str):
        """UI Event: Manually refresh model list."""
        self.model_map = self.model_manager.scan_checkpoints()
        return self.update_model_dropdown(task)

    @staticmethod
    def reset_stream_state():
        """Clear cached stream detections after switching cameras or models."""
        return None, "Waiting for webcam stream...", {}

    def launch(self):
        with gr.Blocks(title="YOLO-Master WebUI", theme=GlobalConfig.THEME) as app:
            gr.HTML(self.brand_header())
            
            with gr.Row(equal_height=False):
                # ================= Sidebar: Control Panel =================
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### 🛠 Settings")
                    
                    # Task and Model Selection
                    with gr.Group():
                        task_radio = gr.Radio(
                            choices=["detect", "seg", "cls", "pose", "obb"], 
                            value="detect", 
                            label="Task"
                        )
                        with gr.Row():
                            model_dd = gr.Dropdown(
                                choices=self.model_map["detect"], 
                                value=self.model_map["detect"][0] if self.model_map["detect"] else None, 
                                label="Model Weights", 
                                scale=5,
                                interactive=True
                            )
                            refresh_btn = gr.Button("🔄", scale=1, min_width=10, size="sm")
                        custom_model_txt = gr.Textbox(
                            value="",
                            label="Custom Model Path (file or directory)",
                            placeholder="./ckpts/yolo_master_n.pt",
                            interactive=True
                        )
                        validate_btn = gr.Button("✅ Validate Path", size="sm")

                    # Advanced Parameters
                    with gr.Accordion("⚙️ Advanced Parameters", open=True):
                        conf_slider = gr.Slider(0, 1, 0.25, step=0.01, label="Confidence (Conf)")
                        iou_slider = gr.Slider(0, 1, 0.7, step=0.01, label="IoU Threshold")
                        
                        with gr.Row():
                            max_det_num = gr.Number(300, label="Max Objects", precision=0)
                            line_width_num = gr.Number(0, label="Line Width", precision=0)
                        
                        with gr.Row():
                            device_txt = gr.Textbox("cpu", label="Device ID (e.g. 0, cpu)", placeholder="0 or cpu")
                            cpu_chk = gr.Checkbox(True, label="Force CPU")

                    with gr.Accordion("🎞️ Live Stream Performance", open=False):
                        stream_preset = gr.Radio(
                            choices=list(GlobalConfig.STREAM_PRESETS.keys()),
                            value="Balanced",
                            label="Stream Preset"
                        )
                        stream_max_side = gr.Slider(320, 960, 640, step=32, label="Stream Max Side (px)")
                        min_frame_interval = gr.Slider(0, 1, 0.15, step=0.05, label="Min Frame Interval (s)")
                        stream_frame_stride = gr.Slider(1, 6, 2, step=1, label="Process Every Nth Frame")
                        stream_max_det = gr.Slider(20, 300, 120, step=10, label="Stream Max Objects")
                        with gr.Row():
                            smooth_preview_chk = gr.Checkbox(True, label="Smooth Preview")
                            smooth_boxes_chk = gr.Checkbox(True, label="Stable Boxes")
                        auto_throttle_chk = gr.Checkbox(True, label="Auto Throttle")

                    # Output Options
                    options_chk = gr.CheckboxGroup(
                        ["half", "show", "save", "save_txt", "save_crop", "hide_labels", "hide_conf", "agnostic_nms", "retina_masks"],
                        label="Output Options",
                        value=[]
                    )

                    with gr.Accordion("🧠 Agent Report API", open=False):
                        agent_api_provider = gr.Radio(
                            choices=list(GlobalConfig.AGENT_API_PROVIDERS.keys()),
                            value="OpenAI",
                            label="Provider"
                        )
                        agent_api_base = gr.Textbox(
                            value=os.environ.get("OPENAI_BASE_URL", GlobalConfig.AGENT_API_PROVIDERS["OpenAI"]["base_url"]),
                            label="API Base URL",
                            interactive=True
                        )
                        agent_api_mode = gr.Radio(
                            choices=["auto", "responses", "chat.completions"],
                            value=os.environ.get("OPENAI_API_MODE", "auto"),
                            label="API Mode"
                        )
                        agent_api_key = gr.Textbox(
                            value="",
                            label="API Key",
                            type="password",
                            interactive=True
                        )
                        with gr.Row():
                            agent_vlm_model = gr.Textbox(
                                value=os.environ.get("OPENAI_VLM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")),
                                label="VLM Model",
                                interactive=True
                            )
                            agent_llm_model = gr.Textbox(
                                value=os.environ.get("OPENAI_LLM_MODEL", ""),
                                label="LLM Refine Model",
                                interactive=True
                            )
                        agent_prompt_template = gr.Dropdown(
                            choices=GlobalConfig.AGENT_PROMPT_TEMPLATES,
                            value="vlm_coco_multitask",
                            label="Prompt Template"
                        )
                        agent_report_prompt = gr.Textbox(
                            value="请交叉验证 YOLO-Master 的检测/分割结果，输出中文视觉推理报告、风险点、可能漏检和下一步建议。",
                            label="Report Prompt",
                            lines=3,
                            interactive=True
                        )
                        with gr.Row():
                            agent_structured_output = gr.Checkbox(True, label="Structured Output")
                            agent_thinking_image = gr.Checkbox(True, label="Thinking With Image")
                        with gr.Row():
                            agent_use_marked_image = gr.Checkbox(True, label="Use Marked Image")
                            agent_llm_refine = gr.Checkbox(False, label="LLM Refine")
                        with gr.Row():
                            agent_max_tokens = gr.Slider(512, 6000, 3500, step=128, label="Max Output Tokens")
                            agent_temperature = gr.Slider(0, 1, 0.2, step=0.05, label="Temperature")
                    
                    # Run Button
                    run_btn = gr.Button("🔥 Start Inference", variant="primary", size="lg")

                # ================= Main Area: Display Panel =================
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("🖼️ Visualization"):
                            with gr.Tabs():
                                with gr.TabItem("Image"):
                                    with gr.Row():
                                        inp_img = gr.Image(
                                            type="numpy",
                                            label="Input Image",
                                            height=500,
                                            value=self.load_default_image()
                                        )
                                        out_img = gr.Image(
                                            type="numpy",
                                            label="Inference Result",
                                            height=500,
                                            interactive=False
                                        )
                                    info_md = gr.Markdown(value="Waiting for input...")

                                with gr.TabItem("Live Webcam"):
                                    with gr.Row():
                                        webcam_img = gr.Image(
                                            sources=["webcam"],
                                            streaming=True,
                                            type="numpy",
                                            label="Webcam Stream",
                                            height=500,
                                            mirror_webcam=True,
                                        )
                                        webcam_out_img = gr.Image(
                                            type="numpy",
                                            label="Live Inference Result",
                                            height=500,
                                            interactive=False
                                        )
                                    reset_stream_btn = gr.Button("Reset Stream", size="sm")
                                    webcam_info_md = gr.Markdown(value="Waiting for webcam stream...")
                                    stream_state = gr.State({})

                        with gr.TabItem("📊 Data Analysis"):
                            gr.Markdown("### Detections Data")
                            out_df = gr.Dataframe(
                                headers=["Class ID", "Class Name", "Confidence", "x1", "y1", "x2", "y2"],
                                label="Raw Detections"
                            )

                        with gr.TabItem("🧠 Agent Report"):
                            agent_report_btn = gr.Button("Generate Visual Report", variant="primary")
                            with gr.Row():
                                agent_report_img = gr.Image(
                                    type="numpy",
                                    label="Marked Evidence Image",
                                    height=420,
                                    interactive=False
                                )
                                agent_report_json = gr.JSON(
                                    label="Agent Evidence JSON",
                                    value={}
                                )
                            agent_report_md = gr.Markdown(value="Waiting for agent report...")

            # ================= Event Binding =================
            
            # 1. Auto-refresh model list
            task_radio.change(fn=self.update_model_dropdown, inputs=task_radio, outputs=model_dd)
            refresh_btn.click(fn=self.refresh_models, inputs=task_radio, outputs=model_dd)
            validate_btn.click(fn=self.describe_model, inputs=[task_radio, custom_model_txt], outputs=info_md)
            agent_api_provider.change(
                fn=self.update_agent_provider,
                inputs=agent_api_provider,
                outputs=[agent_api_base, agent_api_mode, agent_vlm_model, agent_llm_model],
                show_api=False
            )
            reset_stream_btn.click(
                fn=self.reset_stream_state,
                inputs=[],
                outputs=[webcam_out_img, webcam_info_md, stream_state],
                show_api=False
            )
            stream_preset.change(
                fn=self.update_stream_preset,
                inputs=stream_preset,
                outputs=[stream_max_side, min_frame_interval, stream_frame_stride, stream_max_det],
                show_api=False
            )
            
            # 2. Inference Logic
            run_btn.click(
                fn=self.inference,
                inputs=[
                    task_radio, inp_img, model_dd, custom_model_txt,
                    conf_slider, iou_slider, device_txt, 
                    max_det_num, line_width_num, cpu_chk, options_chk
                ],
                outputs=[out_img, out_df, info_md],
                concurrency_limit=1,
                concurrency_id="model-inference",
                show_api=False
            )
            agent_report_btn.click(
                fn=self.generate_agent_report,
                inputs=[
                    task_radio, inp_img, model_dd, custom_model_txt,
                    conf_slider, iou_slider, device_txt,
                    max_det_num, line_width_num, cpu_chk, options_chk,
                    agent_api_provider, agent_api_base, agent_api_mode, agent_api_key,
                    agent_vlm_model, agent_llm_model, agent_prompt_template, agent_report_prompt,
                    agent_structured_output, agent_thinking_image, agent_use_marked_image,
                    agent_llm_refine, agent_max_tokens, agent_temperature
                ],
                outputs=[agent_report_img, agent_report_md, agent_report_json, out_img, out_df, info_md],
                concurrency_limit=1,
                concurrency_id="model-inference",
                show_api=False
            )
            webcam_img.stream(
                fn=self.inference_stream,
                inputs=[
                    task_radio, webcam_img, model_dd, custom_model_txt,
                    conf_slider, iou_slider, device_txt,
                    line_width_num, cpu_chk, options_chk,
                    stream_max_side, min_frame_interval, stream_frame_stride, stream_max_det,
                    smooth_preview_chk, smooth_boxes_chk, auto_throttle_chk, stream_state
                ],
                outputs=[webcam_out_img, webcam_info_md, stream_state],
                show_progress="hidden",
                trigger_mode="always_last",
                concurrency_limit=1,
                concurrency_id="model-inference",
                show_api=False
            )

        app.launch(share=True)


if __name__ == "__main__":
    # Configure your checkpoints path
    CKPTS_DIR = Path(__file__).parent / "ckpts"
    
    # Create default dir if not exists
    if not CKPTS_DIR.exists():
        CKPTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created default checkpoints dir: {CKPTS_DIR}")
    
    print(f"Starting YOLO-Master WebUI...")
    print(f"Scanning models in: {CKPTS_DIR}")
    
    ui = YOLO_Master_WebUI(str(CKPTS_DIR))
    ui.launch()
