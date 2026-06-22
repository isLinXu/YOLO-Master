"""YOLO-Master WebUI — 智能 Agent 增强版

核心特性:
- 🤖 YOLOMasterAgent: 具备工具调用、记忆系统、决策引擎的嵌入式 Agent
- 🧠 智能参数推荐: Agent 分析图像特征后自动推荐 conf/iou/max_det
- 💡 结果智能分析: 检测漏检/过检、置信度分布异常、场景密度评估
- 🔁 自适应重试: 结果不满意时 Agent 自动调整策略并重新推理
- 💬 自然语言交互: Chatbot 面板，支持"检测这张图中的行人并告诉我数量"
- 📝 偏好记忆: Agent 记住用户常用配置和任务习惯
- 📦 保留所有原有功能: Single/Batch/Video/Webcam/History/Logs/About
"""
import os
import gc
import warnings
import time
import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Callable
from collections import deque
from dataclasses import dataclass, asdict, field
from enum import Enum
import datetime

import gradio as gr
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import urllib.error
import urllib.request
from collections import Counter

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

warnings.filterwarnings("ignore")

# ============================================================
# 1. 配置与常量
# ============================================================

@dataclass
class UIConfig:

    # ── Streaming Presets ──
    STREAM_DISABLED_OPTIONS = {"half", "show", "save", "save_txt", "save_crop"}
    STREAM_PRESETS = {
        "Realtime": {"max_side": 416, "interval": 0.06, "stride": 2, "max_det": 60},
        "Fast": {"max_side": 480, "interval": 0.10, "stride": 3, "max_det": 80},
        "Balanced": {"max_side": 640, "interval": 0.15, "stride": 2, "max_det": 120},
        "Quality": {"max_side": 832, "interval": 0.20, "stride": 1, "max_det": 200},
    }
    STREAM_BOX_COLORS = (
        (56, 189, 248), (52, 211, 153), (251, 191, 36), (248, 113, 113),
        (167, 139, 250), (244, 114, 182), (45, 212, 191), (250, 204, 21),
    )
    PROJECT_URL = "https://github.com/isLinXu/YOLO-Master"
    MASCOT_IMAGE_URL = "https://github.com/user-attachments/assets/bbf751ea-af27-465d-a8a9-7822db343638"

    DEFAULT_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "detect": "ckpts/yolo-master-v0.1-n.pt",
        "seg": "ckpts/yolo-master-seg-n.pt",
        "cls": "ckpts/yolo-master-cls-n.pt",
        "pose": "yolov8n-pose.pt",
        "obb": "yolov8n-obb.pt",
    })
    IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    DEFAULT_IMAGE_DIR: str = "./image"
    HISTORY_DIR: str = "./history"
    AGENT_MEMORY_DIR: str = "./agent_memory"
    MAX_HISTORY: int = 50
    MAX_LOG_LINES: int = 200
    PLOT_DPI: int = 120
    PLOT_FIGSIZE: Tuple[int, int] = (6, 4)
    AGENT_MAX_CONV: int = 20

CONFIG = UIConfig()


# ============================================================
# 2. 日志系统
# ============================================================

class Logger:
    def __init__(self, max_lines: int = CONFIG.MAX_LOG_LINES):
        self._logs: deque = deque(maxlen=max_lines)
        self._callbacks: List[Callable] = []

    def log(self, level: str, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] [{level}] {msg}"
        self._logs.append(line)
        for cb in self._callbacks:
            try:
                cb(line)
            except Exception:
                pass

    def info(self, msg: str): self.log("INFO", msg)
    def warn(self, msg: str): self.log("WARN", msg)
    def error(self, msg: str): self.log("ERROR", msg)
    def success(self, msg: str): self.log("OK", msg)
    def get_text(self) -> str: return "\n".join(self._logs)
    def clear(self): self._logs.clear()

LOGGER = Logger()


# ============================================================
# 3. 模型管理器
# ============================================================

class ModelManager:
    def __init__(self, ckpts_root: Path):
        self.ckpts_root = Path(ckpts_root)
        self.current_model: Optional[YOLO] = None
        self.current_model_path: str = ""
        self.current_task: str = "detect"
        self._cache: Dict[str, Dict] = {}

    def scan(self) -> Dict[str, List[str]]:
        m = {k: [] for k in CONFIG.DEFAULT_MODELS.keys()}
        if not self.ckpts_root.exists():
            return m
        for p in self.ckpts_root.rglob("*.pt"):
            if p.is_dir(): continue
            s = str(p.absolute())
            f, pa = p.name.lower(), p.parent.name.lower()
            if "seg" in f or "seg" in pa: m["seg"].append(s)
            elif "cls" in f or "class" in f or "cls" in pa: m["cls"].append(s)
            elif "pose" in f or "pose" in pa: m["pose"].append(s)
            elif "obb" in f or "obb" in pa: m["obb"].append(s)
            else: m["detect"].append(s)
        for k in m: m[k] = sorted(list(set(m[k])))
        return m

    def unload(self):
        if self.current_model:
            del self.current_model
            self.current_model = None
            self.current_model_path = ""
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            LOGGER.info("GPU memory cleared.")

    def load(self, model_path: str, task: str) -> YOLO:
        target = self._resolve(model_path, task)
        if self.current_model and self.current_model_path == target:
            LOGGER.info(f"Cache hit: {Path(target).name}")
            return self.current_model
        self.unload()
        LOGGER.info(f"Loading: {target} ...")
        t0 = time.time()
        model = YOLO(target)
        self.current_model = model
        self.current_model_path = target
        self.current_task = task
        LOGGER.success(f"Loaded in {(time.time()-t0)*1000:.1f}ms | Task: {task}")
        return model

    def _resolve(self, model_path: str, task: str) -> str:
        if model_path and os.path.exists(model_path):
            if os.path.isdir(model_path):
                for c in [Path(model_path)/"weights"/"best.pt", Path(model_path)/"weights"/"last.pt",
                          Path(model_path)/"best.pt", Path(model_path)/"last.pt"]:
                    if c.exists(): return str(c)
            return model_path
        fb = CONFIG.DEFAULT_MODELS.get(task, "yolov8n.pt")
        if os.path.exists(fb): return fb
        repo = os.environ.get("YOLO_MASTER_WEIGHTS_REPO", "")
        if hf_hub_download and repo:
            try:
                d = Path(__file__).parent / "ckpts"
                d.mkdir(parents=True, exist_ok=True)
                return hf_hub_download(repo_id=repo, filename=Path(fb).name, repo_type="model", local_dir=str(d))
            except Exception as e:
                LOGGER.warn(f"HF fail: {e}")
        return {"detect": "yolov8n.pt", "seg": "yolov8n-seg.pt", "cls": "yolov8n-cls.pt"}.get(task, "yolov8n.pt")

    def get_device(self) -> str:
        try:
            if self.current_model and hasattr(self.current_model, "model"):
                return str(next(self.current_model.model.parameters()).device)
        except Exception: pass
        return "cpu"

    def get_info(self) -> Dict[str, Any]:
        if not self.current_model or not hasattr(self.current_model, "model"):
            return {}
        p = self.current_model_path
        if p in self._cache: return self._cache[p]
        try:
            m = self.current_model.model
            tp = sum(p_.numel() for p_ in m.parameters())
            tr = sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)
            info = {
                "name": Path(p).name, "task": getattr(self.current_model, "task", "unknown"),
                "nc": getattr(m, "nc", "unknown"),
                "names": list(getattr(self.current_model, "names", {}).values())[:10],
                "total_params": f"{tp/1e6:.2f}M", "trainable_params": f"{tr/1e6:.2f}M",
                "device": self.get_device(), "path": p,
            }
            self._cache[p] = info
            return info
        except Exception as e:
            LOGGER.error(f"get_info error: {e}")
            return {}


# ============================================================
# 4. 推理历史
# ============================================================

class InferenceHistory:
    def __init__(self, max_size: int = CONFIG.MAX_HISTORY):
        self.max_size = max_size
        self._records: deque = deque(maxlen=max_size)
        self._dir = Path(CONFIG.HISTORY_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        f = self._dir / "summary.json"
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    for item in data[-self.max_size:]:
                        self._records.append(item)
            except Exception as e:
                LOGGER.warn(f"History load fail: {e}")

    def _save(self):
        try:
            with open(self._dir / "summary.json", "w", encoding="utf-8") as fh:
                json.dump(list(self._records), fh, ensure_ascii=False, indent=2)
        except Exception as e:
            LOGGER.warn(f"History save fail: {e}")

    def add(self, image: np.ndarray, df: pd.DataFrame, summary_md: str, task: str, model_name: str, infer_time: float, num_objects: int) -> str:
        rid = str(uuid.uuid4())[:8]
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip = self._dir / f"{rid}.jpg"
        try:
            cv2.imwrite(str(ip), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        except Exception:
            ip = None
        cp = self._dir / f"{rid}.csv"
        try:
            df.to_csv(str(cp), index=False)
        except Exception:
            cp = None
        rec = {"id": rid, "timestamp": ts, "task": task, "model": model_name,
               "infer_time_ms": round(infer_time, 2), "num_objects": num_objects,
               "image_path": str(ip) if ip else "", "csv_path": str(cp) if cp else "",
               "summary": summary_md}
        self._records.append(rec)
        self._save()
        return rid

    def get_all(self) -> List[Dict]: return list(self._records)
    def get_latest(self, n: int = 1) -> List[Dict]: return list(self._records)[-n:]
    def clear(self):
        self._records.clear()
        self._save()
        for f in self._dir.glob("*.jpg"): f.unlink(missing_ok=True)
        for f in self._dir.glob("*.csv"): f.unlink(missing_ok=True)


# ============================================================
# 5. 图表生成器
# ============================================================

class StatsPlotter:
    @staticmethod
    def _make_fig() -> Tuple[plt.Figure, plt.Axes]:
        return plt.subplots(figsize=CONFIG.PLOT_FIGSIZE, dpi=CONFIG.PLOT_DPI)

    @staticmethod
    def _to_pil(fig: plt.Figure) -> Image.Image:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return Image.fromarray(buf)

    @classmethod
    def class_dist(cls, df: pd.DataFrame) -> Optional[Image.Image]:
        if df.empty or "Class Name" not in df.columns: return None
        try:
            c = df["Class Name"].value_counts()
            if len(c) == 0: return None
            fig, ax = cls._make_fig()
            colors = plt.cm.tab10(np.linspace(0, 1, len(c)))
            bars = ax.bar(range(len(c)), c.values, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(c)))
            ax.set_xticklabels(c.index, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title('Class Distribution', fontsize=12, fontweight='bold', color='white')
            ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom', 'left']: ax.spines[sp].set_color('white')
            for bar, val in zip(bars, c.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(val),
                        ha='center', va='bottom', fontsize=8, color='white')
            plt.tight_layout()
            return cls._to_pil(fig)
        except Exception as e:
            LOGGER.error(f"class_dist plot: {e}"); return None

    @classmethod
    def conf_hist(cls, df: pd.DataFrame) -> Optional[Image.Image]:
        if df.empty or "Confidence" not in df.columns: return None
        try:
            confs = df["Confidence"].values
            fig, ax = cls._make_fig()
            ax.hist(confs, bins=20, color='#4ecca3', edgecolor='white', alpha=0.85)
            ax.axvline(confs.mean(), color='#ff6b6b', linestyle='--', linewidth=2,
                       label=f'Mean: {confs.mean():.3f}')
            ax.set_xlabel('Confidence', fontsize=10); ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold', color='white')
            ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
            ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom', 'left']: ax.spines[sp].set_color('white')
            plt.tight_layout()
            return cls._to_pil(fig)
        except Exception as e:
            LOGGER.error(f"conf_hist plot: {e}"); return None

    @classmethod
    def timing_trend(cls, history: List[Dict]) -> Optional[Image.Image]:
        if len(history) < 2: return None
        try:
            times = [r["infer_time_ms"] for r in history]
            labels = [f"#{i+1}" for i in range(len(times))]
            fig, ax = cls._make_fig()
            ax.plot(labels, times, marker='o', color='#4ecca3', linewidth=2, markersize=6)
            ax.fill_between(labels, times, alpha=0.2, color='#4ecca3')
            ax.set_xlabel('Inference #', fontsize=10); ax.set_ylabel('Time (ms)', fontsize=10)
            ax.set_title('Latency Trend', fontsize=12, fontweight='bold', color='white')
            ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom', 'left']: ax.spines[sp].set_color('white')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return cls._to_pil(fig)
        except Exception as e:
            LOGGER.error(f"timing_trend plot: {e}"); return None


# ============================================================
# 6. 系统状态
# ============================================================

def get_system_status() -> str:
    lines = [f"**PyTorch:** `{torch.__version__}`"]
    if torch.cuda.is_available():
        lines.append(f"**CUDA:** `{torch.version.cuda}` | **GPU:** `{torch.cuda.get_device_name(0)}`")
        alloc = torch.cuda.memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        lines.append(f"**GPU Mem:** {alloc:.0f}MB / {total:.0f}MB")
    else:
        lines.append("**CUDA:** Not available")
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        lines.append(f"**CPU:** {cpu:.1f}% | **RAM:** {ram.used/1024**3:.1f}GB / {ram.total/1024**3:.1f}GB ({ram.percent}%)")
    except Exception: pass
    return "\n".join(lines)


# ============================================================
# 7. 🤖 YOLOMasterAgent — 核心 Agent 系统
# ============================================================

class AgentAction(Enum):
    INFER = "infer"
    ADJUST_CONF = "adjust_conf"
    ADJUST_IOU = "adjust_iou"
    SWITCH_MODEL = "switch_model"
    ANALYZE = "analyze"
    AUTO_ENHANCE = "auto_enhance"
    EXPLAIN = "explain"
    RECOMMEND = "recommend"
    NOOP = "noop"


@dataclass
class AgentMemory:
    """Agent 持久化记忆"""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    task_history: List[Dict] = field(default_factory=list)
    conversation: deque = field(default_factory=lambda: deque(maxlen=CONFIG.AGENT_MAX_CONV))
    last_params: Dict[str, Any] = field(default_factory=dict)
    last_result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "user_preferences": self.user_preferences,
            "task_history": list(self.task_history[-20:]),
            "conversation": list(self.conversation),
            "last_params": self.last_params,
        }

    @classmethod
    def from_dict(cls, d: Dict):
        m = cls()
        m.user_preferences = d.get("user_preferences", {})
        m.task_history = d.get("task_history", [])
        m.conversation = deque(d.get("conversation", []), maxlen=CONFIG.AGENT_MAX_CONV)
        m.last_params = d.get("last_params", {})
        return m


class YOLOMasterAgent:
    """
    YOLOMaster 智能 Agent.

    能力:
    1. 图像特征分析 -> 智能推荐参数
    2. 推理结果分析 -> 质量评估 + 改进建议
    3. 自然语言理解 -> 指令解析 -> 工具调用
    4. 自适应重试 -> 自动调优策略
    5. 偏好记忆 -> 记住用户习惯
    """

    def __init__(self, model_manager: ModelManager, history: InferenceHistory):
        self.model_manager = model_manager
        self.inference_history = history
        self.memory = AgentMemory()
        self._load_memory()
        self.tools: Dict[str, Callable] = {
            "infer": self._tool_infer,
            "adjust_conf": self._tool_adjust_conf,
            "adjust_iou": self._tool_adjust_iou,
            "switch_model": self._tool_switch_model,
            "analyze": self._tool_analyze,
            "explain": self._tool_explain,
            "recommend": self._tool_recommend,
        }
        LOGGER.info("Agent initialized with memory.")

    # ---------- 记忆管理 ----------
    def _load_memory(self):
        mdir = Path(CONFIG.AGENT_MEMORY_DIR)
        mdir.mkdir(parents=True, exist_ok=True)
        mf = mdir / "agent_memory.json"
        if mf.exists():
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    self.memory = AgentMemory.from_dict(json.load(f))
                LOGGER.info("Agent memory loaded.")
            except Exception as e:
                LOGGER.warn(f"Memory load failed: {e}")

    def _save_memory(self):
        try:
            mdir = Path(CONFIG.AGENT_MEMORY_DIR)
            mdir.mkdir(parents=True, exist_ok=True)
            with open(mdir / "agent_memory.json", "w", encoding="utf-8") as f:
                json.dump(self.memory.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            LOGGER.warn(f"Memory save failed: {e}")

    def remember_preference(self, key: str, value: Any):
        self.memory.user_preferences[key] = value
        self._save_memory()

    def get_preference(self, key: str, default=None):
        return self.memory.user_preferences.get(key, default)

    def add_to_conversation(self, role: str, content: str):
        self.memory.conversation.append({"role": role, "content": content, "time": datetime.datetime.now().isoformat()})
        self._save_memory()

    # ---------- 视觉分析工具 ----------
    def analyze_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像特征，用于智能推荐参数.
        不依赖深度学习模型，纯 CV 特征分析.
        """
        if image is None or image.size == 0:
            return {"error": "No image"}
        h, w = image.shape[:2]
        # 色彩分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image
        else:
            gray = image
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        # 边缘密度 (潜在目标密度指标)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / (h * w))
        # 纹理复杂度 (Laplacian 方差)
        texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # 场景类型启发
        scene_type = "standard"
        if edge_density > 0.15:
            scene_type = "dense"
        elif edge_density < 0.03:
            scene_type = "sparse"
        if brightness < 50:
            scene_type = "dark" if scene_type == "standard" else f"{scene_type}_dark"
        if texture > 1000:
            scene_type = "complex" if scene_type == "standard" else f"{scene_type}_complex"

        return {
            "resolution": (w, h),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "edge_density": round(edge_density, 4),
            "texture": round(texture, 1),
            "scene_type": scene_type,
        }

    def suggest_params(self, image: np.ndarray, task: str = "detect") -> Dict[str, Any]:
        """基于图像特征智能推荐推理参数"""
        features = self.analyze_image_features(image)
        # 默认基线
        conf = 0.25
        iou = 0.7
        max_det = 300
        line_width = 0  # auto
        advice = []

        scene = features.get("scene_type", "standard")
        if "dense" in scene:
            conf = 0.15  # 密集场景降低 conf，避免漏检
            iou = 0.5    # 降低 IoU 减少 NMS 误杀
            max_det = 500
            advice.append("Scene appears dense with many edges. Lowered conf to 0.15 and IoU to 0.5 to catch more objects.")
        elif "sparse" in scene:
            conf = 0.35  # 稀疏场景提高 conf，减少误报
            iou = 0.7
            max_det = 100
            advice.append("Scene appears sparse. Raised conf to 0.35 to reduce false positives.")
        if "dark" in scene:
            conf = max(0.15, conf - 0.05)
            advice.append("Low brightness detected. Lowered conf slightly for dark regions.")
        if "complex" in scene:
            iou = max(0.45, iou - 0.1)
            advice.append("High texture complexity. Reduced IoU to avoid overlapping detections.")

        # 用户偏好覆盖
        if self.get_preference("default_conf") is not None:
            conf = self.get_preference("default_conf")
        if self.get_preference("default_iou") is not None:
            iou = self.get_preference("default_iou")

        return {
            "conf": round(conf, 2),
            "iou": round(iou, 2),
            "max_det": max_det,
            "line_width": line_width,
            "features": features,
            "advice": advice,
        }

    # ---------- 结果分析工具 ----------
    def analyze_result(self, df: pd.DataFrame, task: str, infer_time: float) -> Dict[str, Any]:
        """分析推理结果质量"""
        if df.empty:
            return {
                "quality": "empty",
                "issues": ["No detections found."],
                "suggestions": ["Try lowering confidence threshold.", "Check if model matches task."],
                "stats": {}
            }
        num = len(df)
        confs = df["Confidence"].values if "Confidence" in df.columns else np.array([])
        avg_conf = float(np.mean(confs)) if len(confs) > 0 else 0
        min_conf = float(np.min(confs)) if len(confs) > 0 else 0
        issues = []
        suggestions = []
        quality = "good"

        if num == 0:
            quality = "empty"
            issues.append("Zero detections.")
        elif avg_conf < 0.3:
            quality = "poor"
            issues.append(f"Low average confidence ({avg_conf:.2f}). Model may be uncertain.")
            suggestions.append("Consider using a larger model or training with more data.")
        elif num > 100:
            quality = "dense"
            issues.append(f"Very high object count ({num}). Possible over-detection or dense scene.")
            suggestions.append("Consider raising conf threshold or lowering max_det.")
        if min_conf < 0.1 and num > 0:
            issues.append(f"Some detections have very low confidence (min {min_conf:.2f}).")
            suggestions.append("Filter low-confidence boxes or use NMS more aggressively.")
        if infer_time > 100:
            issues.append(f"High inference time ({infer_time:.1f}ms).")
            suggestions.append("Consider using a smaller model, half precision, or GPU.")

        if not issues:
            suggestions.append("Result looks good. You can export or proceed with analysis.")

        # 类别分布分析
        class_balance = ""
        if "Class Name" in df.columns and num > 0:
            counts = df["Class Name"].value_counts()
            if len(counts) > 1:
                ratio = counts.iloc[0] / counts.iloc[-1] if counts.iloc[-1] > 0 else 999
                if ratio > 10:
                    class_balance = f"Class imbalance detected: dominant class is {counts.index[0]} ({counts.iloc[0]} vs {counts.iloc[-1]})."

        stats = {
            "total_objects": num,
            "avg_confidence": round(avg_conf, 3),
            "min_confidence": round(min_conf, 3),
            "infer_time_ms": round(infer_time, 2),
        }

        return {
            "quality": quality,
            "issues": issues,
            "suggestions": suggestions,
            "class_balance": class_balance,
            "stats": stats,
        }

    # ---------- 自适应重试 ----------
    def adaptive_strategy(self, last_analysis: Dict) -> Optional[Dict[str, Any]]:
        """基于上次结果分析，生成自动优化策略"""
        if not last_analysis or "quality" not in last_analysis:
            return None
        quality = last_analysis["quality"]
        strategy = {"action": AgentAction.NOOP, "changes": {}, "reason": ""}

        if quality == "empty":
            strategy = {
                "action": AgentAction.ADJUST_CONF,
                "changes": {"conf": 0.1, "iou": 0.6},
                "reason": "Last inference found zero objects. Lowering confidence to catch more candidates.",
            }
        elif quality == "poor":
            strategy = {
                "action": AgentAction.AUTO_ENHANCE,
                "changes": {"conf": 0.2, "iou": 0.5, "half": True},
                "reason": "Low confidence detected. Trying lower conf + stricter NMS + half precision for speed.",
            }
        elif quality == "dense":
            strategy = {
                "action": AgentAction.ADJUST_CONF,
                "changes": {"conf": 0.4, "max_det": 300},
                "reason": "Too many detections. Raising conf and limiting max_det to reduce clutter.",
            }
        else:
            return None
        return strategy

    # ---------- 自然语言解析 ----------
    def parse_natural_language(self, text: str) -> Dict[str, Any]:
        """
        解析用户自然语言指令，提取意图和参数.
        这是一个轻量级规则引擎，无需 LLM API.
        """
        text_lower = text.lower()
        intent = {"action": None, "params": {}, "response": ""}

        # 意图识别
        if any(k in text_lower for k in ["detect", "find", "识别", "检测", "找"]):
            intent["action"] = AgentAction.INFER
            intent["params"]["task"] = "detect"
        elif any(k in text_lower for k in ["segment", "分割", "抠图"]):
            intent["action"] = AgentAction.INFER
            intent["params"]["task"] = "seg"
        elif any(k in text_lower for k in ["classify", "分类", "识别类别"]):
            intent["action"] = AgentAction.INFER
            intent["params"]["task"] = "cls"
        elif any(k in text_lower for k in ["analyze", "分析", "result", "结果"]):
            intent["action"] = AgentAction.ANALYZE
        elif any(k in text_lower for k in ["enhance", "优化", "retry", "重试", "improve", "改进"]):
            intent["action"] = AgentAction.AUTO_ENHANCE
        elif any(k in text_lower for k in ["recommend", "建议", "推荐", "参数"]):
            intent["action"] = AgentAction.RECOMMEND
        elif any(k in text_lower for k in ["explain", "解释", "说明"]):
            intent["action"] = AgentAction.EXPLAIN
        else:
            intent["action"] = AgentAction.EXPLAIN
            intent["response"] = "I can help you with detection, segmentation, analysis, or parameter tuning. Try saying 'detect objects in this image' or 'analyze the last result'."

        # 参数提取 (正则匹配)
        conf_match = re.search(r'conf(?:idence)?\s*[=:]?\s*(0\.\d+)', text_lower)
        if conf_match:
            intent["params"]["conf"] = float(conf_match.group(1))
        iou_match = re.search(r'iou\s*[=:]?\s*(0\.\d+)', text_lower)
        if iou_match:
            intent["params"]["iou"] = float(iou_match.group(1))
        max_match = re.search(r'max\s*(?:det|objects)?\s*[=:]?\s*(\d+)', text_lower)
        if max_match:
            intent["params"]["max_det"] = int(max_match.group(1))

        # 类别过滤
        class_keywords = {
            "person": "person", "people": "person", "人": "person", "行人": "person",
            "car": "car", "vehicle": "car", "车": "car", "汽车": "car",
            "dog": "dog", "cat": "cat", "animal": "animal", "动物": "animal",
        }
        for kw, cls_name in class_keywords.items():
            if kw in text_lower:
                intent["params"]["filter_class"] = cls_name
                break

        return intent

    # ---------- 工具方法 ----------
    def _tool_infer(self, **kwargs): pass  # 由 UI 层执行
    def _tool_adjust_conf(self, conf: float): return {"conf": conf}
    def _tool_adjust_iou(self, iou: float): return {"iou": iou}
    def _tool_switch_model(self, model_path: str, task: str): return {"model_path": model_path, "task": task}
    def _tool_analyze(self, df: pd.DataFrame, **kw): return self.analyze_result(df, **kw)
    def _tool_explain(self, result: Dict): return self._generate_explanation(result)
    def _tool_recommend(self, image: np.ndarray, task: str): return self.suggest_params(image, task)

    def _generate_explanation(self, result: Dict) -> str:
        """用自然语言解释结果"""
        lines = ["### 🤖 Agent Analysis"]
        stats = result.get("stats", {})
        if stats:
            lines.append(f"- **Objects detected:** {stats.get('total_objects', 'N/A')}")
            lines.append(f"- **Average confidence:** {stats.get('avg_confidence', 'N/A')}")
            lines.append(f"- **Inference time:** {stats.get('infer_time_ms', 'N/A')}ms")
        issues = result.get("issues", [])
        if issues:
            lines.append("- **Issues found:**")
            for issue in issues:
                lines.append(f"  - ⚠️ {issue}")
        suggestions = result.get("suggestions", [])
        if suggestions:
            lines.append("- **Suggestions:**")
            for s in suggestions:
                lines.append(f"  - 💡 {s}")
        cb = result.get("class_balance", "")
        if cb:
            lines.append(f"- **Balance:** {cb}")
        return "\n".join(lines)

    # ---------- Agent 主决策入口 ----------
    def decide(self, image: Optional[np.ndarray] = None, user_text: Optional[str] = None,
               current_task: str = "detect", current_df: Optional[pd.DataFrame] = None,
               current_time: float = 0.0) -> Dict[str, Any]:
        """
        Agent 核心决策入口.
        返回一个 action plan，由 UI 层执行.
        """
        self.add_to_conversation("user", user_text or "[image inference]")
        response_lines = []
        action_plan = {"type": "noop", "updates": {}, "messages": []}

        # 场景 1: 用户输入自然语言
        if user_text:
            intent = self.parse_natural_language(user_text)
            if intent["action"] == AgentAction.INFER:
                action_plan["type"] = "infer"
                action_plan["updates"].update(intent["params"])
                # 如果有图像，做智能推荐
                if image is not None:
                    rec = self.suggest_params(image, intent["params"].get("task", current_task))
                    action_plan["updates"].update({k: v for k, v in rec.items() if k in ["conf", "iou", "max_det"]})
                    response_lines.append("🧠 I've analyzed the image and tuned parameters for you.")
                    response_lines.append(f"   Recommended: conf={rec['conf']}, iou={rec['iou']}, max_det={rec['max_det']}")
                    if rec["advice"]:
                        response_lines.append(f"   Reason: {rec['advice'][0]}")
                response_lines.append("🔥 Running inference now...")
            elif intent["action"] == AgentAction.ANALYZE and current_df is not None:
                analysis = self.analyze_result(current_df, current_task, current_time)
                action_plan["type"] = "show_analysis"
                action_plan["analysis"] = analysis
                response_lines.append(self._generate_explanation(analysis))
            elif intent["action"] == AgentAction.AUTO_ENHANCE:
                if self.memory.last_result:
                    strategy = self.adaptive_strategy(self.memory.last_result)
                    if strategy:
                        action_plan["type"] = "auto_enhance"
                        action_plan["updates"] = strategy["changes"]
                        response_lines.append(f"🔁 {strategy['reason']}")
                        response_lines.append("Re-running with optimized parameters...")
                    else:
                        response_lines.append("✅ Last result was good. No auto-enhancement needed.")
                else:
                    response_lines.append("⚠️ No previous result to enhance. Please run inference first.")
            elif intent["action"] == AgentAction.RECOMMEND and image is not None:
                rec = self.suggest_params(image, current_task)
                action_plan["type"] = "show_recommendation"
                action_plan["recommendation"] = rec
                response_lines.append("### 🧠 Parameter Recommendation")
                response_lines.append(f"- **Confidence:** {rec['conf']}")
                response_lines.append(f"- **IoU:** {rec['iou']}")
                response_lines.append(f"- **Max Det:** {rec['max_det']}")
                response_lines.append(f"- **Scene type:** {rec['features']['scene_type']}")
                if rec["advice"]:
                    response_lines.append(f"- **Why:** {rec['advice'][0]}")
            elif intent["action"] == AgentAction.EXPLAIN:
                if current_df is not None:
                    analysis = self.analyze_result(current_df, current_task, current_time)
                    response_lines.append(self._generate_explanation(analysis))
                else:
                    response_lines.append(intent["response"] or "How can I help?")
            else:
                response_lines.append(intent["response"] or "How can I help?")

        # 场景 2: 图像直接输入（Smart Infer 模式）
        elif image is not None:
            rec = self.suggest_params(image, current_task)
            action_plan["type"] = "infer"
            action_plan["updates"] = {k: v for k, v in rec.items() if k in ["conf", "iou", "max_det"]}
            response_lines.append("🧠 Smart mode: Agent analyzed the image and auto-tuned parameters.")
            response_lines.append(f"   Scene: {rec['features']['scene_type']} | conf={rec['conf']} | iou={rec['iou']}")

        # 记忆结果
        self.memory.last_params = action_plan.get("updates", {})
        response = "\n".join(response_lines)
        self.add_to_conversation("agent", response)
        action_plan["messages"] = [response]
        return action_plan

    def remember_result(self, df: pd.DataFrame, infer_time: float, task: str):
        """记住推理结果，用于后续自适应分析"""
        self.memory.last_result = self.analyze_result(df, task, infer_time)
        self._save_memory()


# ============================================================
# 8. 主 UI 类（Agent 增强版）
# ============================================================

class YOLO_Master_WebUI:
    def __init__(self, ckpts_root: str):
        self.ckpts_root = Path(ckpts_root)
        self.model_manager = ModelManager(self.ckpts_root)
        self.history = InferenceHistory()
        self.agent = YOLOMasterAgent(self.model_manager, self.history)
        self.model_map = self.model_manager.scan()
        LOGGER.info("YOLO-Master Studio (Agent Enhanced) initialized.")

    # ---------- 工具方法 ----------
    def load_default_image(self):
        p = Path(CONFIG.DEFAULT_IMAGE_DIR)
        if not p.exists(): return None
        files = []
        for ext in CONFIG.IMAGE_EXTS:
            files += sorted(p.glob(f"*{ext}"))
        if not files: return None
        img = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def update_model_dropdown(self, task):
        choices = self.model_map.get(task, [])
        if not choices:
            choices = [CONFIG.DEFAULT_MODELS.get(task, "yolov8n.pt")]
        return gr.update(choices=choices, value=choices[0])

    def refresh_models(self, task):
        self.model_map = self.model_manager.scan()
        return self.update_model_dropdown(task)

    def describe_model(self, task, model_path):
        if not model_path or not model_path.strip():
            return "⚠️ Please enter a model path."
        path = Path(model_path.strip())
        if not path.exists():
            return f"❌ Path does not exist: `{model_path}`"
        if path.is_dir():
            for c in [path/"weights"/"best.pt", path/"weights"/"last.pt", path/"best.pt", path/"last.pt"]:
                if c.exists(): path = c; break
        try:
            model = YOLO(str(path))
            names = model.names; nc = len(names); mtask = getattr(model, "task", "unknown")
            total = sum(p.numel() for p in model.model.parameters()) / 1e6
            return (
                f"### ✅ Model Validated\n"
                f"| Property | Value |\n|---|---|\n"
                f"| **Path** | `{path}` |\n| **Task** | `{mtask}` |\n"
                f"| **Classes** | {nc} |\n| **Parameters** | {total:.2f}M |\n"
                f"| **Names** | {list(names.values())[:8]}... |\n"
            )
        except Exception as e:
            return f"❌ Invalid Model: {e}"

    def get_model_info_panel(self):
        info = self.model_manager.get_info()
        if not info: return "No model loaded."
        return (
            f"### 📋 Model Info\n| Property | Value |\n|---|---|\n"
            f"| **Name** | `{info.get('name', 'N/A')}` |\n| **Task** | `{info.get('task', 'N/A')}` |\n"
            f"| **Classes** | {info.get('nc', 'N/A')} |\n| **Parameters** | {info.get('total_params', 'N/A')} |\n"
            f"| **Trainable** | {info.get('trainable_params', 'N/A')} |\n| **Device** | `{info.get('device', 'N/A')}` |\n"
            f"| **Class Names** | {info.get('names', [])} |\n"
        )

    def get_history_table(self):
        records = self.history.get_all()
        if not records:
            return pd.DataFrame(columns=["ID", "Time", "Task", "Model", "Objects", "Time(ms)"])
        df = pd.DataFrame([{"ID": r["id"], "Time": r["timestamp"], "Task": r["task"],
                            "Model": r["model"], "Objects": r["num_objects"], "Time(ms)": r["infer_time_ms"]}
                           for r in reversed(records)])
        return df

    def clear_history(self):
        self.history.clear()
        return self.get_history_table(), "History cleared.", LOGGER.get_text()

    def clear_logs(self):
        LOGGER.clear(); LOGGER.info("Logs cleared.")
        return LOGGER.get_text()

    # ---------- 核心推理（内部方法） ----------
    def _do_inference(self, image, task, model_path, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes):
        """执行推理的通用逻辑，返回所有结果组件"""
        if image is None:
            return None, None, None, None, "⚠️ No image.", None, LOGGER.get_text()
        device_opt = "cpu" if cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width > 0 else None
        max_det_opt = int(max_det)
        options = {k: True for k in checkboxes}
        if task == "seg" and "retina_masks" not in options:
            options["retina_masks"] = True
        mpath = (custom_path or "").strip() or (model_path or "").strip()
        try:
            model = self.model_manager.load(mpath, task)
        except Exception as e:
            return None, None, None, None, f"❌ Model load: {e}", None, LOGGER.get_text()

        t0 = time.time()
        try:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
            results = model(img_bgr, conf=conf, iou=iou, device=device_opt,
                            max_det=max_det_opt, line_width=line_width_opt, **options)
        except Exception as e:
            LOGGER.error(f"Inference: {e}")
            return None, None, None, None, f"❌ Inference: {e}", None, LOGGER.get_text()

        total_time = (time.time() - t0) * 1000
        res = results[0]
        try:
            res_img = res.plot()
            res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB) if len(res_img.shape) == 3 else res_img
        except Exception:
            res_rgb = image

        # 数据提取
        data_list = []
        if res.boxes:
            for box in res.boxes:
                try:
                    cls_id = int(box.cls[0]) if box.cls.numel() > 0 else 0
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    conf_val = float(box.conf[0]) if box.conf.numel() > 0 else 0.0
                    coords = box.xyxy[0].tolist()
                    data_list.append({
                        "Class ID": cls_id, "Class Name": cls_name,
                        "Confidence": round(conf_val, 3),
                        "x1": round(coords[0], 1), "y1": round(coords[1], 1),
                        "x2": round(coords[2], 1), "y2": round(coords[3], 1),
                    })
                except Exception: pass
        df = pd.DataFrame(data_list)
        num_objects = len(data_list)
        speed = res.speed
        infer_ms = speed.get('inference', total_time)
        preprocess_ms = speed.get('preprocess', 0.0)
        postprocess_ms = speed.get('postprocess', 0.0)
        model_device = self.model_manager.get_device()
        model_name = Path(self.model_manager.current_model_path).name

        # 图表
        class_dist = StatsPlotter.class_dist(df)
        conf_hist = StatsPlotter.conf_hist(df)
        timing = StatsPlotter.timing_trend(self.history.get_all())

        # 摘要
        summary = (
            f"### ✅ Inference Complete\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| **Model** | `{model_name}` |\n| **Task** | `{task}` |\n"
            f"| **Device** | `{model_device}` |\n| **Objects** | **{num_objects}** |\n"
            f"| **Preprocess** | `{preprocess_ms:.1f}ms` |\n| **Inference** | `{infer_ms:.1f}ms` |\n"
            f"| **Postprocess** | `{postprocess_ms:.1f}ms` |\n| **Total** | `{total_time:.1f}ms` |\n"
        )

        # 记录历史 + Agent 记忆
        self.history.add(res_rgb, df, summary, task, model_name, total_time, num_objects)
        self.agent.remember_result(df, total_time, task)

        return res_rgb, df, class_dist, conf_hist, summary, timing, LOGGER.get_text()

    # ---------- 普通推理 ----------
    def inference_single(self, task, image, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes):
        return self._do_inference(image, task, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes)

    # ---------- 🧠 Smart Infer（Agent 自动推荐参数） ----------
    def inference_smart(self, task, image, model_dropdown, custom_path, device, cpu, checkboxes):
        """Agent 分析图像后自动推荐参数并推理"""
        if image is None:
            return None, None, None, None, "⚠️ No image.", None, LOGGER.get_text(), "Please upload an image first."
        plan = self.agent.decide(image=image, current_task=task)
        updates = plan.get("updates", {})
        conf = updates.get("conf", 0.25)
        iou = updates.get("iou", 0.7)
        max_det = updates.get("max_det", 300)
        line_width = updates.get("line_width", 0)
        agent_msg = plan["messages"][0] if plan.get("messages") else "🧠 Smart inference with Agent-tuned parameters."
        LOGGER.info(f"Agent Smart Infer: conf={conf}, iou={iou}, max_det={max_det}")
        res = self._do_inference(image, task, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes)
        return (*res, agent_msg)

    # ---------- 🔁 Auto-Enhance（自适应重试） ----------
    def auto_enhance(self, task, image, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes):
        """基于 Agent 的上次分析，自动优化参数并重试"""
        if not self.agent.memory.last_result:
            return None, None, None, None, "⚠️ No previous result to enhance. Run inference first.", None, LOGGER.get_text(), ""
        strategy = self.agent.adaptive_strategy(self.agent.memory.last_result)
        if not strategy:
            return None, None, None, None, "✅ Last result was good. No enhancement needed.", None, LOGGER.get_text(), ""
        changes = strategy["changes"]
        conf = changes.get("conf", conf)
        iou = changes.get("iou", iou)
        max_det = changes.get("max_det", max_det)
        LOGGER.info(f"Auto-Enhance: {strategy['reason']}")
        res = self._do_inference(image, task, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes)
        msg = f"🔁 Auto-Enhanced: {strategy['reason']}\nNew params: conf={conf}, iou={iou}"
        return (*res, msg)

    # ---------- 🤖 Agent 分析（结果后处理） ----------
    def agent_analyze(self, task, infer_time):
        """Agent 对最近一次结果进行深度分析"""
        # 从 history 获取最新结果
        records = self.history.get_latest(1)
        if not records:
            return "⚠️ No inference result available."
        # 尝试读取对应 CSV
        rid = records[0]["id"]
        csv_path = Path(CONFIG.HISTORY_DIR) / f"{rid}.csv"
        df = pd.read_csv(str(csv_path)) if csv_path.exists() else pd.DataFrame()
        analysis = self.agent.analyze_result(df, task, infer_time)
        return self.agent._generate_explanation(analysis)

    # ---------- 💬 Agent 对话处理 ----------
    def agent_chat(self, message, task, image, current_df_json, current_time, chat_history):
        """处理用户自然语言消息，返回 Agent 回复 + 可能的动作"""
        df = pd.read_json(current_df_json) if current_df_json else None
        plan = self.agent.decide(
            image=image, user_text=message,
            current_task=task, current_df=df, current_time=current_time
        )
        response = plan["messages"][0] if plan.get("messages") else "🤖 How can I help?"
        # 更新对话历史
        if chat_history is None:
            chat_history = []
        chat_history.append([message, response])
        # 将 action plan 编码为 JSON 字符串，供前端解析
        action_json = json.dumps({
            "type": plan.get("type", "noop"),
            "updates": plan.get("updates", {}),
        })
        return chat_history, response, action_json

    def agent_recommend_params(self, image, task):
        """Agent 仅推荐参数，不执行推理"""
        if image is None:
            return "⚠️ Please upload an image first."
        rec = self.agent.suggest_params(image, task)
        lines = [
            "### 🧠 Agent Parameter Recommendation",
            f"- **Confidence:** `{rec['conf']}`",
            f"- **IoU:** `{rec['iou']}`",
            f"- **Max Detections:** `{rec['max_det']}`",
            f"- **Scene Type:** `{rec['features']['scene_type']}`",
            f"- **Image Features:** {rec['features']['resolution']} | brightness={rec['features']['brightness']} | edges={rec['features']['edge_density']}",
        ]
        if rec["advice"]:
            lines.append(f"- **Reasoning:** {rec['advice'][0]}")
        return "\n".join(lines)

    # ---------- 批量处理（保持不变） ----------
    def inference_batch(self, task, files, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes, progress=gr.Progress()):
        if not files: return None, "⚠️ No files.", LOGGER.get_text()
        options = {k: True for k in checkboxes}
        if task == "seg" and "retina_masks" not in options: options["retina_masks"] = True
        device_opt = "cpu" if cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width > 0 else None
        max_det_opt = int(max_det)
        mpath = (custom_path or "").strip() or (model_dropdown or "").strip()
        try: model = self.model_manager.load(mpath, task)
        except Exception as e: return None, f"❌ Model: {e}", LOGGER.get_text()
        total = len(files); all_records = []; batch_times = []
        for i, f in enumerate(files):
            progress(i/total, desc=f"{i+1}/{total}")
            try:
                img = Image.open(f.name).convert("RGB")
                img_np = np.array(img); img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                t0 = time.time()
                results = model(img_bgr, conf=conf, iou=iou, device=device_opt, max_det=max_det_opt, line_width=line_width_opt, **options)
                elapsed = (time.time()-t0)*1000; batch_times.append(elapsed)
                res = results[0]; num = len(res.boxes) if res.boxes else 0
                all_records.append({"File": Path(f.name).name, "Objects": num, "Time(ms)": round(elapsed,2), "Status": "OK"})
            except Exception as e:
                all_records.append({"File": Path(f.name).name, "Objects": 0, "Time(ms)": 0, "Status": f"Error: {e}"})
                LOGGER.error(f"Batch fail: {f.name}")
        df = pd.DataFrame(all_records); avg = sum(batch_times)/len(batch_times) if batch_times else 0
        report = (
            f"### 📦 Batch Report\n| Metric | Value |\n|---|---|\n"
            f"| **Total** | {total} |\n| **Success** | {sum(1 for r in all_records if r['Status']=='OK')} |\n"
            f"| **Failed** | {sum(1 for r in all_records if r['Status']!='OK')} |\n"
            f"| **Total Objects** | {sum(r['Objects'] for r in all_records)} |\n"
            f"| **Avg Time** | `{avg:.1f}ms` |\n"
        )
        progress(1.0, desc="Done")
        return df, report, LOGGER.get_text()

    # ---------- 视频处理（保持不变） ----------
    def inference_video(self, task, video_path, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes, progress=gr.Progress()):
        if not video_path: return None, "⚠️ No video.", LOGGER.get_text()
        options = {k: True for k in checkboxes}
        if task == "seg" and "retina_masks" not in options: options["retina_masks"] = True
        device_opt = "cpu" if cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width > 0 else None
        max_det_opt = int(max_det)
        mpath = (custom_path or "").strip() or (model_dropdown or "").strip()
        try: model = self.model_manager.load(mpath, task)
        except Exception as e: return None, f"❌ Model: {e}", LOGGER.get_text()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, "❌ Cannot open video.", LOGGER.get_text()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out = Path(CONFIG.HISTORY_DIR) / f"vid_{uuid.uuid4().hex[:8]}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        frame_idx = 0; counts = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            try:
                results = model(frame, conf=conf, iou=iou, device=device_opt, max_det=max_det_opt, line_width=line_width_opt, **options)
                res = results[0]; ann = res.plot()
                if ann.shape[0] != h or ann.shape[1] != w: ann = cv2.resize(ann, (w, h))
                writer.write(ann); counts.append(len(res.boxes) if res.boxes else 0)
            except Exception as e:
                LOGGER.error(f"Frame {frame_idx}: {e}"); writer.write(frame)
            frame_idx += 1
            if frame_idx % 10 == 0: progress(frame_idx/total, desc=f"Frame {frame_idx}/{total}")
        cap.release(); writer.release()
        avg = sum(counts)/len(counts) if counts else 0
        report = (
            f"### 🎬 Video Done\n| Metric | Value |\n|---|---|\n"
            f"| **Frames** | {frame_idx} |\n| **Avg Objects/Frame** | {avg:.1f} |\n"
            f"| **Max/Frame** | {max(counts) if counts else 0} |\n| **Output** | `{out.name}` |\n"
        )
        return str(out), report, LOGGER.get_text()

    # ---------- 📷 Webcam (Capture Mode) ----------
    def webcam_infer(self, task, image, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes):
        """Simplified webcam inference for Gradio 4.x capture mode."""
        if image is None:
            return None, "⚠️ Click Capture to take a photo.", LOGGER.get_text()
        res_rgb, df, _, _, summary, _, logs = self._do_inference(
            image, task, model_dropdown, custom_path, conf, iou, device, max_det, line_width, cpu, checkboxes
        )
        return res_rgb, summary, logs

    # ---------- UI 构建 ----------
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
                    <a class="ym-action ym-action-primary" href="{CONFIG.PROJECT_URL}" target="_blank" rel="noopener noreferrer">
                        Star Tencent/YOLO-Master
                    </a>
                    <a class="ym-action ym-action-secondary" href="https://github.com/Tencent/YOLO-Master#-citation" target="_blank" rel="noopener noreferrer">
                        CVPR 2026 Citation
                    </a>
                </div>
            </div>
            <a class="ym-mascot-link" href="{CONFIG.PROJECT_URL}" target="_blank" rel="noopener noreferrer" aria-label="Open Tencent YOLO-Master on GitHub">
                <img class="ym-mascot" src="{CONFIG.MASCOT_IMAGE_URL}" alt="YOLO-Master Tencent mascot">
            </a>
        </section>
        """
    
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
    def stream_color(cls_id: int) -> Tuple[int, int, int]:
        colors = CONFIG.STREAM_BOX_COLORS
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

    def update_stream_preset(self, preset: str):
        config = CONFIG.STREAM_PRESETS.get(preset, CONFIG.STREAM_PRESETS["Balanced"])
        return (
            gr.update(value=config["max_side"]),
            gr.update(value=config["interval"]),
            gr.update(value=config["stride"]),
            gr.update(value=config["max_det"]),
        )

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
        enabled_options = set(checkboxes or []) - CONFIG.STREAM_DISABLED_OPTIONS
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

    def reset_stream_state():
        """Clear cached stream detections after switching cameras or models."""
        return None, "Waiting for webcam stream...", {}

    def launch(self):
        custom_css = """
        .gradio-container { background: #0f0f23 !important; }
        .tabitem { background: #1a1a2e !important; border-radius: 12px !important; }
        .panel { background: #16213e !important; border-radius: 10px !important; border: 1px solid #2a2a4a !important; }
        .block-label { color: #e0e0ff !important; font-weight: 600 !important; }
        button.primary { background: linear-gradient(135deg, #4ecca3, #2d8a6e) !important; color: #0f0f23 !important; font-weight: bold !important; border: none !important; }
        button.secondary { background: #2a2a4a !important; color: #e0e0ff !important; border: 1px solid #4ecca3 !important; }
        .dataframe { background: #16213e !important; color: #e0e0ff !important; }
        .markdown-body { color: #e0e0ff !important; }
        .input-image { border: 2px dashed #4ecca3 !important; border-radius: 12px !important; }
        .output-image { border: 2px solid #4ecca3 !important; border-radius: 12px !important; }
        .status-bar { background: #16213e !important; border-radius: 8px !important; padding: 10px !important; border-left: 4px solid #4ecca3 !important; }
        .log-box { background: #0a0a1a !important; color: #4ecca3 !important; font-family: 'Courier New', monospace !important; border: 1px solid #2a2a4a !important; }
        .chat-user { background: #2a2a4a !important; color: #e0e0ff !important; border-radius: 12px !important; padding: 10px !important; margin: 4px 0 !important; }
        .chat-agent { background: #16213e !important; color: #4ecca3 !important; border-radius: 12px !important; padding: 10px !important; margin: 4px 0 !important; border-left: 3px solid #4ecca3 !important; }
        .insight-box { background: #1a1a2e !important; border: 1px solid #4ecca3 !important; border-radius: 8px !important; padding: 12px !important; }
        """

        with gr.Blocks(title="YOLO-Master Studio | Agent Edition", theme=gr.themes.Base(
            primary_hue="teal", secondary_hue="slate", neutral_hue="zinc"
        ), css=custom_css) as app:
            # 状态变量（隐藏）
            state_agent_action = gr.State("")  # Agent 动作计划
            state_current_df = gr.State("")    # 当前结果 JSON
            state_current_time = gr.State(0.0)  # 当前推理时间

            # ========== Header ==========
            gr.HTML(self.brand_header())

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(get_system_status(), elem_classes=["status-bar"])
                with gr.Column(scale=1):
                    gr.Markdown("📋 No model loaded.", elem_classes=["status-bar"])

            # ========== Main Layout ==========
            with gr.Row(equal_height=False):
                # ---- Sidebar ----
                with gr.Column(scale=1, min_width=320):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 🛠 Model & Task")
                        task_radio = gr.Radio(
                            choices=["detect", "seg", "cls", "pose", "obb"],
                            value="detect", label="Task"
                        )
                        with gr.Row():
                            model_dd = gr.Dropdown(
                                choices=self.model_map["detect"],
                                value=self.model_map["detect"][0] if self.model_map["detect"] else None,
                                label="Model", interactive=True
                            )
                            refresh_btn = gr.Button("🔄", )
                        custom_model_txt = gr.Textbox(
                            value="", label="Custom Model Path",
                            placeholder="./ckpts/yolo_master_n.pt"
                        )
                        validate_btn = gr.Button("✅ Validate", variant="secondary")
                        model_info_btn = gr.Button("📋 Info", variant="secondary")

                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### ⚙️ Parameters")
                        with gr.Accordion("Advanced", open=True):
                            conf_slider = gr.Slider(0, 1, 0.25, step=0.01, label="Confidence")
                            iou_slider = gr.Slider(0, 1, 0.7, step=0.01, label="IoU")
                            with gr.Row():
                                max_det_num = gr.Number(300, label="Max Objects")
                                line_width_num = gr.Number(0, label="Line Width")
                            with gr.Row():
                                device_txt = gr.Textbox("cpu", label="Device", placeholder="0 or cpu")
                                cpu_chk = gr.Checkbox(True, label="Force CPU")

                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 📤 Output Options")
                        options_chk = gr.CheckboxGroup(
                            ["half", "show", "save", "save_txt", "save_crop",
                             "hide_labels", "hide_conf", "agnostic_nms", "retina_masks"],
                            label="Options", value=["retina_masks"]
                        )

                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 🛠 Maintenance")
                        clear_log_btn = gr.Button("🧹 Clear Logs", variant="secondary")
                        clear_hist_btn = gr.Button("🗑 Clear History", variant="secondary")
                        recommend_btn = gr.Button("🧠 Recommend Params", variant="secondary")

                # ---- Main Tabs ----
                with gr.Column(scale=3):
                    with gr.Tabs():
                        # --- Tab 1: Single Image (Agent Enhanced) ---
                        with gr.Tab("🖼️ Single Image"):
                            with gr.Row():
                                inp_img = gr.Image(type="numpy", label="Input",
                                                   value=self.load_default_image(), elem_classes=["input-image"])
                                out_img = gr.Image(type="numpy", label="Output",
                                                   interactive=False, elem_classes=["output-image"])
                            with gr.Row():
                                run_btn = gr.Button("🔥 Run Inference", variant="primary")
                                smart_btn = gr.Button("🧠 Smart Infer", variant="primary")
                                enhance_btn = gr.Button("🔁 Auto-Enhance", variant="secondary")
                            with gr.Row():
                                with gr.Column(scale=2):
                                    info_md = gr.Markdown("Waiting...", elem_classes=["status-bar"])
                                with gr.Column(scale=1):
                                    out_df = gr.Dataframe(
                                        headers=["Class ID", "Class Name", "Confidence", "x1", "y1", "x2", "y2"],
                                        label="Detections", interactive=False
                                    )
                            # Agent Insights 面板
                            with gr.Row():
                                agent_insight = gr.Markdown(
                                    "### 💡 Agent Insights\nUpload an image and click **Smart Infer** to see Agent analysis.",
                                    elem_classes=["insight-box"]
                                )
                            with gr.Row():
                                class_dist_img = gr.Image(label="Class Distribution", interactive=False)
                                conf_hist_img = gr.Image(label="Confidence Histogram", interactive=False)
                            timing_img = gr.Image(label="Latency Trend", interactive=False)

                        # --- Tab 2: Batch ---
                        with gr.Tab("📦 Batch"):
                            batch_files = gr.Files(file_types=["image"], label="Upload Images", file_count="multiple")
                            run_batch_btn = gr.Button("🔥 Run Batch", variant="primary")
                            with gr.Row():
                                batch_df = gr.Dataframe(label="Batch Results", interactive=False)
                                batch_report = gr.Markdown(elem_classes=["status-bar"])

                        # --- Tab 3: Video ---
                        with gr.Tab("🎬 Video"):
                            video_in = gr.Video(label="Upload Video")
                            run_video_btn = gr.Button("🔥 Process Video", variant="primary")
                            with gr.Row():
                                video_out = gr.Video(label="Output Video", interactive=False)
                                video_report = gr.Markdown(elem_classes=["status-bar"])

                        # --- Tab 4: Webcam (Capture Mode) ---
                        with gr.Tab("📷 Webcam"):
                            gr.Markdown("### 📸 Camera Capture\nClick **Capture** to take a photo and run inference.")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    webcam_in = gr.Image(
                                        sources=["webcam"], label="Camera"
                                    )
                                    run_webcam_btn = gr.Button("📸 Capture & Infer", variant="primary")
                                with gr.Column(scale=1):
                                    webcam_out = gr.Image(
                                        label="Result", interactive=False,
                                        elem_classes=["output-image"]
                                    )
                                    webcam_info = gr.Markdown(
                                        "Click Capture to start.", elem_classes=["status-bar"]
                                    )

                        # --- Tab 5: 🤖 Agent ---
                        with gr.Tab("🤖 Agent"):
                            gr.Markdown(
                                "### 💬 Agent Chat\n"
                                "Talk to the Agent naturally. Try: *'detect people with conf 0.3'*, *'analyze the last result'*, *'why so few detections?'*"
                            )
                            with gr.Row():
                                with gr.Column(scale=2):
                                    chatbot = gr.Chatbot(label="Conversation", height=400, elem_classes=["chat-agent"])
                                    with gr.Row():
                                        chat_input = gr.Textbox(
                                            label="Message", placeholder="Ask the Agent...",
                                            show_label=False
                                        )
                                        chat_send = gr.Button("➤ Send", variant="primary")
                                    with gr.Row():
                                        gr.Button("Quick: Detect objects", variant="secondary").click(
                                            lambda: "detect objects in this image", outputs=chat_input
                                        )
                                        gr.Button("Quick: Analyze result", variant="secondary").click(
                                            lambda: "analyze the last result", outputs=chat_input
                                        )
                                        gr.Button("Quick: Recommend params", variant="secondary").click(
                                            lambda: "recommend parameters for this image", outputs=chat_input
                                        )
                                        gr.Button("Clear Chat", variant="secondary").click(
                                            lambda: None, outputs=chatbot
                                        )
                                with gr.Column(scale=1):
                                    agent_response = gr.Markdown(
                                        "### 🤖 Agent Status\nAgent is ready.",
                                        elem_classes=["insight-box"]
                                    )
                                    agent_recommend = gr.Markdown(
                                        "### 🧠 Parameter Recommendation\nUpload an image and ask for recommendations.",
                                        elem_classes=["insight-box"]
                                    )

                        # --- Tab 6: History ---
                        with gr.Tab("🕘 History"):
                            history_df = gr.Dataframe(label="Inference History", interactive=False)
                            refresh_hist_btn = gr.Button("🔄 Refresh", )
                            hist_detail = gr.Markdown()

                        # --- Tab 7: Logs ---
                        with gr.Tab("📜 Logs"):
                            log_box = gr.Textbox(label="Runtime Logs", lines=20, interactive=False,
                                                value=LOGGER.get_text(), elem_classes=["log-box"])
                            refresh_log_btn = gr.Button("🔄 Refresh", )

                        # --- Tab 8: About ---
                        with gr.Tab("ℹ️ About"):
                            gr.Markdown(
                                "### YOLO-Master Studio | Agent Edition\n"
                                "AI-powered real-time object detection with embedded Agent intelligence.\n\n"
                                "**Agent Features:**\n"
                                "- 🧠 Smart Parameter Recommendation\n"
                                "- 💡 Result Quality Analysis\n"
                                "- 🔁 Auto-Enhance Adaptive Retry\n"
                                "- 💬 Natural Language Chat Interface\n"
                                "- 📝 Persistent User Preference Memory\n\n"
                                "**Paper:** [arXiv:2512.23273](https://arxiv.org/abs/2512.23273)\n"
                                "**GitHub:** [isLinXu/YOLO-Master](https://github.com/isLinXu/YOLO-Master)\n"
                            )

            # ========== Event Bindings ==========
            # Model management
            task_radio.change(fn=self.update_model_dropdown, inputs=task_radio, outputs=model_dd)
            refresh_btn.click(fn=self.refresh_models, inputs=task_radio, outputs=model_dd)
            validate_btn.click(fn=self.describe_model, inputs=[task_radio, custom_model_txt], outputs=info_md)
            model_info_btn.click(fn=self.get_model_info_panel, outputs=info_md)

            # Normal inference
            run_btn.click(
                fn=self.inference_single,
                inputs=[task_radio, inp_img, model_dd, custom_model_txt,
                        conf_slider, iou_slider, device_txt, max_det_num, line_width_num, cpu_chk, options_chk],
                outputs=[out_img, out_df, class_dist_img, conf_hist_img, info_md, timing_img, log_box]
            )

            # 🧠 Smart Infer (Agent auto-tuned)
            smart_outputs = [out_img, out_df, class_dist_img, conf_hist_img, info_md, timing_img, log_box, agent_insight]
            smart_btn.click(
                fn=self.inference_smart,
                inputs=[task_radio, inp_img, model_dd, custom_model_txt, device_txt, cpu_chk, options_chk],
                outputs=smart_outputs
            )

            # 🔁 Auto-Enhance
            enhance_outputs = [out_img, out_df, class_dist_img, conf_hist_img, info_md, timing_img, log_box, agent_insight]
            enhance_btn.click(
                fn=self.auto_enhance,
                inputs=[task_radio, inp_img, model_dd, custom_model_txt,
                        conf_slider, iou_slider, device_txt, max_det_num, line_width_num, cpu_chk, options_chk],
                outputs=enhance_outputs
            )

            # 📤 Recommend Params (仅推荐，不推理)
            recommend_btn.click(
                fn=self.agent_recommend_params,
                inputs=[inp_img, task_radio],
                outputs=agent_recommend
            )

            # 💬 Agent Chat
            chat_send.click(
                fn=self.agent_chat,
                inputs=[chat_input, task_radio, inp_img, state_current_df, state_current_time, chatbot],
                outputs=[chatbot, agent_response, state_agent_action]
            )
            # 按 Enter 发送
            chat_input.submit(
                fn=self.agent_chat,
                inputs=[chat_input, task_radio, inp_img, state_current_df, state_current_time, chatbot],
                outputs=[chatbot, agent_response, state_agent_action]
            )

            # Batch / Video
            run_batch_btn.click(
                fn=self.inference_batch,
                inputs=[task_radio, batch_files, model_dd, custom_model_txt,
                        conf_slider, iou_slider, device_txt, max_det_num, line_width_num, cpu_chk, options_chk],
                outputs=[batch_df, batch_report, log_box]
            )
            run_video_btn.click(
                fn=self.inference_video,
                inputs=[task_radio, video_in, model_dd, custom_model_txt,
                        conf_slider, iou_slider, device_txt, max_det_num, line_width_num, cpu_chk, options_chk],
                outputs=[video_out, video_report, log_box]
            )

            # 🎥 Webcam (Capture Mode)
            run_webcam_btn.click(
                fn=self.webcam_infer,
                inputs=[task_radio, webcam_in, model_dd, custom_model_txt,
                        conf_slider, iou_slider, device_txt, max_det_num, line_width_num, cpu_chk, options_chk],
                outputs=[webcam_out, webcam_info, log_box],
            )

            # History / Logs
            refresh_hist_btn.click(fn=self.get_history_table, outputs=history_df)
            clear_hist_btn.click(fn=self.clear_history, outputs=[history_df, hist_detail, log_box])
            refresh_log_btn.click(fn=lambda: LOGGER.get_text(), outputs=log_box)
            clear_log_btn.click(fn=self.clear_logs, outputs=log_box)

        app.launch(share=True, server_name="0.0.0.0")


# ============================================================
# 9. 启动入口
# ============================================================

if __name__ == "__main__":
    CKPTS = Path(__file__).parent / "ckpts"
    CKPTS.mkdir(parents=True, exist_ok=True)
    Path(CONFIG.HISTORY_DIR).mkdir(parents=True, exist_ok=True)
    Path(CONFIG.AGENT_MEMORY_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Starting YOLO-Master Studio | Agent Edition")
    print(f"Checkpoints: {CKPTS}")
    print(f"History: {Path(CONFIG.HISTORY_DIR)}")
    print(f"Agent Memory: {Path(CONFIG.AGENT_MEMORY_DIR)}")
    ui = YOLO_Master_WebUI(str(CKPTS))
    ui.launch()
