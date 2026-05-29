import os
import gc
import time
import warnings
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
    STREAM_PRESETS = {
        "Fast": {"max_side": 480, "interval": 0.10, "stride": 3, "max_det": 80},
        "Balanced": {"max_side": 640, "interval": 0.15, "stride": 2, "max_det": 120},
        "Quality": {"max_side": 832, "interval": 0.20, "stride": 1, "max_det": 200},
    }
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
        stream_state: Optional[Dict[str, Any]],
    ):
        """Run inference for a webcam frame without refreshing the detections table."""
        stream_state = stream_state or {}
        if frame is None:
            return None, "Waiting for webcam stream...", stream_state

        now = time.monotonic()
        frame_index = int(stream_state.get("frame_index", 0)) + 1
        processed_frames = int(stream_state.get("processed_frames", 0))
        skipped_frames = int(stream_state.get("skipped_frames", 0))
        last_time = float(stream_state.get("last_time", 0.0))
        last_summary = stream_state.get("last_summary")
        stream_state["frame_index"] = frame_index

        frame_stride = max(1, int(frame_stride or 1))
        should_skip_stride = last_summary is not None and (frame_index - 1) % frame_stride != 0
        should_skip_time = (
            min_frame_interval
            and last_summary is not None
            and now - last_time < float(min_frame_interval)
        )
        if (
            should_skip_stride
            or should_skip_time
        ):
            skipped_frames += 1
            stream_state["skipped_frames"] = skipped_frames
            return gr.update(), last_summary, stream_state

        process_start = time.monotonic()
        out_img, _df, summary = self.inference(
            task,
            frame,
            model_dropdown,
            custom_model_path,
            conf,
            iou,
            device,
            stream_max_det,
            line_width,
            cpu,
            checkboxes,
            stream_mode=True,
            stream_max_side=stream_max_side,
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
                f"- **Frame Size:** `{int(stream_max_side)}px max side`\n"
                f"- **Processed / Skipped:** `{processed_frames}` / `{skipped_frames}`"
            )
            stream_state = {
                "frame_index": frame_index,
                "processed_frames": processed_frames,
                "skipped_frames": skipped_frames,
                "last_time": process_end,
                "last_summary": summary,
                "ema_fps": ema_fps,
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

                    # Output Options
                    options_chk = gr.CheckboxGroup(
                        ["half", "show", "save", "save_txt", "save_crop", "hide_labels", "hide_conf", "agnostic_nms", "retina_masks"],
                        label="Output Options",
                        value=[]
                    )
                    
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
                                    webcam_info_md = gr.Markdown(value="Waiting for webcam stream...")
                                    stream_state = gr.State({})

                        with gr.TabItem("📊 Data Analysis"):
                            gr.Markdown("### Detections Data")
                            out_df = gr.Dataframe(
                                headers=["Class ID", "Class Name", "Confidence", "x1", "y1", "x2", "y2"],
                                label="Raw Detections"
                            )

            # ================= Event Binding =================
            
            # 1. Auto-refresh model list
            task_radio.change(fn=self.update_model_dropdown, inputs=task_radio, outputs=model_dd)
            refresh_btn.click(fn=self.refresh_models, inputs=task_radio, outputs=model_dd)
            validate_btn.click(fn=self.describe_model, inputs=[task_radio, custom_model_txt], outputs=info_md)
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
            webcam_img.stream(
                fn=self.inference_stream,
                inputs=[
                    task_radio, webcam_img, model_dd, custom_model_txt,
                    conf_slider, iou_slider, device_txt,
                    line_width_num, cpu_chk, options_chk,
                    stream_max_side, min_frame_interval, stream_frame_stride, stream_max_det, stream_state
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
