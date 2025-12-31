import os
import gc
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import gradio as gr
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")


class GlobalConfig:
    """Global configuration parameters for easy modification."""
    # Default model files mapping
    DEFAULT_MODELS = {
        "detect": "yolov8n.pt",
        "seg": "yolov8n-seg.pt",
        "cls": "yolov8n-cls.pt",
        "pose": "yolov8n-pose.pt",
        "obb": "yolov8n-obb.pt"
    }
    # Allowed image formats
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # UI Theme
    THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")


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
                  checkboxes: List[str]):
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
        options = {k: True for k in checkboxes}
        
        # Optimization for segmentation task
        if task == "seg" and "retina_masks" not in options:
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
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
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
        
        df = pd.DataFrame(data_list)
        
        # 4.3 Summary Info
        speed = res.speed
        infer_time = speed.get('inference', 0.0)
        model_device = self.model_manager.get_current_model_info()
        
        summary = (
            f"### ✅ Inference Done\n"
            f"- **Model:** `{Path(self.model_manager.current_model_path).name}`\n"
            f"- **Time:** `{infer_time:.1f}ms`\n"
            f"- **Objects:** {len(data_list)}\n"
            f"- **Device:** `{model_device}`"
        )
        
        return res_img, df, summary

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
            gr.Markdown("# 🚀 YOLO-Master Dashboard")
            
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
                            device_txt = gr.Textbox("0", label="Device ID (e.g. 0, cpu)", placeholder="0 or cpu")
                            cpu_chk = gr.Checkbox(False, label="Force CPU")

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
                            with gr.Row():
                                inp_img = gr.Image(type="numpy", label="Input Image", height=500)
                                out_img = gr.Image(type="numpy", label="Inference Result", height=500, interactive=False)
                            info_md = gr.Markdown(value="Waiting for input...")

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
            
            # 2. Inference Logic
            run_btn.click(
                fn=self.inference,
                inputs=[
                    task_radio, inp_img, model_dd, custom_model_txt,
                    conf_slider, iou_slider, device_txt, 
                    max_det_num, line_width_num, cpu_chk, options_chk
                ],
                outputs=[out_img, out_df, info_md]
            )

        app.launch(share=False, inbrowser=True)


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
