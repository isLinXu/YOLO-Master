import os
import sys

# Add project root to path (at the beginning!) to ensure local ultralytics is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import glob
import time
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

# Configure environment: Ensure local project root is prioritized in sys.path
# to avoid conflicts with installed site-packages.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from sparse_sahi_inference import SparseSAHIPredictor
except ImportError:
    print("RuntimeWarning: SparseSAHIPredictor not found. Legacy baseline will be disabled.")
    SparseSAHIPredictor = None


def run_comparison():
    """
    Executes a performance and precision benchmark comparing Standard Inference,
    Full SAHI, and Integrated Sparse SAHI.
    """

    # --- Configuration & Resource Initialization ---
    model_ckpt = os.path.join(project_root, 'yolo_master_n.pt')
    fallback_path = "/Users/gatilin/PycharmProjects/ultralytics-8.3.240-v251220/ckpts/yolo-master.pt"

    model_path = model_ckpt if os.path.exists(model_ckpt) else fallback_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"IOError: Model checkpoint not found at {model_path}")

    print(f"Loading Model Weights: {model_path}")
    model = YOLO(model_path)

    # Initialize baseline predictor for comparative analysis
    legacy_sahi = SparseSAHIPredictor(model_path) if SparseSAHIPredictor else None

    # Define Dataset: Prioritize local COCO128 subset for high-resolution testing
    test_images = []
    coco_subset = "/Users/gatilin/PycharmProjects/datasets/coco128/images/train2017"
    if os.path.exists(coco_subset):
        # test_images = sorted(glob.glob(os.path.join(coco_subset, "*.jpg")))[:5]
        test_images = sorted(glob.glob(os.path.join(coco_subset, "*.jpg")))

    else:
        test_images = [os.path.join(ASSETS, "bus.jpg"), os.path.join(ASSETS, "zidane.jpg")]

    output_dir = os.path.join(project_root, "runs/integrated_vis")
    os.makedirs(output_dir, exist_ok=True)

    # Cache class names for efficient string lookup during visualization
    class_map = model.names

    for img_path in test_images:
        fname = os.path.basename(img_path)
        print(f"Benchmarking Inference Pipeline: {fname}")

        frame = cv2.imread(img_path)
        if frame is None: continue

        # --- Stage 1: Standard Inference (Low-Resolution Baseline) ---
        # Purpose: Benchmark speed at low spatial resolution (224px)
        t_start = time.perf_counter()
        std_results = model.predict(img_path, imgsz=224, conf=0.25, verbose=False, sparse_sahi=False)[0]
        std_latency = time.perf_counter() - t_start

        vis_std = frame.copy()
        for box in std_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = f"{class_map[int(box.cls[0])]} {float(box.conf[0]):.2f}"
            cv2.rectangle(vis_std, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_std, label, (x1, max(y1 - 5, 20)), 0, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_std, f"Standard (224px) | Objs: {len(std_results.boxes)}", (15, 30), 0, 0.6, (0, 255, 0), 2)

        # --- Stage 2: Standard SAHI (Exhaustive Sliding Window) ---
        # Purpose: Benchmark legacy slicing method (computational upper bound)
        vis_full_sahi = frame.copy()
        full_sahi_latency = 0
        if legacy_sahi:
            try:
                res_full = legacy_sahi.predict_standard(img_path, conf_thres=0.25)
                f_boxes, f_scores, f_cls, meta_full = res_full
                full_sahi_latency = meta_full.get('inference_time', 0)

                # Render slicing grid (Spatial Partitions)
                overlay = vis_full_sahi.copy()
                for sx1, sy1, sx2, sy2 in meta_full.get('slices', []):
                    cv2.rectangle(overlay, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (200, 200, 200), 1)
                cv2.addWeighted(overlay, 0.3, vis_full_sahi, 0.7, 0, vis_full_sahi)

                for b, s, c in zip(f_boxes, f_scores, f_cls):
                    pts = b.cpu().numpy().astype(int)
                    label = f"{class_map[int(c)]} {float(s):.2f}"
                    cv2.rectangle(vis_full_sahi, (pts[0], pts[1]), (pts[2], pts[3]), (255, 0, 255), 2)
                cv2.putText(vis_full_sahi, f"Full SAHI | Objs: {len(f_boxes)}", (15, 30), 0, 0.6, (255, 0, 255), 2)
            except Exception as e:
                print(f"Exception in Full SAHI Branch: {e}")

        # --- Stage 3: Integrated Sparse SAHI (Proposed Optimization) ---
        # Purpose: Validate Content-Adaptive Sparse Slicing (Dynamic ROI selection)
        t_start = time.perf_counter()
        sparse_res = model.predict(img_path, sparse_sahi=True, conf=0.25, verbose=False)[0]
        sparse_latency = time.perf_counter() - t_start

        vis_sparse = frame.copy()
        meta_sparse = getattr(sparse_res, 'sparse_sahi_metadata', {})

        # Render Active Slices (ROIs identified by Objectness Mask)
        if 'slices' in meta_sparse:
            overlay = vis_sparse.copy()
            for sx1, sy1, sx2, sy2 in meta_sparse['slices']:
                cv2.rectangle(overlay, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 255), 2)
            cv2.addWeighted(overlay, 0.3, vis_sparse, 0.7, 0, vis_sparse)

        sources = meta_sparse.get('final_sources', [0] * len(sparse_res.boxes))
        for i, box in enumerate(sparse_res.boxes):
            pts = box.xyxy[0].cpu().numpy().astype(int)
            src_id = sources[i] if i < len(sources) else 0
            # Color Coding: Global detections (Blue) vs. Slice-refined detections (Red)
            color = (255, 0, 0) if src_id == 0 else (0, 0, 255)
            label = f"{'[G]' if src_id == 0 else '[S]'} {class_map[int(box.cls[0])]} {float(box.conf[0]):.2f}"
            cv2.rectangle(vis_sparse, (pts[0], pts[1]), (pts[2], pts[3]), color, 2)
            cv2.putText(vis_sparse, label, (pts[0], max(pts[1] - 5, 20) + (15 if src_id == 1 else 0)), 0, 0.5,
                        (255, 255, 255), 1)
        cv2.putText(vis_sparse, f"Sparse SAHI | Objs: {len(sparse_res.boxes)}", (15, 30), 0, 0.6, (0, 0, 255), 2)

        # --- Stage 4: Objectness Mask Visualization ---
        # Purpose: Visualize the Spatial Probability Distribution of target objects
        vis_mask = frame.copy()
        if 'objectness_map' in meta_sparse and meta_sparse['objectness_map'] is not None:
            obj_map = meta_sparse['objectness_map']
            # Feature Normalization for Heatmap Generation
            norm_map = (obj_map / (obj_map.max() + 1e-6) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            vis_mask = cv2.addWeighted(heatmap, 0.6, vis_mask, 0.4, 0)
            cv2.putText(vis_mask, f"Objectness Max: {obj_map.max():.2f}", (15, 30), 0, 0.6, (255, 255, 255), 2)

        # --- Multi-View Orchestration (Stitching) ---
        target_h = 720  # Normalize vertical resolution for side-by-side display
        processed_views = []
        for v in [vis_std, vis_full_sahi, vis_sparse, vis_mask]:
            h, w = v.shape[:2]
            processed_views.append(cv2.resize(v, (int(w * target_h / h), target_h)))

        canvas_body = np.hstack(processed_views)

        # Metadata Header & Performance Footer
        canvas_w = canvas_body.shape[1]
        header = np.zeros((70, canvas_w, 3), dtype=np.uint8)
        footer = np.zeros((60, canvas_w, 3), dtype=np.uint8)

        cv2.putText(header, f"Inference Comparison: {fname} | Project YOLO-Master", (30, 45), 0, 0.8, (255, 255, 255),
                    2)

        # Align latency metrics with respective viewports
        x_offsets = [0] + list(np.cumsum([v.shape[1] for v in processed_views]))
        cv2.putText(footer, f"Latency: {std_latency:.3f}s", (x_offsets[0] + 20, 40), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(footer, f"Latency: {full_sahi_latency:.3f}s", (x_offsets[1] + 20, 40), 0, 0.7, (255, 0, 255), 2)
        cv2.putText(footer, f"Latency: {sparse_latency:.3f}s", (x_offsets[2] + 20, 40), 0, 0.7, (0, 255, 255), 2)

        final_output = np.vstack((header, canvas_body, footer))
        save_path = os.path.join(output_dir, f"benchmark_{fname}")
        cv2.imwrite(save_path, final_output)
        print(f"Diagnostic Visualization Exported: {save_path}")


if __name__ == "__main__":
    run_comparison()