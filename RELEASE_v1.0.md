# YOLO-Master v1.0: The MoE Era of Real-Time Detection

We are excited to announce the first official release of **YOLO-Master**, a novel YOLO architecture integrated with **Mixture-of-Experts (MoE)**. This release brings significant improvements in accuracy-latency trade-offs, specifically targeting real-time object detection and segmentation tasks.

> **"Adaptive Intelligence for Every Scene"** — YOLO-Master introduces instance-conditional computation, dynamically allocating resources where they are needed most.

## 🚀 Key Highlights

*   **MoE Architecture Integration**: Native support for Mixture-of-Experts with dynamic routing.
*   **SOTA Performance**: Achieves **42.4% mAP** on COCO with just **1.62ms** latency (N-scale), outperforming YOLOv10/v11/v12.
*   **Segmentation Breakthrough**: **+2.8% mAPmask** gain over YOLOv12-seg-N.
*   **Hardware-Aware Optimization**: Optimized for GPU (Batched Compute) and Mobile (Ghost Experts).

## 🛠 New Features in v1.0

### 1. Advanced MoE Modules
*   **`ModularRouterExpertMoE` (Recommended)**: A highly stable, plug-and-play MoE block featuring:
    *   **Shared Experts**: Ensures baseline performance and prevents training collapse.
    *   **Z-Loss Integration**: Stabilizes router logits for smoother convergence.
*   **`UltraOptimizedMoE`**: Designed for extreme speed, featuring **Batched Expert Computation** which eliminates Python loops, delivering **3-5x inference speedup** on GPUs.
*   **`GhostExpert`**: Parameter-efficient experts based on GhostNet, reducing memory bandwidth pressure for mobile/edge deployment.

### 2. Intelligent Routing
*   **`EfficientSpatialRouter`**: Reduces routing FLOPs by >90% via spatial pre-pooling.
*   **`DynamicRouting`**: Adaptive computational resource allocation based on scene complexity.

### 3. Stability & Ease of Use
*   **Training Stability**: Solved common MoE training instability with Shared Expert paths and specialized router initialization (`std=0.01`).
*   **Deployment Ready**: Full support for ONNX and TensorRT export.
    *   *New Wiki Guide*: [Hardware Deployment & Inference Optimization](wiki/MoE_Modules_Explanation.md#5-硬件部署与推理优化指南)

## 📊 Benchmarks (COCO val2017)

| Model | Size | mAP (box) | Latency (T4 GPU) | Comparison |
| :--- | :--- | :--- | :--- | :--- |
| YOLOv10-N | 640 | 38.5 | 1.84ms | - |
| YOLOv11-N | 640 | 39.4 | 1.50ms | - |
| YOLOv12-N | 640 | 40.6 | 1.64ms | - |
| **YOLO-Master-N** | **640** | **42.4** | **1.62ms** | **SOTA** 🏆 |

## 🔗 Resources

*   **📜 Detailed Documentation**: [Wiki: MoE Modules Explained](wiki/MoE_Modules_Explanation.md)
*   **📄 Paper**: [arXiv:2512.23273](https://arxiv.org/abs/2512.23273)
*   **🤗 HuggingFace Demo**: [YOLO-Master Spaces](https://huggingface.co/spaces/xx)

## 📥 Quick Start

**Installation**
```bash
git clone https://github.com/isLinXu/YOLO-Master.git
cd YOLO-Master
pip install -r requirements.txt
pip install -e .
```

**Training (Single GPU)**
```python
from ultralytics import YOLO

model = YOLO('yolo-master-n.yaml')
results = model.train(data='coco.yaml', epochs=100, imgsz=640)
```

## 🤝 Contributors

Special thanks to the research team at Tencent Youtu Lab and Singapore Management University.

---
*Full Changelog*: https://github.com/isLinXu/YOLO-Master/commits/v1.0
