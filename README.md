<div align="center">
  <h1>YOLO-MASTER</h1>


<p align="left"> <a href="https://huggingface.co/spaces/xx"> <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces"> </a> <a href="https://colab.research.google.com/github/isLinXu/YOLO-Master"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a> <a href="https://arxiv.org/abs/2512.23273"> <img src="https://img.shields.io/badge/arXiv-2512.23273-b31b1b.svg" alt="arXiv"> </a>  <a href="https://github.com/isLinXu/YOLO-Master/releases"> <img src="https://img.shields.io/badge/%F0%9F%93%A6-Model%20Zoo-orange" alt="Model Zoo"> </a> <a href="./LICENSE"> <img src="https://img.shields.io/badge/License-AGPL%203.0-blue.svg" alt="AGPL 3.0"> </a> <a href="https://github.com/ultralytics/ultralytics"> <img src="https://img.shields.io/badge/Ultralytics-YOLO-blue" alt="Ultralytics"> </a> </p>


  <p align="center">
    YOLO-Master: 
    <b><u>M</u></b>OE-<b><u>A</u></b>ccelerated with 
    <b><u>S</u></b>pecialized <b><u>T</u></b>ransformers for 
    <b><u>E</u></b>nhanced <b><u>R</u></b>eal-time Detection.
  </p>
</div>

<div align="center">
  <div style="text-align: center; margin-bottom: 8px;">
    <a href="https://github.com/isLinXu" style="text-decoration: none;"><b>Xu Lin</b></a><sup>1*</sup>&nbsp;&nbsp;
    <a href="https://pjl1995.github.io/" style="text-decoration: none;"><b>Jinlong Peng</b></a><sup>1*</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=fa4NkScAAAAJ" style="text-decoration: none;"><b>Zhenye Gan</b></a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?hl=en&user=cU0UfhwAAAAJ" style="text-decoration: none;"><b>Jiawen Zhu</b></a><sup>2</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=JIKuf4AAAAAJ&hl=zh-TW" style="text-decoration: none;"><b>Jun Liu</b></a><sup>1</sup>
  </div>

  <div style="text-align: center; margin-bottom: 4px; font-size: 0.95em;">
    <sup>1</sup>Tencent Youtu Lab &nbsp;&nbsp;&nbsp;
    <sup>2</sup>Singapore Management University
  </div>

  <div style="text-align: center; margin-bottom: 12px; font-size: 0.85em; color: #666; font-style: italic;">
    <sup>*</sup>Equal Contribution
  </div>

  <div style="text-align: center;">
    <div style="font-family: 'Courier New', Courier, monospace; font-size: 0.85em; background-color: #f6f8fa; padding: 10px; border-radius: 6px; display: inline-block; line-height: 1.4; text-align: left;">
      {gatilin, jeromepeng, wingzygan, juliusliu}@tencent.com <br>
      jwzhu.2022@phdcs.smu.edu.sg
    </div>
  </div>
</div>
<br>

---

## 💡 A Humble Beginning (Introduction)

> **"Exploring the frontiers of Dynamic Intelligence in YOLO."**

This work represents our passionate exploration into the evolution of Real-Time Object Detection (RTOD). To the best of our knowledge, **YOLO-Master is the first work to deeply integrate Mixture-of-Experts (MoE) with the YOLO architecture on general-purpose datasets.**

Most existing YOLO models rely on static, dense computation—allocating the same computational budget to a simple sky background as they do to a complex, crowded intersection. We believe detection models should be more "adaptive", much like the human visual system. While this initial exploration may be not perfect, it demonstrates the significant potential of **Efficient Sparse MoE (ES-MoE)** in balancing high precision with ultra-low latency. We are committed to continuous iteration and optimization to refine this approach further.

Looking forward, we draw inspiration from the transformative advancements in LLMs and VLMs. We are committed to refining this approach and extending these insights to fundamental vision tasks, with the ultimate goal of tackling more ambitious frontiers like Open-Vocabulary Detection and Open-Set Segmentation.

<details>
  <summary>
  <font size="+1"><b>Abstract</b></font>
  </summary>
Existing Real-Time Object Detection (RTOD) methods commonly adopt YOLO-like architectures for their favorable trade-off between accuracy and speed. However, these models rely on static dense computation that applies uniform processing to all inputs, misallocating representational capacity and computational resources such as over-allocating on trivial scenes while under-serving complex ones. This mismatch results in both computational redundancy and suboptimal detection performance.

To overcome this limitation, we propose YOLO-Master, a novel YOLO-like framework that introduces instance-conditional adaptive computation for RTOD. This is achieved through an Efficient Sparse Mixture-of-Experts (ES-MoE) block that dynamically allocates computational resources to each input according to its scene complexity. At its core, a lightweight dynamic routing network guides expert specialization during training through a diversity enhancing objective, encouraging complementary expertise among experts. Additionally, the routing network adaptively learns to activate only the most relevant experts, thereby improving detection performance while minimizing computational overhead during inference.

Comprehensive experiments on five large-scale benchmarks demonstrate the superiority of YOLO-Master. On MS COCO, our model achieves 42.4\% AP with 1.62ms latency, outperforming YOLOv13-N by +0.8\% mAP and 17.8\% faster inference. Notably, the gains are most pronounced on challenging dense scenes, while the model preserves efficiency on typical inputs and maintains real-time inference speed. Code: [isLinXu/YOLO-Master](https://github.com/isLinXu/YOLO-Master)
</details>

---

## 🎨 Architecture

<div align="center">
  <img width="90%" alt="YOLO-Master Architecture" src="https://github.com/user-attachments/assets/6caa1065-af77-4f77-8faf-7551c013dacd" />
  <p><i>YOLO-Master introduces ES-MoE blocks to achieve "compute-on-demand" via dynamic routing.</i></p>
</div>


## 📖 Table of Contents

- [A Humble Beginning](#-a-humble-beginning-introduction)
- [Architecture](#-architecture)
- [Updates](#-updates-latest-first)
- [Main Results](#-main-results)
  - [Detection](#detection)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
- [Detection Examples](#-detection-examples)
- [Supported Tasks](#-supported-tasks)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Validation](#validation)
  - [Training](#training)
  - [Inference](#inference)
  - [Export](#export)
  - [Gradio Demo](#gradio-demo)
- [Community & Contributing](#-community--contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)



## 🚀 Updates (Latest First)

- **2025/12/30**: arXiv paper published.


## 📊 Main Results
### Detection
<div align="center">
  <img width="450" alt="Radar chart comparing YOLO models on various datasets" src="https://github.com/user-attachments/assets/743fa632-659b-43b1-accf-f865c8b66754"/>
</div>


<div align="center">
  <p><b>Table 1. Comparison with state-of-the-art Nano-scale detectors across five benchmarks.</b></p>
  <table style="border-collapse:collapse; width:100%; font-family:sans-serif; text-align:center; border-top:2px solid #000; border-bottom:2px solid #000; font-size:0.9em;">
    <thead>
      <tr style="border-bottom:1px solid #ddd;">
        <th style="padding:8px; border-right:1px solid #ddd;">Dataset</th>
        <th colspan="2" style="border-right:1px solid #ddd;">COCO</th>
        <th colspan="2" style="border-right:1px solid #ddd;">PASCAL VOC</th>
        <th colspan="2" style="border-right:1px solid #ddd;">VisDrone</th>
        <th colspan="2" style="border-right:1px solid #ddd;">KITTI</th>
        <th colspan="2" style="border-right:1px solid #ddd;">SKU-110K</th>
        <th>Efficiency</th>
      </tr>
      <tr style="border-bottom:1px solid #000;">
        <th style="padding:8px; border-right:1px solid #ddd;">Method</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>Latency<br>(ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv10</td>
        <td>38.5</td><td style="border-right:1px solid #ddd;">53.8</td>
        <td>60.6</td><td style="border-right:1px solid #ddd;">80.3</td>
        <td>18.7</td><td style="border-right:1px solid #ddd;">32.4</td>
        <td>66.0</td><td style="border-right:1px solid #ddd;">88.3</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.84</td>
      </tr>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv11-N</td>
        <td>39.4</td><td style="border-right:1px solid #ddd;">55.3</td>
        <td>61.0</td><td style="border-right:1px solid #ddd;">81.2</td>
        <td>18.5</td><td style="border-right:1px solid #ddd;">32.2</td>
        <td>67.8</td><td style="border-right:1px solid #ddd;">89.8</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.50</td>
      </tr>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv12-N</td>
        <td>40.6</td><td style="border-right:1px solid #ddd;">56.7</td>
        <td>60.7</td><td style="border-right:1px solid #ddd;">80.8</td>
        <td>18.3</td><td style="border-right:1px solid #ddd;">31.7</td>
        <td>67.6</td><td style="border-right:1px solid #ddd;">89.3</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.64</td>
      </tr>
      <tr style="border-bottom:1px solid #000;">
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv13-N</td>
        <td>41.6</td><td style="border-right:1px solid #ddd;">57.8</td>
        <td>60.7</td><td style="border-right:1px solid #ddd;">80.3</td>
        <td>17.5</td><td style="border-right:1px solid #ddd;">30.6</td>
        <td>67.7</td><td style="border-right:1px solid #ddd;">90.6</td>
        <td>57.5</td><td style="border-right:1px solid #ddd;">90.3</td>
        <td>1.97</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; text-align:left; border-right:1px solid #ddd;"><b>YOLO-Master-N</b></td>
        <td><b>42.4</b></td><td style="border-right:1px solid #ddd;"><b>59.2</b></td>
        <td><b>62.1</b></td><td style="border-right:1px solid #ddd;"><b>81.9</b></td>
        <td><b>19.6</b></td><td style="border-right:1px solid #ddd;"><b>33.7</b></td>
        <td><b>69.2</b></td><td style="border-right:1px solid #ddd;"><b>91.3</b></td>
        <td><b>58.2</b></td><td style="border-right:1px solid #ddd;"><b>90.6</b></td>
        <td><b>1.62</b></td>
      </tr>
    </tbody>
  </table>
</div>

### Segmentation

| **Model**             | **Size** | **mAPbox (%)** | **mAPmask (%)** | **Gain (mAPmask)** |
| --------------------- | -------- | -------------- | --------------- | ------------------ |
| YOLOv11-seg-N         | 640      | 38.9           | 32.0            | -                  |
| YOLOv12-seg-N         | 640      | 39.9           | 32.8            | Baseline           |
| **YOLO-Master-seg-N** | **640**  | **42.9**       | **35.6**        | **+2.8%** 🚀        |

### Classification

| **Model**             | **Dataset**  | **Input Size** | **Top-1 Acc (%)** | **Top-5 Acc (%)** | **Comparison**    |
| --------------------- | ------------ | -------------- | ----------------- | ----------------- | ----------------- |
| YOLOv11-cls-N         | ImageNet     | 224            | 70.0              | 89.4              | Baseline          |
| YOLOv12-cls-N         | ImageNet     | 224            | 71.7              | 90.5              | +1.7% Top-1       |
| **YOLO-Master-cls-N** | **ImageNet** | **224**        | **76.6**          | **93.4**          | **+4.9% Top-1** 🔥 |

## 🖼️ Detection Examples

<div align="center">
  <img width="1416" height="856" alt="Detection Examples" src="https://github.com/user-attachments/assets/0e1fbe4a-34e7-489e-b936-6d121ede5cf6" /> </div>
<table border="0"> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>Detection</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/db350acd-1d91-4be6-96b2-6bdf8aac57e8" alt="Detection 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/b6c80dbd-120e-428b-8d26-ea2b38a40b47" alt="Detection 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>Segmentation</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/edb05e3c-cd83-41db-89f8-8ef09fc22798" alt="Segmentation 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/ea138674-d7c7-48fb-b272-3ec211d161bf" alt="Segmentation 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> </table>



## 🧩 Supported Tasks

YOLO-Master builds upon the robust Ultralytics framework, inheriting support for various computer vision tasks. While our research primarily focuses on Real-Time Object Detection, the codebase is capable of supporting:

| Task | Status | Description |
|:-----|:------:|:------------|
| **Object Detection** | ✅ | Real-time object detection with ES-MoE acceleration. |
| **Instance Segmentation** | ✅ | Experimental support (inherited from Ultralytics). |
| **Pose Estimation** | 🚧 | Experimental support (inherited from Ultralytics). |
| **OBB Detection** | 🚧 | Experimental support (inherited from Ultralytics). |
| **Classification** | ✅ | Image classification support. |

## ⚙️ Quick Start

### Installation

<details open>
<summary><strong>Install via pip (Recommended)</strong></summary>

```bash
# 1. Create and activate a new environment
conda create -n yolo_master python=3.11 -y
conda activate yolo_master

# 2. Clone the repository
git clone https://github.com/isLinXu/YOLO-Master
cd YOLO-Master

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Optional: Install FlashAttention for faster training (CUDA required)
pip install flash_attn
```
</details>

### Validation

Validate the model accuracy on the COCO dataset.

```python
from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo_master_n.pt") 

# Run validation
metrics = model.val(data="coco.yaml", save_json=True)
print(metrics.box.map)  # map50-95
```

### Training

Train a new model on your custom dataset or COCO.

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolo_master.yaml')  # build a new model from YAML

# Train the model
results = model.train(
    data='coco.yaml',
    epochs=600, 
    batch=256, 
    imgsz=640,
    device="0,1,2,3", # Use multiple GPUs
    scale=0.5, 
    mosaic=1.0,
    mixup=0.0, 
    copy_paste=0.1
)
```

### Inference

Run inference on images or videos.

**Python:**
```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
results = model("path/to/image.jpg")
results[0].show()
```

**CLI:**
```bash
yolo predict model=yolo_master_n.pt source='path/to/image.jpg' show=True
```

### Export

Export the model to other formats for deployment (TensorRT, ONNX, etc.).

```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
model.export(format="engine", half=True)  # Export to TensorRT
# formats: onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
```

### Gradio Demo

Launch a local web interface to test the model interactively. This application provides a user-friendly Gradio dashboard for model inference, supporting automatic model scanning, task switching (Detection, Segmentation, Classification), and real-time visualization.

```bash
python app.py
# Open http://127.0.0.1:7860 in your browser
```

## 🤝 Community & Contributing

We welcome contributions! Please check out our [Contribution Guidelines](CONTRIBUTING.md) for details on how to get involved.

- **Issues**: Report bugs or request features [here](https://github.com/isLinXu/YOLO-Master/issues).
- **Pull Requests**: Submit your improvements.

## 📄 License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

## 🙏 Acknowledgements

This work builds upon the excellent [Ultralytics](https://github.com/ultralytics/ultralytics) framework. Huge thanks to the community for contributions, deployments, and tutorials!

## 📝 Citation

If you use YOLO-Master in your research, please cite our paper:

```bibtex
@article{lin2025yolomaster,
  title={{YOLO-Master}: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection},
  author={Lin, Xu and Peng, Jinlong and Gan, Zhenye and Zhu, Jiawen and Liu, Jun},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

⭐ **If you find this work useful, please star the repository!**
