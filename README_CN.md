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

[English](README.md) | [简体中文](README_CN.md)

---

## 💡 初心 (Introduction)

> **"探索 YOLO 中动态智能的前沿。"**

这项工作代表了我们对实时目标检测 (RTOD) 演进的热情探索。据我们所知，**YOLO-Master 是首个在通用数据集上将混合专家 (MoE) 架构与 YOLO 深度融合的工作。**

大多数现有的 YOLO 模型依赖于静态的密集计算——即对简单的天空背景和复杂的拥挤路口分配相同的计算预算。我们认为检测模型应该更加“自适应”，就像人类视觉系统一样。虽然这次初步探索可能并不完美，但它展示了 **高效稀疏 MoE (ES-MoE)** 在平衡高精度与超低延迟方面的巨大潜力。我们将致力于持续迭代和优化，以进一步完善这一方法。

展望未来，我们从 LLM 和 VLM 的变革性进步中汲取灵感。我们将致力于完善这一方法，并将这些见解扩展到基础视觉任务中，最终目标是解决更具雄心的前沿问题，如开放词汇检测和开放集分割。

<details>
  <summary>
  <font size="+1"><b>摘要 (Abstract)</b></font>
  </summary>
现有的实时目标检测 (RTOD) 方法通常采用类 YOLO 架构，因为它们在精度和速度之间取得了良好的平衡。然而，这些模型依赖于静态密集计算，对所有输入应用统一的处理，导致表示能力和计算资源的分配不当，例如在简单场景上过度分配，而在复杂场景上服务不足。这种不匹配导致了计算冗余和次优的检测性能。

为了克服这一限制，我们提出了 YOLO-Master，这是一种新颖的类 YOLO 框架，为 RTOD 引入了实例条件自适应计算。这是通过高效稀疏混合专家 (ES-MoE) 块实现的，该块根据场景复杂度动态地为每个输入分配计算资源。其核心是一个轻量级的动态路由网络，通过多样性增强目标指导专家在训练期间的专业化，鼓励专家之间形成互补的专业知识。此外，路由网络自适应地学习仅激活最相关的专家，从而在提高检测性能的同时，最大限度地减少推理过程中的计算开销。

在五个大规模基准测试上的综合实验证明了 YOLO-Master 的优越性。在 MS COCO 上，我们的模型实现了 42.4% 的 AP 和 1.62ms 的延迟，比 YOLOv13-N 高出 +0.8% mAP，推理速度快 17.8%。值得注意的是，在具有挑战性的密集场景中收益最为明显，同时模型在典型输入上保持了效率并维持了实时推理速度。代码: [isLinXu/YOLO-Master](https://github.com/isLinXu/YOLO-Master)
</details>

---

## 🎨 架构

<div align="center">
  <img width="90%" alt="YOLO-Master Architecture" src="https://github.com/user-attachments/assets/6caa1065-af77-4f77-8faf-7551c013dacd" />
  <p><i>YOLO-Master 引入 ES-MoE 块，通过动态路由实现“按需计算”。</i></p>
</div>

### 📚 深度文档
关于 MoE 模块的设计理念、路由机制详解以及针对不同硬件（GPU/CPU/NPU）的部署优化指南，请参阅我们的 Wiki 文档：
👉 **[Wiki: MoE 模块详解与演进](wiki/MoE_Modules_Explanation.md)**

## 📖 目录

- [初心](#-初心-introduction)
- [架构](#-架构)
- [更新](#-更新-latest-first)
- [主要结果](#-主要结果)
  - [检测](#检测)
  - [分割](#分割)
  - [分类](#分类)
- [检测示例](#-检测示例)
- [支持的任务](#-支持的任务)
- [快速开始](#-快速开始)
  - [安装](#安装)
  - [验证](#验证)
  - [训练](#训练)
  - [推理](#推理)
  - [导出](#导出)
  - [Gradio 演示](#gradio-演示)
- [社区与贡献](#-社区与贡献)
- [许可证](#-许可证)
- [致谢](#-致谢)
- [引用](#-引用)

## 🚀 更新 (Latest First)

- **2025/12/30**: arXiv 论文发布。

## 📊 主要结果
### 检测
<div align="center">
  <img width="450" alt="Radar chart comparing YOLO models on various datasets" src="https://github.com/user-attachments/assets/743fa632-659b-43b1-accf-f865c8b66754"/>
</div>


<div align="center">
  <p><b>表 1. 五个基准测试上与最先进 Nano 级检测器的比较。</b></p>
  <table style="border-collapse:collapse; width:100%; font-family:sans-serif; text-align:center; border-top:2px solid #000; border-bottom:2px solid #000; font-size:0.9em;">
    <thead>
      <tr style="border-bottom:1px solid #ddd;">
        <th style="padding:8px; border-right:1px solid #ddd;">数据集</th>
        <th colspan="2" style="border-right:1px solid #ddd;">COCO</th>
        <th colspan="2" style="border-right:1px solid #ddd;">PASCAL VOC</th>
        <th colspan="2" style="border-right:1px solid #ddd;">VisDrone</th>
        <th colspan="2" style="border-right:1px solid #ddd;">KITTI</th>
        <th colspan="2" style="border-right:1px solid #ddd;">SKU-110K</th>
        <th>效率</th>
      </tr>
      <tr style="border-bottom:1px solid #000;">
        <th style="padding:8px; border-right:1px solid #ddd;">方法</th>
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
        <th>延迟<br>(ms)</th>
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

### 分割

| **模型**             | **尺寸** | **mAPbox (%)** | **mAPmask (%)** | **增益 (mAPmask)** |
| --------------------- | -------- | -------------- | --------------- | ------------------ |
| YOLOv11-seg-N         | 640      | 38.9           | 32.0            | -                  |
| YOLOv12-seg-N         | 640      | 39.9           | 32.8            | Baseline           |
| **YOLO-Master-seg-N** | **640**  | **42.9**       | **35.6**        | **+2.8%** 🚀        |

### 分类

| **模型**             | **数据集**  | **输入尺寸** | **Top-1 Acc (%)** | **Top-5 Acc (%)** | **对比**    |
| --------------------- | ------------ | -------------- | ----------------- | ----------------- | ----------------- |
| YOLOv11-cls-N         | ImageNet     | 224            | 70.0              | 89.4              | Baseline          |
| YOLOv12-cls-N         | ImageNet     | 224            | 71.7              | 90.5              | +1.7% Top-1       |
| **YOLO-Master-cls-N** | **ImageNet** | **224**        | **76.6**          | **93.4**          | **+4.9% Top-1** 🔥 |

## 🖼️ 检测示例

<div align="center">
  <img width="1416" height="856" alt="Detection Examples" src="https://github.com/user-attachments/assets/0e1fbe4a-34e7-489e-b936-6d121ede5cf6" /> </div>
<table border="0"> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>检测</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/db350acd-1d91-4be6-96b2-6bdf8aac57e8" alt="Detection 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/b6c80dbd-120e-428b-8d26-ea2b38a40b47" alt="Detection 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>分割</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/edb05e3c-cd83-41db-89f8-8ef09fc22798" alt="Segmentation 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/ea138674-d7c7-48fb-b272-3ec211d161bf" alt="Segmentation 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> </table>



## 🧩 支持的任务

YOLO-Master 建立在强大的 Ultralytics 框架之上，继承了对各种计算机视觉任务的支持。虽然我们的研究主要集中在实时目标检测，但代码库支持：

| 任务 | 状态 | 描述 |
|:-----|:------:|:------------|
| **目标检测** | ✅ | 具有 ES-MoE 加速的实时目标检测。 |
| **实例分割** | ✅ | 实验性支持 (继承自 Ultralytics)。 |
| **姿态估计** | 🚧 | 实验性支持 (继承自 Ultralytics)。 |
| **OBB 检测** | 🚧 | 实验性支持 (继承自 Ultralytics)。 |
| **图像分类** | ✅ | 图像分类支持。 |

## ⚙️ 快速开始

### 安装

<details open>
<summary><strong>通过 pip 安装 (推荐)</strong></summary>

```bash
# 1. 创建并激活新环境
conda create -n yolo_master python=3.11 -y
conda activate yolo_master

# 2. 克隆仓库
git clone https://github.com/isLinXu/YOLO-Master
cd YOLO-Master

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 可选: 安装 FlashAttention 以加速训练 (需要 CUDA)
pip install flash_attn
```
</details>

### 验证

在 COCO 数据集上验证模型精度。

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo_master_n.pt") 

# 运行验证
metrics = model.val(data="coco.yaml", save_json=True)
print(metrics.box.map)  # map50-95
```

### 训练

在自定义数据集或 COCO 上训练新模型。

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('cfg/models/master/v0/det/yolo-master-n.yaml')  # 从 YAML 构建新模型

# 训练模型
results = model.train(
    data='coco.yaml',
    epochs=600, 
    batch=256, 
    imgsz=640,
    device="0,1,2,3", # 使用多 GPU
    scale=0.5, 
    mosaic=1.0,
    mixup=0.0, 
    copy_paste=0.1
)
```

### 推理

对图像或视频进行推理。

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

### 导出

将模型导出为其他格式以进行部署 (TensorRT, ONNX 等)。

```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
model.export(format="engine", half=True)  # 导出为 TensorRT
# 格式: onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
```

### Gradio 演示

启动本地 Web 界面以交互式测试模型。此应用程序提供了一个用户友好的 Gradio 仪表板，用于模型推理，支持自动模型扫描、任务切换（检测、分割、分类）和实时可视化。

```bash
python app.py
# 在浏览器中打开 http://127.0.0.1:7860
```

## 🤝 社区与贡献

我们欢迎贡献！有关如何参与的详细信息，请查看我们的 [贡献指南](CONTRIBUTING.md)。

- **Issues**: 在 [这里](https://github.com/isLinXu/YOLO-Master/issues) 报告错误或请求功能。
- **Pull Requests**: 提交您的改进。

## 📄 许可证

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) 许可证。

## 🙏 致谢

这项工作建立在优秀的 [Ultralytics](https://github.com/ultralytics/ultralytics) 框架之上。非常感谢社区的贡献、部署和教程！

## 📝 引用

如果您在研究中使用 YOLO-Master，请引用我们的论文：

```bibtex
@article{lin2025yolomaster,
  title={{YOLO-Master}: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection},
  author={Lin, Xu and Peng, Jinlong and Gan, Zhenye and Zhu, Jiawen and Liu, Jun},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

⭐ **如果您觉得这项工作有用，请给仓库点个星！**
