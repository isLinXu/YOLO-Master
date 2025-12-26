<div align="center">
  <h1>YOLO-MASTER</h1>
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

<img width="885" height="194" alt="Image" src="https://github.com/user-attachments/assets/6caa1065-af77-4f77-8faf-7551c013dacd" />

## 🚀 Updates (Latest First)

- **2025/12/xx**: arXiv paper published; 

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Existing Real-Time Object Detection (RTOD) methods commonly adopt YOLO-like architectures for their favorable trade-off between accuracy and speed. However, these models rely on static dense computation that applies uniform processing to all inputs, misallocating representational capacity and computational resources such as over-allocating on trivial scenes while under-serving complex ones. This mismatch results in both computational redundancy and suboptimal detection performance.
To overcome this limitation, we propose YOLO-Master, a novel YOLO-like framework that introduces instance-conditional adaptive computation for RTOD. This is achieved through an Efficient Sparse Mixture-of-Experts (ES-MoE) block that dynamically allocates computational resources to each input according to its scene complexity. At its core, a lightweight dynamic routing network guides expert specialization during training through a diversity enhancing objective, encouraging complementary expertise among experts. Additionally, the routing network adaptively learns to activate only the most relevant experts, thereby improving detection performance while minimizing computational overhead during inference.
Comprehensive experiments on five large-scale benchmarks demonstrate the superiority of YOLO-Master. On MS COCO, our model achieves 42.4\% AP with 1.62ms latency, outperforming YOLOv13-N by +0.8\% mAP and 17.8\% faster inference. Notably, the gains are most pronounced on challenging dense scenes, while the model preserves efficiency on typical inputs and maintains real-time inference speed. Code: \href{https://github.com/isLinXu/YOLO-Master}{isLinXu/YOLO-Master}
</details>
## 📊 Main Results

<img width="400" alt="Radar chart comparing YOLO models on various datasets" src="https://github.com/user-attachments/assets/743fa632-659b-43b1-accf-f865c8b66754"/>



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
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv10-N [37]</td>
        <td>38.5</td><td style="border-right:1px solid #ddd;">53.8</td>
        <td>60.6</td><td style="border-right:1px solid #ddd;">80.3</td>
        <td>18.7</td><td style="border-right:1px solid #ddd;">32.4</td>
        <td>66.0</td><td style="border-right:1px solid #ddd;">88.3</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.84</td>
      </tr>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv11-N [16]</td>
        <td>39.4</td><td style="border-right:1px solid #ddd;">55.3</td>
        <td>61.0</td><td style="border-right:1px solid #ddd;">81.2</td>
        <td>18.5</td><td style="border-right:1px solid #ddd;">32.2</td>
        <td>67.8</td><td style="border-right:1px solid #ddd;">89.8</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.50</td>
      </tr>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv12-N [35]</td>
        <td>40.6</td><td style="border-right:1px solid #ddd;">56.7</td>
        <td>60.7</td><td style="border-right:1px solid #ddd;">80.8</td>
        <td>18.3</td><td style="border-right:1px solid #ddd;">31.7</td>
        <td>67.6</td><td style="border-right:1px solid #ddd;">89.3</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.64</td>
      </tr>
      <tr style="border-bottom:1px solid #000;">
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv13-N [18]</td>
        <td>41.6</td><td style="border-right:1px solid #ddd;">57.8</td>
        <td>60.7</td><td style="border-right:1px solid #ddd;">80.3</td>
        <td>17.5</td><td style="border-right:1px solid #ddd;">30.6</td>
        <td>67.7</td><td style="border-right:1px solid #ddd;">90.6</td>
        <td>57.5</td><td style="border-right:1px solid #ddd;">90.3</td>
        <td>1.97</td>
      </tr>
      <tr style="background-color:#f9f9f9; font-weight:bold;">
        <td style="padding:8px; text-align:left; border-right:1px solid #ddd;">YOLO-Master-N</td>
        <td>42.4</td><td style="border-right:1px solid #ddd;">59.2</td>
        <td>62.1</td><td style="border-right:1px solid #ddd;">81.9</td>
        <td>19.6</td><td style="border-right:1px solid #ddd;">33.7</td>
        <td>69.2</td><td style="border-right:1px solid #ddd;">91.3</td>
        <td>58.2</td><td style="border-right:1px solid #ddd;">90.6</td>
        <td>1.62</td>
      </tr>
    </tbody>
  </table>
</div>



## 🖼️ Detection Examples

<img width="1416" height="856" alt="Image" src="https://github.com/user-attachments/assets/0e1fbe4a-34e7-489e-b936-6d121ede5cf6" />

## ⚙️ Quick Start

### Installation

```bash
# Optional: Install FlashAttention for faster training/inference (CUDA required)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-*.whl

# Create environment and install
conda create -n yolo_master python=3.11 -y
conda activate yolo_master
git clone https://github.com/isLinXu/YOLO-Master
cd YOLO-Master
pip install -r requirements.txt
pip install -e .
```

### Validation

```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")  # or s/m/l/x
metrics = model.val(data="coco.yaml", save_json=True)
```

### Training

```python
from ultralytics import YOLO

model = YOLO('yolo_master.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

```

### Inference

```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
results = model("path/to/image.jpg")
results[0].show()  # Display results
```

### Export

```python
model.export(format="engine", half=True)  # TensorRT
# or format="onnx", "openvino", etc.
```

### Gradio Demo

```bash
python app.py
# Open http://127.0.0.1:7860 in your browser
```

## 🙏 Acknowledgements

This work builds upon the excellent [Ultralytics](https://github.com/ultralytics/ultralytics) framework. Huge thanks to the community for contributions, deployments, and tutorials!

## 📝 Citation

```bibtex
@article{lin2025yolomaster,
  title={{YOLO-Master}: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection},
  author={Lin, Xu and Peng, Jinlong and Gan, Zhenye and Zhu, Jiawen and Liu, Jun},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

⭐ If you find this work useful, please star the repository!
