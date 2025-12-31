import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO  # noqa: E402


def main():
    model = YOLO("ultralytics/cfg/models/master/master_exp.yaml")
    model.train(data="ultralytics/cfg/datasets/coco8.yaml", epochs=1, imgsz=640, batch=4, device="cpu")
    model.val()


if __name__ == "__main__":
    main()
