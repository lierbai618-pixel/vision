import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def train_s():
    from ultralytics import YOLO
    print("=== Training yolov8s (90 classes, for realtime) ===")
    model = YOLO("yolov8s.pt")
    model.train(
        data="configs/merged_90class.yaml",
        epochs=10, imgsz=640, batch=8, fraction=0.1,
        project="runs/train", name="unified_s",
        patience=10, workers=0, verbose=True, augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1,
    )
    print("yolov8s done!")

def train_m():
    from ultralytics import YOLO
    print("=== Training yolov8m (90 classes, for accuracy) ===")
    model = YOLO("yolov8m.pt")
    model.train(
        data="configs/merged_90class.yaml",
        epochs=10, imgsz=640, batch=8, fraction=0.1,
        project="runs/train", name="unified_m",
        patience=10, workers=0, verbose=True, augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1,
    )
    print("yolov8m done!")

if __name__ == '__main__':
    train_s()
    train_m()
    print("All training complete!")