from ultralytics import YOLO, settings

settings.update({'runs_dir': './runs/'})
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=100, imgsz=(640//4,480//4), device=[0, 1], batch=32)