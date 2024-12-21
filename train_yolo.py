from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8s.yaml')  # build a new model from YAML
    #model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(data='custom.yaml', epochs=100, imgsz=640, device=0, fliplr=0.5,flipud=0.5, batch=16, plots=False, translate=0, scale=0.5, copy_paste=0.3, pretrained=True, degrees=180, workers=8, lr0=0.01, optimizer='SGD')
