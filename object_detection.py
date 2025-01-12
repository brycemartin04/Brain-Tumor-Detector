from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="/Users/brycemartin/Documents/AI/YOLO/BrainTumorYolov9/data.yaml",  # path to dataset YAML
    epochs=50,  # number of training epochs
    imgsz=640,  # training image size
    patience=20,
    batch=32,
    device="mps",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/Users/brycemartin/Documents/AI/YOLO/BrainTumorYolov9/test/images/2728_jpg.rf.b04df2c64856f68a0cf4c075ff0539e5.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model