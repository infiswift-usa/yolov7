import io
import time
import base64
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

# Imports from YOLOv7 codebase
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords, 
                           xyxy2xywh, set_logging)
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box

app = FastAPI()

# Global variables for the persistent model and config.
model = None
device = None
imgsz = 640*6            # desired inference image size
half = False
names = None
stride = None

# Configuration parameters (adjust as needed)
weights = "yolov7_custom/runs/train/yolov7-custom26/weights/best.pt"
conf_thres = 0.25
iou_thres = 0.45
trace_model = True
augment = False

@app.on_event("startup")
def load_model():
    """Load the YOLO model into memory (once) on startup."""
    global model, device, imgsz, half, names, stride
    set_logging()
    device = select_device("")  # use default device (or set "cpu" explicitly)
    # Load FP32 model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    
    # Optionally trace the model (static TorchScript) to speed up inference.
    if trace_model:
        model = TracedModel(model, device, imgsz)
        
    if device.type != "cpu":
        model.half()  # to FP16
        half = True
    else:
        half = False
        
    names = model.module.names if hasattr(model, "module") else model.names
    print("Model loaded on", device, "with image size", imgsz)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Endpoint to run detection on an uploaded image.
    Expects an image file; returns detection results and an annotated image (base64 encoded).
    """
    try:
        # Read image file and decode using OpenCV.
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img0 is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})
        
        # Preprocess image: letterbox to maintain aspect ratio.
        img = letterbox(img0, imgsz, stride=stride)[0]
        # Convert BGR to RGB and reformat to channel-first
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # convert to FP16 or FP32
        img /= 255.0  # normalize to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        t1 = time.time()
        # Inference
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        t2 = time.time()
        # Non-Maximum Suppression
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        detections = []
        # Process detections (assuming one image in the batch)
        for det in pred:
            if len(det):
                # Rescale coordinates from img size to original image size.
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # Annotate image with bounding box and label.
                    plot_one_box(xyxy, img0, label=label, color=[0, 255, 0], line_thickness=2)
                    # Convert box from xyxy to xywh format.
                    box = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    detections.append({
                        "class": int(cls),
                        "confidence": float(conf),
                        "box": box
                    })
        t3 = time.time()
        
        # Encode annotated image to JPEG and then base64.
        _, im_arr = cv2.imencode('.jpg', img0)
        im_bytes = im_arr.tobytes()
        im_base64 = base64.b64encode(im_bytes).decode('utf-8')
        
        result = {
            "detections": detections,
            "inference_time_ms": (t2 - t1) * 1000,
            "nms_time_ms": (t3 - t2) * 1000,
            "annotated_image": im_base64
        }
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)
