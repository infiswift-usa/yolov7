import torch
from pathlib import Path
from models.experimental import attempt_load

weights = "runs/train/yolov7-custom26/weights/best.pt"
m = attempt_load(weights, map_location="cpu")
names = m.module.names if hasattr(m, "module") else m.names
print(names)          # ['column', 'beam', 'conduit', 'valve']
Path("class_names.txt").write_text("\n".join(names))
