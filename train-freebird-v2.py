import subprocess


command = [
    "yolo", "train",
    "model=freebird.pt",
    "data=voc.yaml",
    "epochs=50",
    "imgsz=640",
    "batch=16"
]


subprocess.run(command)

