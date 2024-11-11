from ultralytics import YOLO

model = YOLO("freebird.pt")

results = model('test-footage-2.mp4', show=True)