from ultralytics import YOLO
from PIL import Image

model = YOLO("yolo11n.pt")

results = model('tank-test.jpg', show=True)
# Access the first result in the results list and plot the annotated image
annotated_image = results[0].plot()

# Convert the image to a PIL Image object if needed and save it
annotated_image_pil = Image.fromarray(annotated_image)
annotated_image_pil.save("tank-output.jpg")
