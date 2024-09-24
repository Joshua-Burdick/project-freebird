import cv2 as cv

cap = cv.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a selfie-view display.
    cv.imshow('FREEBIRD', cv.flip(image, 1))
    if cv.waitKey(5) & 0xFF == 27:
      break