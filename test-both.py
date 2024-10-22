import cv2
from ultralytics import YOLO


model1 = YOLO("freebird-voc.pt")
model2 = YOLO("freebird.pt")








video_path = 'test-footage.mp4'
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  
    
    results1 = model1.predict(frame, save=False, save_txt=False, show=False)
    results2 = model2.predict(frame, save=False, save_txt=False, show=False)

    
    annotated_frame1 = results1[0].plot()  
    annotated_frame2 = results2[0].plot()  

  
    combined_frame = cv2.addWeighted(annotated_frame1, 0.5, annotated_frame2, 0.5, 0)

   
    cv2.imshow('Combined Inference', combined_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
