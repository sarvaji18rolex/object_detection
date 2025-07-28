import cv2
import torch

# Load YOLOv5 pre-trained model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
model.conf = 0.5  # Set confidence threshold

# Start video capture from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame from BGR (OpenCV format) to RGB (PyTorch format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(rgb_frame)

    # Extract detections as pandas DataFrame
    detections = results.pandas().xyxy[0]

    # Draw bounding boxes and labels on the frame
    for _, detection in detections.iterrows():
        xmin = int(detection['xmin'])
        ymin = int(detection['ymin'])
        xmax = int(detection['xmax'])
        ymax = int(detection['ymax'])
        label = detection['name']
        confidence = float(detection['confidence'])

        # Draw rectangle and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Real-Time Object Detection', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
