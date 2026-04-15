import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("runs/segment/train3/weights/best.pt")

# Open the video file
video_path = "videos/test2.mp4"
cap = cv2.VideoCapture(video_path)
# 保存成视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "outputs/test2_v5.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))
)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, conf=0.60)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        # cv2.imshow("YOLO Inference", annotated_frame)
        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
