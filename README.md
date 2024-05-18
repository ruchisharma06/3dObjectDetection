# 3d Object Detection Using MediaPipe

This Python code utilizes the MediaPipe library to perform simultaneous object detection for shoes, chairs, and cups, face detection, and pose detection using the Objectron, FaceMesh, and Pose models from MediaPipe. The code captures video from the default camera using OpenCV and processes each frame to detect and draw landmarks for different objects.

<img width="313" alt="Screen Shot 2024-05-18 at 3 03 01 PM" src="https://github.com/ruchisharma06/3dObjectDetection/assets/116240606/897b891a-3368-4445-b077-df0add7867e7">


# Main Loop

The code runs a loop to continuously capture frames from the camera and process each frame.

Face Detection:
Uses FaceMesh to detect and draw facial landmarks.
Pose Detection:
Uses Pose model to detect and draw body pose landmarks.
Objectron Detection:
Processes each Objectron instance (shoe, chair, cup) and draws detected objects with landmarks, rotation, and translation information.

# Note

This code provides a real-time demonstration of object detection for shoes, chairs, and cups, along with facial and body pose landmarks, making it a versatile and informative application.


