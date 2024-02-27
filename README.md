# 3d Object Detection Using MediaPipe

This Python code utilizes the MediaPipe library to perform simultaneous object detection for shoes, chairs, and cups, face detection, and pose detection using the Objectron, FaceMesh, and Pose models from MediaPipe. The code captures video from the default camera using OpenCV and processes each frame to detect and draw landmarks for different objects.

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


