import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import time

# Objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

# Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Create separate instances of Objectron for each object
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.8,
                            model_name='Shoe') as shoe_objectron, \
        mp_objectron.Objectron(static_image_mode=False,
                               max_num_objects=5,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.8,
                               model_name='Chair') as chair_objectron, \
        mp_objectron.Objectron(static_image_mode=False,
                               max_num_objects=5,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.8,
                               model_name='Cup') as cup_objectron:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Face Detection
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as face_mesh:
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(frame_rgb)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Pose Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if detection_results.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                detection_results.pose_landmarks.landmark
            ])
            mp_drawing.draw_landmarks(frame, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing_styles.get_default_pose_landmarks_style())

        # Process each objectron instance
        results_shoe = shoe_objectron.process(frame)
        results_chair = chair_objectron.process(frame)
        #results_keyboard = keyboard_objectron.process(frame)
        results_cup = cup_objectron.process(frame)

        # Draw detected objects for each instance
        for results, model_name in zip([results_shoe, results_chair, results_cup], ['Shoe', 'Chair', 'Cup']):
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)
                    cv2.putText(frame, model_name, (int(detected_object.translation[0]), int(detected_object.translation[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Combined Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
