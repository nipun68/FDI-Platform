import cv2
import mediapipe as mp
import numpy as np

class CVPostureAnalyzer:
    """Analyzes video frames to extract run-up angle and body lean using MediaPipe."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def analyze_frame(self, frame):
        """Returns body lean angle and posture features."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract left shoulder and hip to calculate body lean
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Calculate lean angle relative to vertical axis
            dx = left_shoulder.x - left_hip.x
            dy = left_shoulder.y - left_hip.y
            lean_angle = np.degrees(np.arctan2(dx, dy))
            return {"cv_body_lean_angle": lean_angle, "posture_detected": True}
            
        return {"cv_body_lean_angle": 0.0, "posture_detected": False}

# Example usage for production:
# analyzer = CVPostureAnalyzer()
# cap = cv2.VideoCapture("penalty_clip.mp4")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     features = analyzer.analyze_frame(frame)
#     print(f"Body Lean Angle: {features['cv_body_lean_angle']:.2f} degrees")
# 
# cap.release()     