import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LIMBS = {
    "left_arm": ["left_shoulder","left_elbow","left_wrist"],
    "right_arm": ["right_shoulder","right_elbow","right_wrist"],
    "left_leg": ["left_hip","left_knee","left_ankle"],
    "right_leg": ["right_hip","right_knee","right_ankle"],
    "trunk": ["left_shoulder","right_shoulder","left_hip","right_hip"],
    "neck": ["head","neck"]
}

JOINTS_IDX = {
    "left_shoulder":11, "right_shoulder":12,
    "left_elbow":13, "right_elbow":14,
    "left_wrist":15, "right_wrist":16,
    "left_hip":23, "right_hip":24,
    "left_knee":25, "right_knee":26,
    "left_ankle":27, "right_ankle":28,
    "head":0, "neck":1
}


ANGLE_DEFS = {
    # Upper body
    "left_armpit":  ("left_elbow", "left_shoulder", "left_hip"),
    "right_armpit": ("right_elbow", "right_shoulder", "right_hip"),

    "left_shoulder":  ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),

    "left_elbow":  ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),

    # Pelvis / lower body
    "left_hip":  ("left_shoulder", "left_hip", "right_hip"),
    "right_hip": ("right_shoulder", "right_hip", "left_hip"),

    "left_groin":  ("left_knee", "left_hip", "right_hip"),
    "right_groin": ("right_knee", "right_hip", "left_hip"),

    "left_knee":  ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))

def get_position(theta, rom_min, rom_max):
    n = (theta - rom_min)/(rom_max - rom_min)
    if n < 0.33: return "LOW"
    elif n < 0.66: return "MID"
    else: return "HIGH"

def get_phase(omega, eps=2):
    if omega > eps: return "UP"
    elif omega < -eps: return "DOWN"
    else: return "HOLD"

def classify_pose(theta, omega, rom_min=0, rom_max=180):
    position = get_position(theta, rom_min, rom_max)
    phase = get_phase(omega)
    acc = 0.95 if position!="LOW" else 0.9
    return position + "_" + phase, acc



def ExtractLandmarkAngles(video_path: str):
    raw_data = []
    pose_seq = []
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    dt = 1 / fps
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((frame_id/fps)*1000)

        result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            frame_id +=1
            continue

        landmarks = result.pose_landmarks[0]

        time_stamp = frame_id/fps

        frame_id+=1

    cap.release()
    pose_landmarker.close()


def GetBodyPartsAnglesFromLandmark(landmarks, time_stamp):
    raw_data = []

    for limb in ["left_arm", "right_arm", "left_leg", "right_leg"]:
        if limb == "left_arm":
            a, b, c = JOINTS_IDX["left_shoulder"], JOINTS_IDX["left_elbow"], JOINTS_IDX["left_wrist"]
        elif limb == "right_arm":
            a, b, c = JOINTS_IDX["right_shoulder"], JOINTS_IDX["right_elbow"], JOINTS_IDX["right_wrist"]
        elif limb == "left_leg":
            a, b, c = JOINTS_IDX["left_hip"], JOINTS_IDX["left_knee"], JOINTS_IDX["left_ankle"]
        elif limb == "right_leg":
            a, b, c = JOINTS_IDX["right_hip"], JOINTS_IDX["right_knee"], JOINTS_IDX["right_ankle"]

        p1 = [landmarks[a].x, landmarks[a].y]
        p2 = [landmarks[b].x, landmarks[b].y]
        p3 = [landmarks[c].x, landmarks[c].y]

        theta = calculate_angle(p1, p2, p3)
        raw_data.append([time_stamp, limb, theta])

    # Trunk angle
    shoulder_mid = np.mean(
        [[landmarks[11].x, landmarks[11].y],
         [landmarks[12].x, landmarks[12].y]],
        axis=0
    )

    hip_mid = np.mean(
        [[landmarks[23].x, landmarks[23].y],
         [landmarks[24].x, landmarks[24].y]],
        axis=0
    )

    trunk_theta = calculate_angle(
        shoulder_mid, hip_mid, [hip_mid[0], hip_mid[1] - 0.1]
    )
    raw_data.append([time_stamp, "trunk", trunk_theta])

    # Neck angle
    head = [landmarks[0].x, landmarks[0].y]
    neck = [landmarks[1].x, landmarks[1].y]
    neck_theta = calculate_angle(head, neck, shoulder_mid)
    raw_data.append([time_stamp, "neck", neck_theta])

    return raw_data

