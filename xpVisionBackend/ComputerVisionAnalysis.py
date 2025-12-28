import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

    "neck": ("head", "neck", "shoulder_center"),
    "trunk": ("shoulder_center", "hip_center", "Sacrum")
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

def get_point(landmarks, name):
    idx = JOINTS_IDX[name]
    return np.array([landmarks[idx].x, landmarks[idx].y])

def GetBodyPartsAnglesFromLandmark(landmarks, time_stamp):
    raw_data = []

    # Virtual joints
    shoulder_center = np.mean([
        get_point(landmarks, "left_shoulder"),
        get_point(landmarks, "right_shoulder")
    ], axis=0)

    hip_center = np.mean([
        get_point(landmarks, "left_hip"),
        get_point(landmarks, "right_hip")
    ], axis=0)

    sacrum = np.array([hip_center[0], hip_center[1]-0.1])
    virtual = {
        "shoulder_center": shoulder_center,
        "hip_center": hip_center,
        "Sacrum": sacrum

    }

    for angle_name, (a, b, c) in ANGLE_DEFS.items():

        p1 = virtual[a] if a in virtual else get_point(landmarks, a)
        p2 = virtual[b] if b in virtual else get_point(landmarks, b)
        p3 = virtual[c] if c in virtual else get_point(landmarks, c)

        theta = calculate_angle(p1, p2, p3)
        raw_data.append([time_stamp, angle_name, theta])

    return raw_data

def GenerateAnnotatedFrame(landmarks, frame):

    POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(24,26),(26,28),
    (11,12),(23,24),(11,23),(12,24),(0,11),(0,12)
    ]

    h, w, c = frame.shape
    annotated_frame = frame.copy()
    for i, j in POSE_CONNECTIONS:
        x1, y1 = int(landmarks[i].x * w), int(landmarks[i].y * h)
        x2, y2 = int(landmarks[j].x * w), int(landmarks[j].y * h)
        cv2.line(annotated_frame, (x1,y1), (x2,y2), (255,255,255), 2)

    for p in landmarks:
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(annotated_frame, (x,y), 4, (0,255,0), -1)

    return annotated_frame

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
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "annotated_video.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

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

        limbs_data = GetBodyPartsAnglesFromLandmark(landmarks, time_stamp)
        annotated_frame = GenerateAnnotatedFrame(landmarks, frame)
        out.write(annotated_frame)
        raw_data.extend(limbs_data)

        frame_id+=1

    cap.release()
    out.release()
    pose_landmarker.close()
    return raw_data

def ExtractCSVDataFromLandmarkAngles(raw_data: list):
    df_angles = pd.DataFrame(raw_data, columns=["time","limb","theta"])
    df_series_dictionary = {}
    df_series_dictionary["time"] = df_angles["time"].unique()
    for limb in df_angles["limb"].unique():
      df_series_dictionary[limb + "_angle"] = df_angles[df_angles["limb"] == limb]["theta"].to_numpy()
    
    df_output = pd.DataFrame(df_series_dictionary)
    return df_output