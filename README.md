# xpVision Backend

> High-performance computational backend for real-time exercise form evaluation using computer vision and Dynamic Time Warping

## Overview

The xpVision backend is a FastAPI-based server that powers intelligent exercise form analysis. It processes user-uploaded workout videos, extracts biomechanical features using Google MediaPipe Pose, and compares them against professional reference movements using Dynamic Time Warping (DTW) algorithms to provide accuracy scores and corrective feedback.

## Architecture

The backend implements a modular three-stage pipeline:

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Video Upload   │ ───> │  Pose Extraction │ ───> │  DTW Analysis   │
│   (FastAPI)     │      │   (MediaPipe)    │      │  (Evaluation)   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         │                        │                         │
         │                        ▼                         ▼
         │                ┌──────────────┐         ┌──────────────┐
         │                │ 33 Landmarks │         │ Similarity   │
         │                │ per Frame    │         │ Score + Path │
         │                └──────────────┘         └──────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Response: Accuracy Score, Alignment Graphs, Error Signals      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **FastAPI Server** (`Backend.py`)
   - Asynchronous request handling
   - Video upload processing
   - Response serialization

2. **Computer Vision Pipeline** (`ComputerVisionAnalysis.py`)
   - MediaPipe Pose integration
   - 33 skeletal landmark extraction
   - Joint angle computation
   - Annotated video generation

3. **Exercise Evaluator** (`ExercisesEvaluator.py`)
   - Exercise-specific evaluation logic
   - Multi-angle comparison
   - Signal normalization

4. **DTW Analysis** (`DTWAnalysis.py`)
   - Dynamic programming implementation: O(N×M)
   - Optimal alignment path computation
   - Point-wise error calculation

## Technology Stack

- **Framework**: FastAPI (Python 3.8+)
- **Computer Vision**: Google MediaPipe Pose
- **Numerical Computing**: NumPy, Pandas
- **Video Processing**: OpenCV (cv2)
- **Algorithm**: Dynamic Time Warping (DTW)

## API Endpoints

### 1. DTW Analysis (Testing)
```http
POST /
Content-Type: application/json

{
  "signal1": [1, 2, 3, 4, 5],
  "signal2": [1, 3, 4, 5, 6]
}
```

**Response:**
```json
[cost, optimal_path, cost_matrix]
```

### 2. Video Upload & Exercise Evaluation
```http
POST /uploadvideo
Content-Type: multipart/form-data

exerciseName: "Squat" | "LegPush" | "BicepCurl" | "PreacherCurl"
file: <video_file>
```

**Supported Video Formats:**
`.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`

**Success Response:**
```json
{
  "message": "Success",
  "exerciseAccuracy": 87.5,
  "parametersOfPlotting": ["right_knee_angle", "trunk_angle"],
  "optimal_indicies": [[[0,0], [1,1], [2,3], ...]],
  "error_signals": [[0.12, 0.15, 0.08, ...]]
}
```

**Error Responses:**
- `400`: Invalid file type or no exercise detected
- `500`: Server processing error

### 3. Static Files
```http
GET /static/{filename}
```
Access generated annotated videos and CSV files.

## Project Structure

```
xpvision-backend/
├── Backend.py                    # FastAPI server & endpoints
├── ComputerVisionAnalysis.py     # MediaPipe pose processing
├── ExercisesEvaluator.py         # Exercise comparison logic
├── DTWAnalysis.py                # Dynamic Time Warping algorithm
├── pose_landmarker_lite.task     # MediaPipe model file
├── ExercisesReferences/          # Reference exercise data
│   ├── SquatSideWaysReference.csv
│   ├── LegPushSideWaysReference.csv
│   ├── BicepCurlSideWaysReference.csv
│   └── PreacherCurlSideWaysReference.csv
├── static/                       # Generated output files
│   ├── annotated_video.mp4
│   └── exerciseDataFrame.csv
└── requirements.txt
```

## How It Works

### 1. Pose Landmark Extraction

For each video frame, MediaPipe extracts 33 3D landmarks:

```
Φ_f = { (x_i, y_i, z_i, v_i) | i ∈ {shoulder, elbow, wrist, hip, knee, ankle, ...} }
```

Where `v_i` represents visibility confidence (0-1).

### 2. Biomechanical Feature Engineering

The system computes **14 joint angles** from landmarks:

**Upper Body:**
- Armpit angles (left/right)
- Shoulder angles (left/right)
- Elbow angles (left/right)

**Lower Body:**
- Hip angles (left/right)
- Groin angles (left/right)
- Knee angles (left/right)

**Trunk:**
- Neck angle
- Trunk angle

**Angle Calculation:**
```python
θ = arccos((BA · BC) / (||BA|| × ||BC||))
```

### 3. Dynamic Time Warping

**Algorithm Complexity:** O(N × M)

```python
# Initialize cost matrix
C[i,j] = distance(reference[i], user[j]) + min(
    C[i-1, j-1],  # diagonal
    C[i-1, j],    # vertical
    C[i, j-1]     # horizontal
)

# Backtrack for optimal alignment path
path = backtrack(C, N-1, M-1)
```

### 4. Similarity Scoring

```python
# Normalize signals
signal_norm = (signal - μ) / σ

# Compute average error
error = DTW_cost / len(optimal_path)

# Calculate accuracy
accuracy = (1 - error) × 100
```

## Supported Exercises

| Exercise | Evaluated Angles | Reference File |
|----------|------------------|----------------|
| **Squat** | Right armpit, shoulder, trunk, knee | `SquatSideWaysReference.csv` |
| **Leg Push** | Left knee | `LegPushSideWaysReference.csv` |
| **Bicep Curl** | Right elbow, armpit, shoulder | `BicepCurlSideWaysReference.csv` |
| **Preacher Curl** | Left elbow, armpit, shoulder | `PreacherCurlSideWaysReference.csv` |

## Configuration

### MediaPipe Settings
Located in `ComputerVisionAnalysis.py`:

```python
min_pose_detection_confidence = 0.5  # Adjust detection threshold
min_tracking_confidence = 0.5        # Adjust tracking threshold
```

### Adding New Exercises

1. Record reference exercise video
2. Process to generate CSV with angle time series
3. Add to `ExercisesReferences/` directory
4. Update `ExercisesEvaluator.py`:

```python
elif(exerciseName == "NewExercise"):
    referenceDataFrame = pd.read_csv("ExercisesReferences/NewExerciseReference.csv")
    evaluation_angles = ["angle1", "angle2", ...]
```

## Performance Characteristics

- **Pose Extraction:** Real-time capable on CPU (30+ FPS)
- **DTW Computation:** O(N×M) where N, M are sequence lengths
- **Memory:** Scales linearly with video length
- **Latency:** ~2-5 seconds for 10-second video on modern hardware
