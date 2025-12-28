import pandas as pd
import numpy as np
from DTWAnalysis import DTW

def normalize(signal):
    signal = np.asarray(signal)
    return (signal - np.mean(signal)) / np.std(signal)



def EvaluateExercise(exerciseName: str, exerciseDataFrame: pd.DataFrame):
    referenceDataFrame, evaluation_angles = None, None
    if(exerciseName == "Squat"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/SquatSideWaysReference.csv")
        evaluation_angles = ["right_armpit_angle", "right_shoulder_angle", "trunk_angle", "right_knee_angle"]
    
    elif(exerciseName == "LegPush"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/LegPushSideWaysReference.csv")
        evaluation_angles = ["left_knee_angle", "trunk_angle"]
     
    elif(exerciseName == "BicepCurl"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/BicepCurlSideWaysReference.csv")
        evaluation_angles = ["right_elbow_angle", "right_armpit_angle", "right_shoulder_angle"]
    
    elif(exerciseName == "PreacherCurl"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/PreacherCurlSideWaysReference.csv")
        evaluation_angles = ["left_elbow_angle", "left_armpit_angle", "left_shoulder_angle"]
    
    errors = []
    for evaluation_angle in evaluation_angles:
        referenceTimeSeries = referenceDataFrame[evaluation_angle].to_numpy()
        exerciseTimeSeries = exerciseDataFrame[evaluation_angle].to_numpy()

        normalized_referenceTimeSeries = normalize(referenceTimeSeries)
        normalized_exerciseTimeSeries = normalize(exerciseTimeSeries)

        error, optimal_path, costMatrix = DTW(normalized_referenceTimeSeries, normalized_exerciseTimeSeries)
        avg_error = error / len(optimal_path)

        errors.append(avg_error)
    errors = np.array(errors)
    print(errors)
    error = np.mean(errors)
    return error, evaluation_angles