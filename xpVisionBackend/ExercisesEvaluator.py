import pandas as pd
import numpy as np
from DTWAnalysis import DTW, FindDistance

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
        evaluation_angles = ["left_knee_angle"]
     
    elif(exerciseName == "BicepCurl"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/BicepCurlSideWaysReference.csv")
        evaluation_angles = ["right_elbow_angle", "right_armpit_angle", "right_shoulder_angle"]
    
    elif(exerciseName == "PreacherCurl"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/PreacherCurlSideWaysReference.csv")
        evaluation_angles = ["left_elbow_angle", "left_armpit_angle", "left_shoulder_angle"]
    
    signals_errors = []
    optimal_paths = []
    error_signals = []
    #O(EvaluationAngles * OptimalIndicies)
    for evaluation_angle in evaluation_angles:
        referenceTimeSeries = referenceDataFrame[evaluation_angle].to_numpy()
        exerciseTimeSeries = exerciseDataFrame[evaluation_angle].to_numpy()

        normalized_referenceTimeSeries = normalize(referenceTimeSeries)
        normalized_exerciseTimeSeries = normalize(exerciseTimeSeries)

        signalError, optimal_indicies, costMatrix = DTW(normalized_referenceTimeSeries, normalized_exerciseTimeSeries)

        errorSignal = []
        for (i, j) in optimal_indicies:
            pointError = FindDistance(normalized_referenceTimeSeries[i], normalized_exerciseTimeSeries[j])
            errorSignal.append(pointError)
        
        error_signals.append(errorSignal)
        print(optimal_indicies)
        avg_error = signalError / len(optimal_indicies)
        optimal_paths.append(optimal_indicies)

        signals_errors.append(avg_error)
    signals_errors = np.array(signals_errors)
    signalError = np.mean(signals_errors)
    return signalError, evaluation_angles, optimal_paths, error_signals