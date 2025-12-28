import pandas as pd
import numpy as np
from DTWAnalysis import DTW

def normalize(signal):
    signal = np.asarray(signal)
    return (signal - np.mean(signal)) / np.std(signal)



def EvaluateExersice(exersiceName: str, exersiceDataFrame: pd.DataFrame):
    if(exersiceName == "Squat"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/SquatSideWaysReference.csv")
        evaluation_angles = ["right_armpit_angle", "right_shoulder_angle", "trunk_angle", "right_knee_angle"]
        errors = []
        for evaluation_angle in evaluation_angles:
            referenceTimeSeries = referenceDataFrame[evaluation_angle].to_numpy()
            exersiceTimeSeries = exersiceDataFrame[evaluation_angle].to_numpy()

            error, optimal_path, costMatrix = DTW(referenceTimeSeries, exersiceTimeSeries)
            avg_error = error / len(optimal_path)

            errors.append(avg_error)
        errors = np.array(errors)
        print(errors)
        error = (np.sum(errors**2) / len(errors)) / 169458.8500016115
        return error, evaluation_angles