import pandas as pd
import numpy as np
from DTWAnalysis import DTW

def EvaluateExersice(exersiceName: str, exersiceDataFrame: pd.DataFrame):
    if(exersiceName == "Squat"):
        referenceDataFrame = pd.read_csv("ExercisesReferences/SquatSideWaysReference.csv")
        evaluation_angles = ["right_armpit_angle", "right_shoulder_angle", "trunk_angle", "right_knee_angle"]
        errors = []
        for evaluation_angle in evaluation_angles:
            referenceTimeSeries = referenceDataFrame[evaluation_angle].to_numpy()
            exersiceTimeSeries = exersiceDataFrame[evaluation_angle].to_numpy()
            error, backtracking = DTW(referenceTimeSeries, exersiceTimeSeries)
            errors.append(error)
        errors = np.array(errors)
        return np.sqrt(np.sum(errors**2) / len(errors)) / 411.653798721221