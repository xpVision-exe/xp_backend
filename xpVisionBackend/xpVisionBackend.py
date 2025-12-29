from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from DTWAnalysis import DTW
from ComputerVisionAnalysis import ExtractLandmarkAngles, ExtractCSVDataFromLandmarkAngles
from typing import List
from ExercisesEvaluator import EvaluateExercise


class DTWDTO(BaseModel):
    signal1: List[int]
    signal2: List[int]


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.post("/")
async def performDTW(inputs: DTWDTO):
    return DTW(inputs.signal1, inputs.signal2)  

@app.post("uploadgyroreadings")

@app.post("/uploadvideo")
async def create_upload_file(exerciseName: str, file: UploadFile):
    try:
        contents = await file.read()
        with open("pose.mp4", "wb") as binary_file:
            binary_file.write(contents)
        raw_data = ExtractLandmarkAngles("pose.mp4")
        exercise_df = ExtractCSVDataFromLandmarkAngles(raw_data)
        exercise_df.to_csv("static/exerciseDataFrame.csv")
        error, evaluation_angles, optimal_indicies, error_signals = EvaluateExercise(exerciseName, exercise_df)
        responseDTO = {
                "message": "Success",
                "exerciseAccuracy": (1- error) * 100,
                "parametersOfPlotting": evaluation_angles,
                "optimal_indicies": optimal_indicies,
                "error_signals": error_signals
            }
        return responseDTO

    except Exception:
        return JSONResponse(
            content = {
                "message:": Exception
                },
            status_code=500
            )