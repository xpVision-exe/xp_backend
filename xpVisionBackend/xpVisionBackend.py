from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import os
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

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}


@app.post("/")
async def performDTW(inputs: DTWDTO):
    return DTW(inputs.signal1, inputs.signal2)  

@app.post("uploadgyroreadings")

@app.post("/uploadvideo")
async def create_upload_file(exerciseName: str, file: UploadFile):
    try:
        contents = await file.read()
        file_extension = os.path.splitext(file.filename)[-1]
        if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
            return JSONResponse(
            content = {
                "message:": f"Invalid file type. Allowed video formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
                },
            status_code=400
            )
        
        with open("pose" + file_extension, "wb") as binary_file:
            binary_file.write(contents)
        raw_data = ExtractLandmarkAngles("pose" + file_extension)

        if(raw_data == []):
            return JSONResponse(
            content = {
                "message:": f"Invalid Video: The provided video does not contain footage of exersices nor workouts. Please upload another video"
                },
            status_code=400
            )

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