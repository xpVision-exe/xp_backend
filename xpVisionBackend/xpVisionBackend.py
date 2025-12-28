from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from DTWAnalysis import DTW
from ComputerVisionAnalysis import ExtractLandmarkAngles, ExtractCSVDataFromLandmarkAngles
from typing import List
from ExersicesEvaluator import EvaluateExersice


class DWTDTO(BaseModel):
    signal1: List[int]
    signal2: List[int]

app = FastAPI()



@app.post("/")
async def performDTW(inputs: DWTDTO):
    return DTW(inputs.signal1, inputs.signal2)  

@app.post("/uploadvideo")
async def create_upload_file(exersiceName: str, file: UploadFile):
    contents = await file.read()
    with open(file.filename, "wb") as binary_file:
        binary_file.write(contents)
    raw_data = ExtractLandmarkAngles(file.filename)
    exersice_df = ExtractCSVDataFromLandmarkAngles(raw_data)
    return EvaluateExersice(exersiceName, exersice_df)
