from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from DTWAnalysis import DTW
from typing import List

class DWTDTO(BaseModel):
    signal1: List[int]
    signal2: List[int]

app = FastAPI()



@app.post("/")
async def performDTW(inputs: DWTDTO):
    return DTW(inputs.signal1, inputs.signal2)  

@app.post("/uploadvideo/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    with open(file.filename, "wb") as binary_file:
        binary_file.write(contents)
    return {"filename": file.filename}