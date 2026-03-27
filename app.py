from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.predict import predict_resume

app = FastAPI()

class ResumeInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "ML service is running"}

@app.post("/predict")
def predict(data: ResumeInput):
    result = predict_resume(data.text)
    print("from app.py", result)

    return {
        "success": True,
        "data": result
    }