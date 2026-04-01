from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.predict import predict_resume

app = FastAPI()

class ResumeInput(BaseModel):
    text: str
    jobDescription: str

@app.get("/")
def root():
    return {"message": "ML service is running"}

@app.post("/predict")
def predict(payload: ResumeInput):
    try:
        result = predict_resume(payload.text, payload.jobDescription)
        return {"success": True, "data": result}
    except Exception as e:
        traceback.print_exc()   # ← prints full traceback to uvicorn terminal
        return JSONResponse(status_code=500, content={"error": str(e)})
