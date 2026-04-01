import joblib

try:
    model = joblib.load("models/resume_pipeline.pkl")
    
except Exception as e:
    print(e)

def predict_role(text:str):
    return model.predict([text])[0]