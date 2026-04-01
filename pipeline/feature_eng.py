import joblib

pipeline = joblib.load("models/resume_pipeline.pkl")
vectorizer = pipeline.named_steps['tfidf']

def transform(text):
    return vectorizer.transform([text])
