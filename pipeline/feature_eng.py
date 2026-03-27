import joblib

vectorizer = joblib.load("artifacts/vectorizer.pkl")

def transform(text):
    return vectorizer.transform([text])