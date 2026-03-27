import joblib

try:
    model = joblib.load("models/resume_classifier.pk1")
except Exception as e:
    print(e)

def predict_role(features):
    return model.predict(features)[0]