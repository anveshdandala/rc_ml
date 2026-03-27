from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

df = pd.read_csv("data/Resume.csv")
texts = df["Resume_str"]
labels = df["Category"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)
print("Model trained successfully")

joblib.dump(model, "models/resume_classifier.pk1")
joblib.dump(vectorizer, "artifacts/vectorizer.pkl")