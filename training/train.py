import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("./data/gpt_dataset.csv")
df = df.dropna(subset=['Category','Resume']) 

labels = df["Category"]
texts = df["Resume"]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',      # Remove words like 'the', 'is', 'and'
        max_df=0.85,               # Ignore words that appear in >85% of resumes (they don't help differentiate)
        min_df=3,                  # Ignore rare words that appear in <3 resumes (prevents overfitting to typos)
        ngram_range=(1, 2)         # Capture 1-word and 2-word phrases (e.g., "machine learning", "data science")
    )),
    ('clf', LogisticRegression(
        class_weight='balanced',   # Penalize mistakes on rare job categories heavily
        max_iter=1000              # Give the math more time to converge
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "models/resume_pipeline.pkl")
print("\nModel pipeline saved successfully to models/resume_pipeline.pkl")