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
        stop_words='english',      
        max_df=0.85,               
        min_df=3,                  
        ngram_range=(1, 2)         
    )),
    ('clf', LogisticRegression(
        class_weight='balanced',   
        max_iter=1000              
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "models/resume_pipeline.pkl")
print("\nModel pipeline saved successfully to models/resume_pipeline.pkl")