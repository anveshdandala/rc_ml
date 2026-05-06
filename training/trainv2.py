import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
import re

# ── paths ─────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(BASE)
DATA_PATH = os.path.join(ROOT, "data", "gpt_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")
EVAL_DIR  = os.path.join(ROOT, "eval_outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── preprocessing ─────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+",        " ", text)   # urls
    text = re.sub(r"[\w\.-]+@[\w\.-]+",     " ", text)   # emails
    text = re.sub(r"\d{10,}",               " ", text)   # phone numbers
    text = re.sub(r"[^a-z\s\+\#]",          " ", text)   # keep c++, c#
    text = re.sub(r"\s+",                   " ", text)   # collapse spaces
    return text.strip()

# ── load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH).dropna(subset=["Category", "Resume"])
df["clean"] = df["Resume"].apply(clean)

print(f"Dataset: {len(df)} rows, {df['Category'].nunique()} classes")
print(df["Category"].value_counts())

X = df["clean"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── build two pipelines and compare ──────────────────────────────────────
tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.85,
    min_df=2,
    ngram_range=(1, 3),       # unigrams + bigrams + trigrams
    sublinear_tf=True,        # log normalization — helps with long docs
    max_features=30000,
)

# LinearSVC is usually stronger than LogisticRegression on text
# We wrap it with CalibratedClassifierCV to get probability scores
svc_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("clf",   CalibratedClassifierCV(LinearSVC(
        class_weight="balanced",
        max_iter=2000,
        C=0.8
    )))
])

lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", max_df=0.85, min_df=2,
        ngram_range=(1, 3), sublinear_tf=True, max_features=30000
    )),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
    ))
])

# ── cross-validation comparison ───────────────────────────────────────────
print("\n── Cross-validation (5-fold) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in [("LinearSVC", svc_pipeline), ("LogisticRegression", lr_pipeline)]:
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"{name:22s}  F1={scores.mean():.4f} ± {scores.std():.4f}")

# ── train best model (SVC usually wins on small datasets) ────────────────
print("\nTraining final model...")
svc_pipeline.fit(X_train, y_train)
y_pred = svc_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── save model + vectorizer separately ───────────────────────────────────
joblib.dump(svc_pipeline, os.path.join(MODEL_DIR, "resume_pipeline_v2.pkl"))

# also save vectorizer standalone for feature_eng.py
tfidf_fitted = svc_pipeline.named_steps["tfidf"]
joblib.dump(tfidf_fitted, os.path.join(ROOT, "artifacts", "vectorizer_v2.pkl"))

print("\n✓ Model saved to models/resume_pipeline.pkl")
print("✓ Vectorizer saved to artifacts/vectorizer.pkl")

# ── plots ─────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
classes = svc_pipeline.classes_

# 1. Confusion matrix
fig, ax = plt.subplots(figsize=(12, 9))
cm = confusion_matrix(y_test, y_pred, labels=classes)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=classes, yticklabels=classes,
    linewidths=0.5, ax=ax
)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual",    fontsize=11)
ax.set_title("Confusion Matrix — Resume Role Classifier", fontsize=14, fontweight="bold")
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0,  fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("✓ confusion_matrix.png")

# 2. Per-class F1
report = classification_report(y_test, y_pred, output_dict=True)
cats   = [c for c in classes if c in report]
f1s    = [report[c]["f1-score"]  for c in cats]
precs  = [report[c]["precision"] for c in cats]
recs   = [report[c]["recall"]    for c in cats]

x = np.arange(len(cats))
w = 0.26
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w, precs, w, label="Precision", color="#3b82f6", alpha=0.9)
ax.bar(x,     recs,  w, label="Recall",    color="#10b981", alpha=0.9)
ax.bar(x + w, f1s,   w, label="F1",        color="#f59e0b", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Per-Class Metrics", fontsize=13, fontweight="bold")
ax.axhline(0.8, color="white", linewidth=0.5, linestyle="--", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "per_class_metrics.png"), dpi=150)
plt.close()
print("✓ per_class_metrics.png")

# 3. Confidence distribution
probs   = svc_pipeline.predict_proba(X_test)
max_conf = probs.max(axis=1) * 100
correct  = y_pred == y_test.values

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(max_conf[correct],  bins=25, alpha=0.75, color="#10b981", label="Correct")
ax.hist(max_conf[~correct], bins=25, alpha=0.75, color="#fb7185", label="Wrong")
ax.set_xlabel("Confidence (%)")
ax.set_ylabel("Count")
ax.set_title("Prediction Confidence: Correct vs Wrong", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confidence_dist.png"), dpi=150)
plt.close()
print("✓ confidence_dist.png")

# 4. Live sample test
print("\n" + "="*60)
print("  LIVE SAMPLE PREDICTIONS")
print("="*60)
samples = [
    "React TypeScript frontend developer, CSS, Figma, UI components, 4 years experience",
    "Python Django REST API backend engineer, PostgreSQL, Redis, microservices",
    "AWS cloud architect, Terraform, Kubernetes, Docker, DevOps CI/CD pipelines",
    "TensorFlow PyTorch deep learning, NLP, computer vision, ML research engineer",
    "Flutter React Native iOS Android mobile developer, Swift Kotlin cross-platform",
    "Next.js Node.js React fullstack developer, MongoDB, GraphQL, 6 years",
    "Python pandas numpy scikit-learn data scientist, statistical modeling, A/B testing",
    "FastAPI Django Flask Python developer, REST APIs, celery, redis",
]

for text in samples:
    pred   = svc_pipeline.predict([clean(text)])[0]
    probs_ = svc_pipeline.predict_proba([clean(text)])[0]
    conf   = max(probs_) * 100
    top3   = sorted(zip(classes, probs_), key=lambda x: -x[1])[:3]
    print(f"\n► {text[:65]}")
    print(f"  PREDICTED : {pred}  ({conf:.1f}%)")
    print(f"  TOP 3     : {' | '.join(f'{c} {p*100:.0f}%' for c,p in top3)}")

print(f"\n\nAll eval outputs saved to: {EVAL_DIR}")