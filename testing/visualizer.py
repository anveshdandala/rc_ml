import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd

# ── load ──────────────────────────────────────────────────────────────────
pipeline  = joblib.load("models/resume_pipeline.pkl")
tfidf     = pipeline.named_steps["tfidf"]
clf       = pipeline.named_steps["clf"]

df        = pd.read_csv("./data/Resume.csv").dropna(subset=["Resume_str", "Category"])
texts     = df["Resume_str"]
labels    = df["Category"]

_, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
y_pred = pipeline.predict(X_test)

classes   = clf.classes_
fig_dir   = "eval_outputs"
import os; os.makedirs(fig_dir, exist_ok=True)

plt.style.use("dark_background")
ACCENT = "#3b82f6"

# ── 1. Confusion Matrix ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
cm = confusion_matrix(y_test, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
ax.set_title("Confusion Matrix — Resume Role Classifier", fontsize=15, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{fig_dir}/confusion_matrix.png", dpi=150)
plt.close()
print("✓ confusion_matrix.png")

# ── 2. Per-class F1, Precision, Recall bar chart ──────────────────────────
report = classification_report(y_test, y_pred, output_dict=True)
rows   = {k: v for k, v in report.items() if k in classes}
cats   = list(rows.keys())
f1s    = [rows[c]["f1-score"]  for c in cats]
precs  = [rows[c]["precision"] for c in cats]
recs   = [rows[c]["recall"]    for c in cats]

x = np.arange(len(cats))
w = 0.26
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - w, precs, w, label="Precision", color="#3b82f6", alpha=0.9)
ax.bar(x,     recs,  w, label="Recall",    color="#10b981", alpha=0.9)
ax.bar(x + w, f1s,   w, label="F1 Score",  color="#f59e0b", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
ax.legend(); ax.axhline(0.8, color="white", linewidth=0.5, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{fig_dir}/per_class_metrics.png", dpi=150)
plt.close()
print("✓ per_class_metrics.png")

# ── 3. Top predictive words per class ─────────────────────────────────────
feature_names = np.array(tfidf.get_feature_names_out())
n_show        = 12
n_classes     = len(classes)
cols          = 4
rows_grid     = (n_classes + cols - 1) // cols

fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 5, rows_grid * 4))
axes = axes.flatten()

for i, (cls, coef) in enumerate(zip(classes, clf.coef_)):
    top_idx  = np.argsort(coef)[-n_show:][::-1]
    top_words = feature_names[top_idx]
    top_vals  = coef[top_idx]
    
    colors = [ACCENT if v > 0 else "#fb7185" for v in top_vals]
    axes[i].barh(range(n_show), top_vals[::-1], color=colors[::-1], alpha=0.85)
    axes[i].set_yticks(range(n_show))
    axes[i].set_yticklabels(top_words[::-1], fontsize=8)
    axes[i].set_title(cls, fontsize=11, fontweight="bold", color=ACCENT)
    axes[i].axvline(0, color="white", linewidth=0.4)

for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Top Discriminative Words per Role (Logistic Regression Coefficients)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{fig_dir}/top_words_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ top_words_per_class.png")

# ── 4. Live sample predictions ────────────────────────────────────────────
samples = [
    ("Experienced Python developer, machine learning, TensorFlow, data pipelines", "DATA SCIENCE / ML"),
    ("Frontend developer, React, TypeScript, CSS, UI/UX, Figma",                 "FRONTEND"),
    ("Java Spring Boot microservices, REST API, Kafka, backend engineer",         "BACKEND"),
    ("SQL, Excel, Power BI, data analyst, financial reporting, dashboards",       "DATA / FINANCE"),
    ("Registered nurse, patient care, ICU, clinical documentation, EMR",          "HEALTHCARE"),
    ("SEO, Google Ads, content strategy, social media marketing, campaigns",      "MARKETING"),
]

print("\n" + "="*60)
print("  LIVE SAMPLE PREDICTIONS")
print("="*60)
for text, expected_area in samples:
    pred  = pipeline.predict([text])[0]
    probs = pipeline.predict_proba([text])[0]
    conf  = max(probs) * 100
    top3  = sorted(zip(classes, probs), key=lambda x: -x[1])[:3]
    
    print(f"\nINPUT : {text[:70]}...")
    print(f"PREDICTED : {pred}  (confidence: {conf:.1f}%)")
    print(f"TOP 3 : {', '.join(f'{c}={p*100:.0f}%' for c,p in top3)}")

# ── 5. Confidence distribution plot ───────────────────────────────────────
probs_all = pipeline.predict_proba(X_test)
max_conf  = probs_all.max(axis=1) * 100
correct   = (y_pred == y_test.values)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(max_conf[correct],  bins=30, alpha=0.75, color="#10b981", label="Correct predictions")
ax.hist(max_conf[~correct], bins=30, alpha=0.75, color="#fb7185", label="Wrong predictions")
ax.set_xlabel("Prediction Confidence (%)")
ax.set_ylabel("Count")
ax.set_title("Confidence Distribution: Correct vs Wrong Predictions", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/confidence_distribution.png", dpi=150)
plt.close()
print("\n✓ confidence_distribution.png")
print(f"\nAll charts saved to ./{fig_dir}/")