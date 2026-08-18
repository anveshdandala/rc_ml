# ML Resume Classification & ATS Evaluation Service

A RESTful Machine Learning inference and evaluation microservice built with **Python 3.10+** and **FastAPI**. This service processes raw resume text and job descriptions to perform automated role classification, skill extraction, ATS compliance scoring, and semantic match analysis.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [API Reference](#api-reference)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Environment & Configuration](#environment--configuration)
- [Testing & Evaluation](#testing--evaluation)

---

## Overview

The ML service is a core component of the Resume Classifier suite. It receives raw extracted text from candidate resumes along with job description requirements, processes them through NLP pipelines and trained Scikit-Learn models, and returns structured feedback including predicted roles, matching skills, missing skills, and an overall ATS compliance score.

---

## Key Features

- **Role Prediction**: Predicts candidate role and job description role using TF-IDF vectorization and trained Scikit-Learn classifiers.
- **Skill Extraction**: Extracts technical and soft skills from resume text using n-grams and rule-based NLP matchers (`extract_skills_rule_based`).
- **ATS Compliance Scoring**: Evaluates structural formatting, contact details completeness, use of strong action verbs, and quantified achievements.
- **Semantic Job Matching**: Computes match percentage between candidate profile and job description, identifying matched and missing skill gaps.
- **Experience Estimation**: Calculates years of professional experience extracted from resume timelines.

---

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: [FastAPI 0.113.0](https://fastapi.tiangolo.com/)
- **ASGI Server**: [Uvicorn 0.30.1](https://www.uvicorn.org/)
- **Data & ML**: `scikit-learn==1.8.0`, `pandas==2.3.3`, `joblib==1.4.2`, `pydantic==2.8.0`
- **Visualization & Evaluation**: `seaborn==0.13.2`, `matplotlib==3.7.2`

---

## Project Structure

```
ml/
├── app.py                     # FastAPI application entrypoint & REST API handlers
├── requirements.txt           # Python dependencies
├── pipeline/                  # NLP & Machine Learning processing pipeline
│   ├── predict.py             # Main inference orchestrator (predict_resume)
│   ├── preprocess.py          # Text cleaning, structural formatting, n-gram generation
│   ├── extractor.py           # Skill extraction & experience calculation
│   ├── compute_similarity.py  # Match score & ATS evaluation calculations
│   ├── classifier.py         # Role classification loader & inference
│   ├── feature_eng.py         # Feature transformation utilities
│   ├── eval.py                # Section-level quality scoring
│   └── ranker.py              # Candidate ranking algorithms
├── models/                    # Serialized model binaries (.pkl / .joblib)
├── training/                  # Model training scripts and dataset loaders
├── testing/                   # Test suites and mock payload tests
├── eval_outputs/              # Evaluation plots and diagnostic reports
├── artifacts/                 # Serialized feature extractors and encoders
└── data/                      # Training & validation datasets
```

---

## Installation & Setup

### Prerequisites

- Python `3.10` or higher installed.
- `pip` package manager.

### 1. Navigate to the `ml` directory

```bash
cd ml
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Development Server

```bash
fastapi dev app.py
```
*Or using Uvicorn directly:*
```bash
uvicorn app:app --reload --port 8000
```

The service will start at `http://127.0.0.1:8000`. You can inspect the interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

---

## 📑 API Reference

### 1. Service Health Check

- **URL**: `GET /`
- **Description**: Returns operational status of the service.
- **Response**:
```json
{
  "message": "ML service is running"
}
```

---

### 2. Resume & Job Description Prediction

- **URL**: `POST /predict`
- **Content-Type**: `application/json`

#### Request Payload:
```json
{
  "text": "Senior Software Engineer with 5 years experience in Python, FastAPI, React, Docker, and PostgreSQL. Led back-end development for scaled web apps.",
  "jobDescription": "We are seeking a Full Stack Python Developer with expertise in FastAPI, React, Docker, and SQL databases to build modern scalable applications."
}
```

#### Successful Response (`200 OK`):
```json
{
  "success": true,
  "data": {
    "your_role": "Backend Engineer",
    "desired_role": "Full Stack Engineer",
    "skills": ["python", "fastapi", "react", "docker", "postgresql"],
    "matched_skills": ["python", "fastapi", "react", "docker"],
    "missing_skills": ["sql"],
    "experience": 5,
    "match_score": 88.5,
    "ats_score": 92,
    "ats_feedback": [
      "Good inclusion of technical skills.",
      "Quantified achievements detected."
    ],
    "ats_breakdown": {
      "contact_score": 100,
      "structure_score": 90,
      "action_verb_score": 85,
      "quantified_score": 90
    }
  }
}
```

#### Error Response (`500 Internal Server Error`):
```json
{
  "error": "Detailed error message"
}
```

---

## Machine Learning Pipeline

The prediction workflow follows these sequential steps:

1. **Text Normalization & Structuring**: `clean_text_structure` cleans raw text, standardizing whitespace and splitting resume sections (Education, Experience, Skills, Projects).
2. **N-Gram Generation**: `generate_ngrams` generates unigram, bigram, and trigram tokens for phrase matching.
3. **Role Classification**: TF-IDF vectors are passed through pre-trained classifiers to infer `your_role` (candidate) and `desired_role` (from JD).
4. **Skill Extraction**: Matches extracted n-grams against technical taxonomy dictionaries to extract candidate and JD skill sets.
5. **Quality & ATS Breakdown**:
   $$\text{Resume Quality Score} = 0.30(\text{Quantified Metrics}) + 0.25(\text{Action Verbs}) + 0.25(\text{Structure}) + 0.20(\text{Contact Info})$$
6. **Match Score Computation**: Evaluates semantic similarity, skill coverage ratios, and experience compatibility.

---

## Testing & Evaluation

To evaluate model accuracy or run internal test scripts:

```bash
python -m testing.test_pipeline
```
Evaluation metrics and visual charts are saved automatically to `eval_outputs/`.
