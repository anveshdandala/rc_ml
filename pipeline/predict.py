from .preprocess import clean_text
from .feature_eng import transform
from .classifier import predict_role
from .extractor import extract_skills

def predict_resume(text: str):
    # Step 1: Clean
    cleaned = clean_text(text)

    # Step 2: Convert to features
    features = transform(cleaned)

    # Step 3: Predict role
    role = predict_role(features)

    # Step 4: Extract skills
    skills = extract_skills(cleaned)

    # Step 5: (Optional) experience placeholder
    experience = estimate_experience(cleaned)

    

    return {
        "role": role,
        "skills": skills,
        "experience": experience
    }


def estimate_experience(text: str):
    import re
    matches = re.findall(r"(\d+)\s+years?", text)
    if matches:
        return max(map(int, matches))
    return 0