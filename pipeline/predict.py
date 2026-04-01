import logging
from .preprocess import clean_text
from .feature_eng import transform
from .classifier import predict_role
from .extractor import extract_skills
from sklearn.metrics.pairwise import cosine_similarity
from .compute_similarity import compute_match_score, compute_ats_score
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def estimate_experience(text: str):
    import re
    matches = re.findall(r"(\d+)\s+years?", text)
    if matches:
        return max(map(int, matches))
    return 0

def predict_resume(text: str, jobDescription:str):
    cleaned_txt = clean_text(text)
    cleaned_jd = clean_text(jobDescription)

    experience = estimate_experience(cleaned_txt)
    role = predict_role(cleaned_txt)

    skills = extract_skills(cleaned_txt)
        
    match_score,matched_skills = compute_match_score(role, skills, experience, jobDescription)

    ats_evaluation = compute_ats_score(text, jobDescription, skills)
    ats_evaluation = compute_ats_score(text, jobDescription, skills)
    logger.debug(f"ATS breakdown: {ats_evaluation['breakdown']}")

  
    return {
        "role": role,
        "skills": skills,
        "matched_skills": matched_skills,
        "experience": experience,
        "match_score": match_score,
        "ats_score": ats_evaluation["ats_score"],
        "ats_feedback": ats_evaluation["feedback"],
         "ats_breakdown": ats_evaluation["breakdown"],
    }
