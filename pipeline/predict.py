import logging
from .preprocess import clean_text, generate_ngrams
from .feature_eng import transform
from .classifier import predict_role
from .extractor import extract_skills_from_jd, estimate_experience,extract_skills_rule_based #extract_skills_from_resume
from sklearn.metrics.pairwise import cosine_similarity
from .compute_similarity import compute_match_score, compute_ats_score
from .eval import score_resume_section
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_resume(text: str, jobDescription:str):
    cleaned_txt = clean_text(text)
    cleaned_jd = clean_text(jobDescription)
    
    words = cleaned_txt.split()
    phrases = generate_ngrams(words)

    roleByText = predict_role(cleaned_txt)
    roleByJD = predict_role(cleaned_jd)
    jd_skils = extract_skills_from_jd(cleaned_jd)
    user_skills = extract_skills_rule_based(phrases)
    experience = estimate_experience(cleaned_txt)

    # quantified_score  = score_resume_section(cleaned_txt, "quantified achievements and metrics")
    # action_verb_score = score_resume_section(cleaned_txt, "strong action verbs and active voice")
    # structure_score   = score_resume_section(cleaned_txt, "clear professional structure and sections")
    # contact_score     = score_resume_section(cleaned_txt, "complete contact information")

    # print(f"quantified_score: {quantified_score}")
    # print(f"action_verb_score: {action_verb_score}")
    # print(f"structure_score: {structure_score}")
    # print(f"contact_score: {contact_score}")
    matched_score, matched_skills, missing_skills = compute_match_score(cleaned_txt,roleByText, user_skills, jd_skils, experience, cleaned_jd)
    ats_evaluation = compute_ats_score(text, jobDescription)

    final_result = {
        "your_role": roleByText,
        "desired_role": roleByJD,
        "skills": user_skills,
        "matched_skills": matched_skills,
        "missing_skills":missing_skills,
        "experience": experience,
        "match_score": matched_score,
        "ats_score": ats_evaluation["ats_score"],
        "ats_feedback": ats_evaluation["feedback"],
        "ats_breakdown": ats_evaluation["breakdown"],
        # "quantified_score": quantified_score,
        # "action_verb_score": action_verb_score,
        # "structure_score": structure_score,
        # "contact_score": contact_score,
    }
  
    # print(final_result)
    return final_result