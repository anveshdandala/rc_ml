from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer, util
import re
import logging

logger = logging.getLogger(__name__)

# load once at module level
_zero_shot = None
_ner       = None
_sem_model = None

def _get_zero_shot():
    global _zero_shot
    if _zero_shot is None:
        logger.info("Loading zero-shot classifier...")
        _zero_shot = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0
        )
    return _zero_shot

def _get_ner():
    global _ner
    if _ner is None:
        logger.info("Loading NER model...")
        _ner = hf_pipeline(
            "token-classification",
            model="algiraldohe/lm-ner-linkedin-skills-recognition",
            aggregation_strategy="simple",
            device=0
        )
    return _ner

def _get_sem():
    global _sem_model
    if _sem_model is None:
        _sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sem_model


def compute_match_score(resume_text:str, role:str, user_skills:list, jd_skills:list, experience:str, jobDescription:str):
    
    # ── skill matching ────────────────────────────────────────────
    matched_skills = [s for s in user_skills if s in jd_skills]
    missing_skills = [s for s in jd_skills   if s not in user_skills]

    # ── semantic similarity ───────────────────────────────────────
    resume_summary = f"{role} with {experience} years experience, skills: {', '.join(user_skills)}"
    
    resume_vec = _get_sem().encode([resume_summary])
    jd_vec     = _get_sem().encode([jobDescription])
    
    base_score = float(cosine_similarity(resume_vec, jd_vec)[0][0]) * 100

    skill_bonus = (len(matched_skills) / max(len(jd_skills), 1)) * 20

    matched_score = round(min(base_score + skill_bonus, 100), 2)

    return matched_score, matched_skills, missing_skills

def compute_ats_score(resume_text: str, job_description: str = "") -> dict:
    feedback  = []
    breakdown = {}
    score     = 0

    zs  = _get_zero_shot()
    sem = _get_sem()

    resume_chunk = resume_text[:1024]

    # 10 pts
    has_email    = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text))
    has_phone    = bool(re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", resume_text))
    has_linkedin = bool(re.search(r"linkedin\.com/in/[\w-]+", resume_text.lower()))
    has_github   = bool(re.search(r"github\.com/[\w-]+", resume_text.lower()))

    contact_score = (4 if has_email else 0) + (3 if has_phone else 0) + \
                    (2 if has_linkedin else 0) + (1 if has_github else 0)
    if not has_email: feedback.append("Email address missing or unreadable.")
    if not has_phone: feedback.append("Phone number missing or not in standard format.")
    if not has_linkedin: feedback.append("Add your LinkedIn profile URL.")
    score += contact_score
    breakdown["contact_info"] = {
        "score": contact_score, "max": 10,
        "email": has_email, "phone": has_phone,
        "linkedin": has_linkedin, "github": has_github
    }

    # ── 2. Quantified achievements (zero-shot) ── 20 pts
    metrics_result = zs(
        resume_chunk,
        candidate_labels=[
            "contains specific numbers percentages and measurable achievements",
            "vague descriptions without numbers or metrics"
        ]
    )
    metrics_conf = metrics_result["scores"][0] \
        if metrics_result["labels"][0] == "contains specific numbers percentages and measurable achievements" \
        else metrics_result["scores"][1]

    if metrics_conf >= 0.75:
        m_score = 20
    elif metrics_conf >= 0.55:
        m_score = 12
        feedback.append("Add more quantified achievements — numbers, percentages, scale.")
    else:
        m_score = 4
        feedback.append("No measurable impact found. Add metrics like 'increased performance by 30%'.")
    score += m_score
    breakdown["metrics"] = {"score": m_score, "max": 20, "confidence": round(metrics_conf, 2)}

    # ── 3. Action verbs & active voice (zero-shot) ── 15 pts
    verbs_result = zs(
        resume_chunk,
        candidate_labels=[
            "uses strong action verbs and active voice",
            "uses passive language and weak phrasing"
        ]
    )
    verbs_conf = verbs_result["scores"][0] \
        if verbs_result["labels"][0] == "uses strong action verbs and active voice" \
        else verbs_result["scores"][1]

    if verbs_conf >= 0.75:
        v_score = 15
    elif verbs_conf >= 0.55:
        v_score = 9
        feedback.append("Strengthen your language — use action verbs like 'Led', 'Built', 'Optimized'.")
    else:
        v_score = 3
        feedback.append("Too much passive language. Replace 'responsible for' with direct action verbs.")
    score += v_score
    breakdown["action_verbs"] = {"score": v_score, "max": 15, "confidence": round(verbs_conf, 2)}

    # ── 4. Structure & sections (zero-shot) ── 15 pts
    structure_result = zs(
        resume_chunk,
        candidate_labels=[
            "well structured with clear sections for experience education and skills",
            "poorly structured and missing standard resume sections"
        ]
    )
    struct_conf = structure_result["scores"][0] \
        if "well structured" in structure_result["labels"][0] \
        else structure_result["scores"][1]

    if struct_conf >= 0.75:
        s_score = 15
    elif struct_conf >= 0.55:
        s_score = 9
        feedback.append("Improve resume structure — ensure Experience, Education, Skills sections are clearly labeled.")
    else:
        s_score = 3
        feedback.append("Resume structure is unclear. Add standard sections: Summary, Experience, Skills, Education.")
    score += s_score
    breakdown["structure"] = {"score": s_score, "max": 15, "confidence": round(struct_conf, 2)}

    # ── 5. JD semantic match (sentence transformer) ── 25 pts
    jd_score = 0
    if job_description:
        resume_vec = sem.encode([resume_text[:512]])
        jd_vec     = sem.encode([job_description[:512]])
        from sklearn.metrics.pairwise import cosine_similarity
        sim = float(cosine_similarity(resume_vec, jd_vec)[0][0])

        if sim >= 0.55:
            jd_score = 25
        elif sim >= 0.40:
            jd_score = 18
            feedback.append("Resume is moderately aligned with the JD. Add more relevant terminology.")
        elif sim >= 0.25:
            jd_score = 10
            feedback.append("Low alignment with job description. Tailor your resume more specifically.")
        else:
            jd_score = 3
            feedback.append("Resume and job description are very misaligned. Significant tailoring needed.")

        # skill gap via NER
        try:
            ner       = _get_ner()
            jd_entities  = ner(job_description[:512])
            jd_skills    = {e["word"].lower().strip() for e in jd_entities
                           if e["entity_group"] == "SKILL" and e["score"] > 0.80}
            res_entities = ner(resume_text[:512])
            res_skills   = {e["word"].lower().strip() for e in res_entities
                           if e["entity_group"] == "SKILL" and e["score"] > 0.80}

            matched = list(jd_skills & res_skills)
            missing = list(jd_skills - res_skills)

            if missing:
                feedback.append(f"Skills in JD not found in resume: {', '.join(missing[:6])}")

            breakdown["skill_gap"] = {
                "jd_skills":      list(jd_skills),
                "matched_skills": matched,
                "missing_skills": missing
            }
        except Exception as e:
            logger.warning(f"NER skill extraction failed: {e}")

    else:
        jd_score = 12  # neutral
        feedback.append("No job description provided — JD match skipped.")

    score += jd_score
    breakdown["jd_match"] = {"score": jd_score, "max": 25}

    # ── 6. Overall professional quality (zero-shot) ── 15 pts
    quality_result = zs(
        resume_chunk,
        candidate_labels=[
            "professionally written high quality resume",
            "unprofessional or low quality resume"
        ]
    )
    quality_conf = quality_result["scores"][0] \
        if "professionally written" in quality_result["labels"][0] \
        else quality_result["scores"][1]

    if quality_conf >= 0.75:
        q_score = 15
    elif quality_conf >= 0.55:
        q_score = 9
        feedback.append("Resume writing quality could be improved — be more specific and professional.")
    else:
        q_score = 3
        feedback.append("Resume needs significant writing improvement.")
    score += q_score
    breakdown["quality"] = {"score": q_score, "max": 15, "confidence": round(quality_conf, 2)}

    return {
        "ats_score": min(score, 100),
        "feedback":  feedback,
        "breakdown": breakdown
    }