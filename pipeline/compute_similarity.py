from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

sem_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_match_score(role: str, skills: list, experience: int, jobDescription: str):
    resume_summary = f"{role} with {experience} years experience, skills: {', '.join(skills)}"
    
    resume_vec = sem_model.encode([resume_summary])
    jd_vec = sem_model.encode([jobDescription])
    
    base_score = float(cosine_similarity(resume_vec, jd_vec)[0][0]) * 100

    jd_lower = jobDescription.lower()
    jd_words = set(jd_lower.split())

    matched_skills = [
        s for s in skills
        if s.lower() in jd_words          # exact word match: "python" in "python developer"
        or s.lower() in jd_lower          # substring match: "react" in "reactjs developer"  
        or any(s.lower() in w for w in jd_words)  # partial: "sql" in "mysql"
    ]

    skill_bonus = (len(matched_skills) / max(len(skills), 1)) * 20

    final_score = round(min(base_score + skill_bonus, 100), 2)
    return final_score, matched_skills

def compute_ats_score(cleaned_txt: str, job_description: str = "", skills: list = []):
    score = 0
    feedback = []
    breakdown = {}
    
    text_lower = cleaned_txt.lower()
    words = text_lower.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]', cleaned_txt)

    # 1. LENGTH & DENSITY (10 points)
    if 400 <= word_count <= 800:
        score += 10
        breakdown['length'] = {'score': 10, 'max': 10}
    elif 300 <= word_count < 400 or 800 < word_count <= 1000:
        score += 6
        breakdown['length'] = {'score': 6, 'max': 10}
        feedback.append("Aim for 400–800 words for optimal ATS readability.")
    elif word_count < 300:
        score += 2
        breakdown['length'] = {'score': 2, 'max': 10}
        feedback.append(f"Resume is too short ({word_count} words). Expand your experience and project descriptions.")
    else:
        score += 3
        breakdown['length'] = {'score': 3, 'max': 10}
        feedback.append(f"Resume is too long ({word_count} words). Condense to under 1000 words.")

    # 2. QUANTIFIED ACHIEVEMENTS (15 points)

    metrics = re.findall(r'\b\d+\+?\s*%|\$[\d,]+|\b\d+x\b|\b\d{4}\b(?![\d-])|\b\d+\s*(users|clients|teams?|projects?|engineers?|members?|systems?)\b', text_lower)
    year_pattern = re.findall(r'\b(19|20)\d{2}\b', cleaned_txt)
    pure_metrics = [m for m in metrics if m not in year_pattern]

    if len(pure_metrics) >= 8:
        score += 15
        breakdown['metrics'] = {'score': 15, 'max': 15}
    elif len(pure_metrics) >= 4:
        score += 10
        breakdown['metrics'] = {'score': 10, 'max': 15}
        feedback.append("Add more quantified achievements (e.g., 'Reduced load time by 40%', 'Managed team of 8').")
    elif len(pure_metrics) >= 1:
        score += 5
        breakdown['metrics'] = {'score': 5, 'max': 15}
        feedback.append("Very few metrics found. Numbers make your impact concrete and credible.")
    else:
        score += 0
        breakdown['metrics'] = {'score': 0, 'max': 15}
        feedback.append("No quantified achievements detected. Add numbers, percentages, or scale indicators.")

    # 3. ACTION VERBS — QUALITY + VARIETY (10 points)
    action_verbs = {
        'developed', 'engineered', 'architected', 'built', 'designed', 'created', 'implemented',
        'launched', 'deployed', 'automated', 'migrated', 'integrated', 'refactored', 'optimized',
        'led', 'managed', 'mentored', 'supervised', 'coordinated', 'spearheaded', 'orchestrated',
        'increased', 'reduced', 'improved', 'accelerated', 'achieved', 'delivered', 'generated',
        'streamlined', 'resolved', 'diagnosed', 'overhauled', 'revamped', 'transformed',
        'researched', 'analyzed', 'evaluated', 'identified', 'established', 'defined'
    }
    passive_phrases = ['responsible for', 'worked on', 'helped with', 'assisted in', 'involved in', 'tasked with']

    words_set = set(words)
    verbs_found = action_verbs.intersection(words_set)
    passive_found = [p for p in passive_phrases if p in text_lower]

    verb_score = 0
    if len(verbs_found) >= 8:
        verb_score = 10
    elif len(verbs_found) >= 5:
        verb_score = 7
    elif len(verbs_found) >= 3:
        verb_score = 4
    else:
        verb_score = 1
        feedback.append("Use strong action verbs to start bullet points — 'Engineered', 'Led', 'Reduced'.")

    if passive_found:
        verb_score = max(0, verb_score - (len(passive_found) * 2))
        feedback.append(f"Passive phrases detected: {passive_found}. Replace with direct action verbs.")

    score += verb_score
    breakdown['action_verbs'] = {'score': verb_score, 'max': 10, 'found': list(verbs_found)}


    # 4. CONTACT INFO (10 points)

    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cleaned_txt))
    has_phone = bool(re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', cleaned_txt))
    has_linkedin = bool(re.search(r'linkedin\.com/in/[\w-]+', text_lower))
    has_github = bool(re.search(r'github\.com/[\w-]+', text_lower))

    contact_score = 0
    if has_email: contact_score += 4
    else: feedback.append("Email address missing or unreadable.")
    if has_phone: contact_score += 3
    else: feedback.append("Phone number missing or not in a standard format.")
    if has_linkedin: contact_score += 2
    if has_github: contact_score += 1

    score += contact_score
    breakdown['contact_info'] = {
        'score': contact_score, 'max': 10,
        'email': has_email, 'phone': has_phone,
        'linkedin': has_linkedin, 'github': has_github
    }

    # 5. SECTION STRUCTURE (15 points)

    sections = {
        'education': r'\b(education|degree|bachelor|master|b\.?s\.?|m\.?s\.?|ph\.?d)\b',
        'experience': r'\b(experience|employment|work history|career)\b',
        'skills': r'\b(skills|technologies|tech stack|competencies|proficiencies)\b',
        'projects': r'\b(projects?|portfolio|case studies)\b',
        'certifications': r'\b(certifications?|certified|licenses?|credentials?)\b',
        'summary': r'\b(summary|objective|profile|about me|overview)\b',
    }

    found_sections = {}
    for section, pattern in sections.items():
        found_sections[section] = bool(re.search(pattern, text_lower))

    core_sections = ['education', 'experience', 'skills']
    bonus_sections = ['projects', 'certifications', 'summary']

    core_found = sum(found_sections[s] for s in core_sections)
    bonus_found = sum(found_sections[s] for s in bonus_sections)

    section_score = (core_found * 4) + (bonus_found * 1)
    section_score = min(section_score, 15)
    score += section_score
    breakdown['sections'] = {
        'score': section_score, 'max': 15,
        'found': [s for s, v in found_sections.items() if v],
        'missing': [s for s, v in found_sections.items() if not v]
    }

    if core_found < 3:
        missing = [s for s in core_sections if not found_sections[s]]
        feedback.append(f"Missing critical sections: {missing}. ATS systems expect these clearly labeled.")

    # 6. JOB DESCRIPTION KEYWORD MATCH (20 points)

    jd_match_score = 0
    matched_keywords = []
    missing_keywords = []

    if job_description:
        stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'be', 'will', 'we', 'you', 'your',
                     'our', 'that', 'this', 'have', 'has', 'not', 'but', 'from'}
        
        jd_words = set(re.findall(r'\b[a-z]{3,}\b', job_description.lower())) - stopwords
        resume_words = set(re.findall(r'\b[a-z]{3,}\b', text_lower))

        matched_keywords = list(jd_words.intersection(resume_words))
        missing_keywords = list(jd_words - resume_words)

        match_ratio = len(matched_keywords) / max(len(jd_words), 1)

        if match_ratio >= 0.7:
            jd_match_score = 20
        elif match_ratio >= 0.5:
            jd_match_score = 15
        elif match_ratio >= 0.3:
            jd_match_score = 10
            feedback.append(f"Keyword alignment with JD is moderate. Consider adding: {missing_keywords[:5]}")
        elif match_ratio >= 0.1:
            jd_match_score = 5
            feedback.append(f"Low JD keyword match. Missing important terms: {missing_keywords[:8]}")
        else:
            jd_match_score = 0
            feedback.append("Almost no overlap with job description keywords. Tailor your resume to the JD.")
    else:
        jd_match_score = 10  # neutral if no JD provided
        feedback.append("No job description provided. JD matching skipped — score is generalized.")

    score += jd_match_score
    breakdown['jd_keyword_match'] = {
        'score': jd_match_score, 'max': 20,
        'matched': matched_keywords,
        'missing': missing_keywords[:10]
    }

    # 7. SKILLS DEPTH & VARIETY (10 points)
    skill_categories = {
        'languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift', 'ruby', 'php'],
        'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring', 'express', 'nextjs', 'node'],
        'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'firebase', 'dynamodb'],
        'cloud_devops': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ci/cd', 'jenkins', 'github actions'],
        'ml_data': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'nlp']
    }

    categories_covered = 0
    skill_detail = {}
    for category, keywords in skill_categories.items():
        found = [k for k in keywords if k in text_lower]
        skill_detail[category] = found
        if found:
            categories_covered += 1

    if categories_covered >= 4:
        skill_score = 10
    elif categories_covered >= 3:
        skill_score = 7
    elif categories_covered >= 2:
        skill_score = 4
    else:
        skill_score = 1
        feedback.append("Skills section lacks variety. Include languages, frameworks, databases, and tools.")

    score += skill_score
    breakdown['skills_depth'] = {'score': skill_score, 'max': 10, 'categories': skill_detail}

    # 8. CONSISTENCY & FORMATTING SIGNALS (10 points)

    format_score = 0

    # Check for date consistency (e.g., Jan 2022, 2021-2023)
    date_patterns = re.findall(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|\b\d{4}\s*[-–]\s*(\d{4}|present|current)\b', text_lower)
    if len(date_patterns) >= 2:
        format_score += 4
    else:
        feedback.append("Add consistent date ranges for each role (e.g., 'Jan 2022 – Mar 2024').")

    # Check for bullet-point style (lines starting with common bullet chars or dashes)
    bullet_lines = re.findall(r'(?m)^[\s]*[-•*▪➢✓]\s+\w+', cleaned_txt)
    if len(bullet_lines) >= 5:
        format_score += 3

    # Check for repeated words (sign of lazy writing)
    word_freq = Counter(words)
    most_common = word_freq.most_common(5)
    filler_words = ['various', 'multiple', 'several', 'different', 'many']
    filler_found = [w for w in filler_words if word_freq.get(w, 0) > 2]
    if filler_found:
        feedback.append(f"Overused vague words: {filler_found}. Be specific instead.")
    else:
        format_score += 3

    score += format_score
    breakdown['formatting'] = {'score': format_score, 'max': 10}

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------
    final_score = min(score, 100)

    return {
        "ats_score": final_score,
        "feedback": feedback,
        "breakdown": breakdown
    }