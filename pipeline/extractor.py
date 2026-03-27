SKILLS_DB = ["python", "java", "sql", "machine learning", "react"]

def extract_skills(text):
    found = []
    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)
    return found