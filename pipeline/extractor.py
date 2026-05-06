import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import pipeline
from .classifier import predict_role
from .preprocess import clean_text
import re
ner = pipeline(
    "token-classification",
    model="algiraldohe/lm-ner-linkedin-skills-recognition",
    aggregation_strategy="simple"
)

# def extract_skills_from_resume(text: str) -> list[str]:
#     words  = text.split()
#     print(text)
#     chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]
    
#     skill_groups = {"TECHNICAL", "TECHNOLOGY"}
#     all_skills = set()
    
#     for chunk in chunks:
#         entities = ner(chunk[:512])
#         for e in entities:
#             if e["entity_group"] in skill_groups and e["score"] > 0.80:
#                 word = e["word"].lower().strip()
#                 if len(word) > 1:
#                     all_skills.add(word)

#     if not all_skills:
#         KNOWN_SKILLS = {
#             "react", "angular", "vue", "javascript", "typescript", "html", "css",
#             "node", "express", "python", "django", "flask", "fastapi", "java",
#             "spring", "sql", "postgresql", "mysql", "mongodb", "redis", "docker",
#             "kubernetes", "aws", "azure", "gcp", "terraform", "git", "graphql",
#             "machine learning", "deep learning", "tensorflow", "pytorch", "pandas",
#             "numpy", "scikit-learn", "go", "rust", "kotlin", "swift", "flutter",
#             "react native", "next.js", "tailwind", "figma", "linux", "bash"
#         }
#         text_lower = text.lower()
#         all_skills = {s for s in KNOWN_SKILLS if s in text_lower}
    
#     return list(all_skills)

def extract_skills_from_jd(job_description: str) -> list[str]:
    all_skills = set()
    
    ner_skills = _ner_extract(job_description)
    all_skills.update(ner_skills)
      
    keyword_skills = _keyword_extract(job_description)
    all_skills.update(keyword_skills)
    
    if len(all_skills) < 4:
        predicted_role = predict_role(clean_text(job_description))
        role_skills    = ROLE_SKILLS.get(predicted_role, [])
        all_skills.update(role_skills)
    
    return list(all_skills)


def _ner_extract(text: str) -> list[str]:
    try:
        entities = ner(text[:512])
        return [
            e["word"].lower().strip()
            for e in entities
            if e["entity_group"] in {"TECHNICAL", "TECHNOLOGY"} and e["score"] > 0.80
        ]
    except:
        return []


def _keyword_extract(text: str) -> list[str]:
    KNOWN_SKILLS = {
        "react", "angular", "vue", "javascript", "typescript", "html", "css",
        "node", "express", "python", "django", "flask", "fastapi", "java",
        "spring", "sql", "postgresql", "mysql", "mongodb", "redis", "docker",
        "kubernetes", "aws", "azure", "gcp", "terraform", "git", "graphql",
        "machine learning", "deep learning", "tensorflow", "pytorch", "pandas",
        "numpy", "scikit-learn", "go", "rust", "kotlin", "swift", "flutter",
        "react native", "next.js", "tailwind", "figma", "linux", "bash",
        "kafka", "spark", "airflow", "dbt", "snowflake", "elasticsearch",
        "redis", "celery", "nginx", "ci/cd", "jenkins", "github actions"
    }
    text_lower = text.lower()
    return [s for s in KNOWN_SKILLS if s in text_lower]


ROLE_SKILLS = {
    "Full Stack Developer": [
        "react", "node", "node.js", "javascript", "typescript", "postgresql", "mongodb", "docker", "aws", "html", "css", 
        "git", "express", "next.js", "graphql", "rest api", "ci/cd", "linux", "agile", "jest", "redis"
    ],
    "Frontend Developer": [
        "react", "vue", "angular", "javascript", "typescript", "html", "css", "figma", "tailwind", "next.js",
        "redux", "sass", "webpack", "babel", "rxjs", "material-ui", "bootstrap", "cypress", "web performance", "seo"
    ],
    "Backend Developer": [
        "node", "python", "java", "postgresql", "mysql", "redis", "docker", "rest api", "microservices", "kafka",
        "spring boot", "django", "fastapi", "golang", "c#", ".net", "rabbitmq", "graphql", "mongodb", "aws"
    ],
    "Data Scientist": [
        "python", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "sql", "machine learning", "statistics", "jupyter",
        "nlp", "data visualization", "matplotlib", "seaborn", "r", "predictive modeling", "ab testing", "tableau", "power bi"
    ],
    "Machine Learning Engineer": [
        "python", "tensorflow", "pytorch", "mlops", "docker", "kubernetes", "scikit-learn", "deep learning", "airflow",
        "mlflow", "keras", "hugging face", "computer vision", "nlp", "aws sagemaker", "spark", "cuda", "model deployment"
    ],
    "Data Engineer": [
        "sql", "python", "apache spark", "hadoop", "airflow", "kafka", "snowflake", "bigquery", "redshift", "etl",
        "data warehousing", "scala", "dbt", "aws glue", "postgresql", "nosql", "data pipelines", "docker", "kubernetes"
    ],
    "Cloud/DevOps Engineer": [
        "aws", "azure", "gcp", "terraform", "kubernetes", "docker", "linux", "networking", "ci/cd", "jenkins",
        "ansible", "bash", "shell scripting", "github actions", "gitlab ci", "prometheus", "grafana", "nginx", "helm", "bash"
    ],
    "Mobile App Developer": [
        "swift", "kotlin", "flutter", "react native", "ios", "android", "firebase", "xcode", "android studio",
        "dart", "objective-c", "java", "sqlite", "core data", "mobile ui", "app store optimization", "rest api"
    ],
    "Python Developer": [
        "python", "django", "flask", "fastapi", "postgresql", "redis", "celery", "rest api", "docker",
        "sqlalchemy", "pytest", "beautifulsoup", "selenium", "web scraping", "linux", "git", "bash"
    ],
    "QA/Automation Engineer": [
        "selenium", "cypress", "playwright", "appium", "postman", "jest", "pytest", "jmeter", "test automation", "ci/cd",
        "java", "python", "javascript", "sql", "jira", "agile", "manual testing", "api testing"
    ],
    "Cybersecurity Engineer": [
        "python", "linux", "bash", "networking", "wireshark", "penetration testing", "kali linux", "siem", "firewalls",
        "cryptography", "owasp", "security auditing", "aws security", "vulnerability assessment", "incident response"
    ]
}


def estimate_experience(text: str):

    matches = re.findall(r"(\d+)\s+years?", text)
    if matches:
        return max(map(int, matches))
    return 0

def extract_skills_rule_based(phrases):
    KNOWN_SKILLS = {
        "react", "nextjs", "nodejs", "express",
        "python", "fastapi", "django",
        "machine learning", "deep learning", "nlp",
        "scikitlearn", "tfidf",
        "postgresql", "mysql", "mongodb",
        "socketio", "tailwind", "git"
    }

    found = set()
        
    for p in phrases:
        if p in KNOWN_SKILLS:
            found.add(p)
    
    return list(found)