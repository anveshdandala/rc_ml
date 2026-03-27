# pipeline/ranker.py
from sklearn.metrics.pairwise import cosine_similarity

def rank_resume(resume_vec, jd_vec):
    return cosine_similarity(resume_vec, jd_vec)[0][0]