from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_vec, jd_vec):
    score = cosine_similarity(resume_vec, jd_vec)[0][0]
    return float(score)