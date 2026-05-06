from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def score_resume_section(resume_text: str, aspect: str) -> float:
    """
    Ask the model: does this resume demonstrate [aspect]?
    Returns confidence 0-1
    """
    result = classifier(
        resume_text[:1024],
        candidate_labels=[
            f"demonstrates strong {aspect}",
            f"lacks {aspect}"
        ]
    )
    # score for the positive label
    pos_idx = result["labels"].index(f"demonstrates strong {aspect}")
    return round(result["scores"][pos_idx], 3)