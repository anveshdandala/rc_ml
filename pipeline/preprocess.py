import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\.\+\#\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = normalize_text(text)
    return text.strip()

def normalize_text(text: str) -> str:
    replacements = {
        "react.js": "react",
        "next.js": "nextjs",
        "node.js": "nodejs",
        "express.js": "express",
        "vue.js": "vue",
        "three.js": "threejs",
        "c++": "cpp",
        "c#": "csharp",
        ".net": "dotnet",
        "scikit-learn": "scikitlearn",
        "tf-idf": "tfidf",
        "socket.io": "socketio"
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    return text

def generate_ngrams(words, max_n=3):
    ngrams = []
    for i in range(len(words)):
        for j in range(1, max_n + 1):
            if i + j <= len(words):
                ngrams.append(" ".join(words[i:i+j]))
    return ngrams