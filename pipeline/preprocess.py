import re

SECTION_PATTERNS = {
    "skills": [
        "skills",
        "technical skills",
        "tech stack",
        "expertise"
    ],
    
    "projects": [
        "projects",
        "personal projects"
    ],
    
    "experience": [
        "experience",
        "work experience",
        "internship",
        "employment"
    ],
    
    "education": [
        "education",
        "academic background",
        "academic"
    ],
    
    "certifications": [
        "certifications",
        "licenses"
    ],
    
    "achievements": [
        "achievements",
        "awards"
    ]
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = normalize_text(text)
    
    text = re.sub(r"[^a-z0-9\.\+\#\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text_structure(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\.\+\#\-]", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()
    
# def rebuild_lines(lines):
#     rebuilt = []
#     buffer = ""
    
#     SECTION_HEADERS = list(SECTION_PATTERNS.keys())

#     for line in lines:
#         line = line.strip()

#         if not line:
#             continue

#         line_lower = line.lower()

#         # preserve section headers
#         if line_lower in SECTION_HEADERS:
#             if buffer:
#                 rebuilt.append(buffer.strip())
#                 buffer = ""

#             rebuilt.append(line)
#             continue

#         word_count = len(line.split())

#         # tiny broken lines → merge
#         if word_count <= 3:
#             buffer += " " + line

#         else:
#             if buffer:
#                 rebuilt.append(buffer.strip())

#             buffer = line

#     if buffer:
#         rebuilt.append(buffer.strip())

#     return rebuilt

def split_resume_sections(text: str) -> dict[str, str]:
    lines           = text.split("\n")
    sections        = {}
    current_section = "other"
    sections[current_section] = []

    for line in lines:
        
        stripped    = line.strip()
        line_clean  = stripped.lower()
        if not line_clean:
            continue
        print("current line:", line_clean)
    
        found_header = False

        # only treat as header if line is short
        if len(line_clean.split()) < 6:
            for section, patterns in SECTION_PATTERNS.items():
                # use 'in' instead of startswith — more flexible
                if any(p in line_clean for p in patterns):
                    current_section = section
                    if current_section not in sections:
                        sections[current_section] = []
                    found_header = True
                    break

        if not found_header:
            sections.setdefault(current_section, []).append(stripped)

    return {
        k: "\n".join(v).strip()
        for k, v in sections.items()
        if v
    }

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