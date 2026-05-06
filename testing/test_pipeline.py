SAMPLE_RESUME = """
John Doe | john@email.com | +1-234-567-8901 | github.com/johndoe
LinkedIn: linkedin.com/in/johndoe

SUMMARY
Full Stack Developer with 4 years of experience building scalable web applications.

EXPERIENCE
Senior Developer at TechCorp (2021 - 2024)
- Built React dashboards that reduced reporting time by 40%
- Led team of 5 engineers to deliver 3 major product launches
- Optimized PostgreSQL queries improving performance by 60%

SKILLS
React, Node.js, TypeScript, PostgreSQL, Docker, AWS, Python

EDUCATION
B.S. Computer Science, State University, 2020
"""

SAMPLE_JD = "Looking for a full stack developer"

import sys
import os

# Add the parent directory (ml) to sys.path so we can import 'pipeline'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.extractor import extract_skills

def test_pipeline():
    #skill extraction
    print(extract_skills(SAMPLE_RESUME))
    print(extract_skills(SAMPLE_JD))

test_pipeline()