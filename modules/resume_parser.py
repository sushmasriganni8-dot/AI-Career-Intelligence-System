import re
import pandas as pd

# Load skills from career_roles.csv
career_roles = pd.read_csv("data/career_roles.csv")

ALL_SKILLS = set(
    career_roles["Skill"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
)

def extract_skills(resume_text):

    resume_text = resume_text.lower()

    found_skills = []

    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, resume_text):
            found_skills.append(skill.title())

    return sorted(list(set(found_skills)))


def resume_summary(resume_text):

    skills = extract_skills(resume_text)

    return {
        "skills": skills,
        "total_skills": len(skills)
    }