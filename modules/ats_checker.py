import pandas as pd

career_roles = pd.read_csv(
    "data/career_roles.csv"
)

def normalize_skill(skill):
    return " ".join(str(skill).lower().strip().split())

def calculate_ats_score(
        resume_skills,
        target_role
):

    role_skills = career_roles[
        career_roles["Role"] == target_role
    ]["Skill"].tolist()

    resume_set = {normalize_skill(skill) for skill in resume_skills}
    role_skill_map = {
        normalize_skill(skill): skill
        for skill in role_skills
    }

    matched = sorted(
        role_skill_map[skill]
        for skill in resume_set.intersection(role_skill_map)
    )

    missing = sorted(
        role_skill_map[skill]
        for skill in set(role_skill_map) - resume_set
    )

    if len(role_skills) == 0:
        score = 0
    else:
        score = round(
            (len(matched) /
             len(role_skills)) * 100,
            2
        )

    return {
        "score": score,
        "matched": matched,
        "missing": missing
    }
