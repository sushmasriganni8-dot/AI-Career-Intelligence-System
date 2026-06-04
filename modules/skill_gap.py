import pandas as pd

career_roles = pd.read_csv(
    "data/career_roles.csv"
)

def normalize_skill(skill):
    return " ".join(str(skill).lower().strip().split())

def analyze_skill_gap(
        user_skills,
        target_role
):

    required_skills = career_roles[
        career_roles["Role"] == target_role
    ]["Skill"].tolist()

    user_skill_set = {
        normalize_skill(skill)
        for skill in user_skills
    }
    required_skill_map = {
        normalize_skill(skill): skill
        for skill in required_skills
    }

    matched = sorted(
        required_skill_map[skill]
        for skill in user_skill_set.intersection(required_skill_map)
    )

    missing = sorted(
        required_skill_map[skill]
        for skill in set(required_skill_map) - user_skill_set
    )

    readiness = round(
        (
            len(matched)
            / len(required_skills)
        ) * 100,
        2
    )

    return {
        "readiness": readiness,
        "matched": matched,
        "missing": missing,
        "required": required_skills
    }
