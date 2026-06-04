import pandas as pd

career_roles = pd.read_csv(
    "data/career_roles.csv"
)

def normalize_skill(skill):
    return " ".join(str(skill).lower().strip().split())

def recommend_careers(user_skills):

    recommendations = []
    user_skill_set = {
        normalize_skill(skill)
        for skill in user_skills
    }

    for role in career_roles["Role"].unique():

        role_skills = career_roles[
            career_roles["Role"] == role
        ]["Skill"].tolist()
        role_skill_set = {
            normalize_skill(skill)
            for skill in role_skills
        }

        matched = len(
            user_skill_set
            .intersection(role_skill_set)
        )

        total = len(role_skills)

        score = round(
            (matched / total) * 100,
            2
        )

        recommendations.append({
            "Role": role,
            "Match Score (%)": score,
            "Matched Skills": matched,
            "Required Skills": total
        })

    result = pd.DataFrame(
        recommendations
    )

    result = result.sort_values(
        by="Match Score (%)",
        ascending=False
    )

    return result
