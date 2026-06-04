import pandas as pd

courses_df = pd.read_csv(
    "data/courses.csv"
)

def recommend_courses(
        missing_skills
):

    recommendations = []

    for skill in missing_skills:

        course = courses_df[
            courses_df["Skill"]
            .str.lower()
            ==
            skill.lower()
        ]

        if not course.empty:

            recommendations.append(
                course
            )

    if recommendations:

        return pd.concat(
            recommendations,
            ignore_index=True
        )

    return pd.DataFrame()