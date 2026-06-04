def generate_roadmap(missing_skills):

    roadmap = {}

    month = 1

    for skill in missing_skills:

        roadmap[f"Month {month}"] = skill

        month += 1

    return roadmap