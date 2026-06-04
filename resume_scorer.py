def calculate_resume_score(skills):
    return min(len(skills) * 10, 100)


def calculate_advanced_resume_score(text, detected_skills):
    return min(len(detected_skills) * 5, 100)


def get_resume_feedback(score):

    if score >= 85:
        return "Excellent Resume. Industry Ready."

    elif score >= 70:
        return "Good Resume. Add more projects and certifications."

    elif score >= 50:
        return "Average Resume. Improve skills and project experience."

    else:
        return "Resume needs significant improvement."