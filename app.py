from pathlib import Path
from tempfile import NamedTemporaryFile

import altair as alt
import pandas as pd
import streamlit as st
import re

try:
    import pdfplumber
except ImportError:  # The app still works for TXT resumes.
    pdfplumber = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"
LEGACY_ASSET_DIR = BASE_DIR / "assests"

st.set_page_config(
    page_title="AI Career Intelligence System",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = ASSET_DIR / "style.css"
    if not css_path.exists():
        css_path = LEGACY_ASSET_DIR / "style.css"

    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    career_info = pd.read_csv(DATA_DIR / "career_info.csv")
    career_roles = pd.read_csv(DATA_DIR / "career_roles.csv").dropna()
    courses = pd.read_csv(DATA_DIR / "courses.csv").dropna()
    skills_trends = pd.read_csv(DATA_DIR / "skills_trends.csv").dropna()

    for frame in (career_info, career_roles, courses, skills_trends):
        for column in frame.select_dtypes(include="object").columns:
            frame[column] = frame[column].astype(str).str.strip()

    return career_info, career_roles, courses, skills_trends


def normalize(value: str) -> str:
    return " ".join(str(value).lower().strip().split())


def skill_catalog(career_roles: pd.DataFrame) -> dict[str, str]:
    return {normalize(skill): skill for skill in career_roles["Skill"].dropna().unique()}


def extract_resume_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == ".pdf" and pdfplumber is not None:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        text_parts = []
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    return ""


def detect_skills(text: str, catalog: dict[str, str]) -> list[str]:
    normalized_text = normalize(text)
    found = []

    for normalized_skill, display_skill in catalog.items():
        pattern = rf"(?<!\w){re.escape(normalized_skill)}(?!\w)"
        if re.search(pattern, normalized_text):
            found.append(display_skill)

    return sorted(set(found))


def role_required_skills(career_roles: pd.DataFrame, role: str) -> list[str]:
    return sorted(career_roles.loc[career_roles["Role"] == role, "Skill"].dropna().unique())


def score_role(user_skills: list[str], required_skills: list[str]) -> dict[str, object]:
    user_set = {normalize(skill) for skill in user_skills}
    required_map = {normalize(skill): skill for skill in required_skills}

    matched = sorted(required_map[key] for key in user_set.intersection(required_map))
    missing = sorted(required_map[key] for key in set(required_map) - user_set)
    score = round((len(matched) / len(required_skills)) * 100, 1) if required_skills else 0

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "required": required_skills,
    }


def recommendation_table(career_roles: pd.DataFrame, user_skills: list[str]) -> pd.DataFrame:
    rows = []

    for role in sorted(career_roles["Role"].unique()):
        required = role_required_skills(career_roles, role)
        result = score_role(user_skills, required)
        rows.append(
            {
                "Role": role,
                "Match %": result["score"],
                "Matched Skills": len(result["matched"]),
                "Required Skills": len(required),
                "Missing Skills": ", ".join(result["missing"][:6]),
            }
        )

    return pd.DataFrame(rows).sort_values("Match %", ascending=False)


def badge_list(items: list[str], tone: str = "info", empty_text: str = "Nothing to show yet") -> None:
    if not items:
        st.markdown(f"<p class='muted'>{empty_text}</p>", unsafe_allow_html=True)
        return

    badges = "".join(f"<span class='skill-badge {tone}'>{item}</span>" for item in items)
    st.markdown(f"<div class='badge-wrap'>{badges}</div>", unsafe_allow_html=True)


def section_header(title: str, eyebrow: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <span>{eyebrow}</span>
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <p>{label}</p>
            <h3>{value}</h3>
            <span>{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_tile(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="feature-tile">
            <div class="tile-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def roadmap_step(number: str, title: str, body: str, tone: str = "cyan") -> None:
    st.markdown(
        f"""
        <div class="roadmap-step {tone}">
            <div class="roadmap-number">{number}</div>
            <div>
                <h3>{title}</h3>
                <p>{body}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


load_css()
career_info, career_roles, courses, skills_trends = load_data()
catalog = skill_catalog(career_roles)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 3 4 7v6c0 4.7 3.3 7.6 8 9 4.7-1.4 8-4.3 8-9V7l-8-4Z"/>
                    <path d="M8.5 12.5h7M12 8.5v7"/>
                </svg>
            </div>
            <div>
                <h1>CareerPilot AI</h1>
                <p>Smart career command center</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Resume Analysis",
            "Career Recommendation",
            "Skill Gap",
            "Course Finder",
            "Market Trends",
            "Roadmap",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-note'>Build a stronger profile with resume scoring, role matching, skill gaps, learning paths, and market signals.</div>", unsafe_allow_html=True)


if menu == "Dashboard":
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Career Intelligence Platform</div>
            <h1>AI-Powered Career Decision System</h1>
            <div class="hero-action">Upload Analyze Advance</div>
            <p>Upload your resume, choose a target career role, and instantly understand your ATS score, matched skills, missing skills, best-fit careers, recommended courses, market trends, and a step-by-step roadmap to become job-ready.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Career Roles", str(len(career_info)), "Mapped role database")
    with col2:
        metric_card("Skills", str(career_roles["Skill"].nunique()), "Unique skill signals")
    with col3:
        metric_card("Courses", str(len(courses)), "Learning recommendations")
    with col4:
        metric_card("Trending Skills", str(len(skills_trends)), "Market demand signals")

    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    with feature_col1:
        feature_tile("CV", "Resume Scanner", "Detects skills and compares your profile with target-role requirements.")
    with feature_col2:
        feature_tile("FIT", "Role Matching", "Ranks careers by match percentage using your current skill stack.")
    with feature_col3:
        feature_tile("GAP", "Skill Gap Map", "Shows what is missing and what to learn next for better readiness.")
    with feature_col4:
        feature_tile("MAP", "Career Roadmap", "Turns missing skills into a focused preparation path.")

    left, right = st.columns([1.25, 1])

    with left:
        section_header("Top Career Opportunities", "role intelligence", "Compare salary bands, demand, growth, and industry direction.")
        st.dataframe(career_info, use_container_width=True, hide_index=True)

    with right:
        section_header("Demand Leaders", "market pulse", "Highest scoring skills in the current dataset.")
        top_skills = skills_trends.sort_values("DemandScore", ascending=False).head(8)
        chart = (
            alt.Chart(top_skills)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("DemandScore:Q", title=None),
                y=alt.Y("Skill:N", sort="-x", title=None),
                color=alt.Color("Category:N", legend=None),
                tooltip=["Skill", "DemandScore", "Category"],
            )
            .properties(height=330)
        )
        st.altair_chart(chart, use_container_width=True)


elif menu == "Resume Analysis":
    section_header("Resume Analysis", "profile scanner", "Upload a resume and compare detected skills against a target role.")

    input_col, score_col = st.columns([1.1, 0.9])

    with input_col:
        uploaded_file = st.file_uploader("Upload resume", type=["txt", "pdf"])
        target_role = st.selectbox("Target career role", sorted(career_roles["Role"].unique()))

    resume_text = extract_resume_text(uploaded_file)

    if uploaded_file and not resume_text:
        st.warning("I could not read this file. Please upload a text-based PDF or TXT resume.")

    if resume_text:
        detected = detect_skills(resume_text, catalog)
        required = role_required_skills(career_roles, target_role)
        ats = score_role(detected, required)

        with score_col:
            st.markdown("<div class='score-panel'>", unsafe_allow_html=True)
            st.metric("ATS Match Score", f"{ats['score']}%")
            st.progress(int(ats["score"]))
            st.caption(f"{len(ats['matched'])} of {len(required)} target-role skills detected")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Matched Skills")
        badge_list(ats["matched"], "success", "No target-role skills detected yet")

        st.markdown("### Missing Skills")
        badge_list(ats["missing"], "danger", "You already cover this role's skill list")

        preview_col, roadmap_col = st.columns([1, 1])
        with preview_col:
            st.markdown("### Resume Preview")
            st.text_area("Resume text", resume_text[:3500], height=260, label_visibility="collapsed")
        with roadmap_col:
            st.markdown("### Suggested Learning Focus")
            focus_skills = ats["missing"][:5]
            if focus_skills:
                related_courses = courses[courses["Skill"].isin(focus_skills)]
                st.dataframe(related_courses.head(8), use_container_width=True, hide_index=True)
            else:
                st.success("Strong match. Focus on projects, proof of impact, and interview preparation.")


elif menu == "Career Recommendation":
    section_header("Career Recommendation", "fit engine", "Select your skills and discover your strongest role matches.")

    user_skills = st.multiselect("Your skills", sorted(career_roles["Skill"].unique()))

    if user_skills:
        result = recommendation_table(career_roles, user_skills)
        top = result.head(5)

        st.markdown("### Best Matches")
        st.dataframe(top, use_container_width=True, hide_index=True)

        chart = (
            alt.Chart(top)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("Match %:Q", title="Match percentage", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Role:N", sort="-x", title=None),
                color=alt.value("#22d3ee"),
                tooltip=["Role", "Match %", "Matched Skills", "Required Skills"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Choose a few skills to generate career matches.")


elif menu == "Skill Gap":
    section_header("Skill Gap Analysis", "readiness check", "Compare your current skill stack with the skills required for a target role.")

    role = st.selectbox("Choose career", sorted(career_roles["Role"].unique()))
    user_skills = st.multiselect("Your skills", sorted(career_roles["Skill"].unique()))

    required = role_required_skills(career_roles, role)
    gap = score_role(user_skills, required)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Readiness", f"{gap['score']}%", "Target role fit")
    with col2:
        metric_card("Matched", str(len(gap["matched"])), "Skills already covered")
    with col3:
        metric_card("Missing", str(len(gap["missing"])), "Skills to build next")

    st.markdown("### Required Skills")
    badge_list(required, "info")
    st.markdown("### Missing Skills")
    badge_list(gap["missing"], "danger", "No missing skills for this role")


elif menu == "Course Finder":
    section_header("Course Finder", "learning path", "Find courses that support the skill you want to build next.")

    skill = st.selectbox("Select skill", sorted(courses["Skill"].unique()))
    level = st.selectbox("Level", ["All"] + sorted(courses["Level"].dropna().unique()))

    filtered = courses[courses["Skill"] == skill]
    if level != "All":
        filtered = filtered[filtered["Level"] == level]

    st.dataframe(filtered, use_container_width=True, hide_index=True)


elif menu == "Market Trends":
    section_header("Market Trends", "skill demand", "Explore which skills are strongest in the current career dataset.")

    sorted_trends = skills_trends.sort_values("DemandScore", ascending=False)
    st.dataframe(sorted_trends, use_container_width=True, hide_index=True)

    chart = (
        alt.Chart(sorted_trends)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("Skill:N", sort="-y", title=None),
            y=alt.Y("DemandScore:Q", title="Demand score"),
            color=alt.Color("Category:N", title="Category"),
            tooltip=["Skill", "DemandScore", "Category"],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)


elif menu == "Roadmap":
    section_header("Career Roadmap", "growth plan", "Build a practical path from current skills to target-role readiness.")

    role = st.selectbox("Target role", sorted(career_roles["Role"].unique()))
    user_skills = st.multiselect("Your current skills", sorted(career_roles["Skill"].unique()))

    required = role_required_skills(career_roles, role)
    gap = score_role(user_skills, required)
    missing = gap["missing"]

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Current Readiness", f"{gap['score']}%", "Role preparation score")
    with col2:
        metric_card("Skills Covered", str(len(gap["matched"])), "Already in your profile")
    with col3:
        metric_card("Skills To Build", str(len(missing)), "Priority learning list")

    st.markdown("### Recommended Roadmap")

    if not user_skills:
        roadmap_step("01", "Add Your Current Skills", "Select your skills above so the roadmap can personalize your next steps.", "cyan")
        roadmap_step("02", "Choose A Target Role", "Pick the role you want to prepare for and compare your readiness.", "violet")
        roadmap_step("03", "Follow The Learning Plan", "Use missing skills and course recommendations to build a focused study path.", "green")
    elif not missing:
        roadmap_step("01", "Strengthen Portfolio Proof", "Convert your skills into projects, dashboards, case studies, or GitHub work.", "green")
        roadmap_step("02", "Prepare Interview Stories", "Write impact-focused examples for projects, problem solving, and teamwork.", "cyan")
        roadmap_step("03", "Apply Strategically", "Target roles that match your skill stack and customize your resume for each job description.", "violet")
    else:
        priority = missing[:3]
        next_skills = missing[3:6]
        roadmap_step("01", "Build Priority Skills", f"Start with: {', '.join(priority)}.", "red")
        roadmap_step("02", "Create Proof Projects", "Build one small project that uses your target-role skills together in a realistic workflow.", "cyan")
        if next_skills:
            roadmap_step("03", "Expand Your Stack", f"After the basics, continue with: {', '.join(next_skills)}.", "violet")
        else:
            roadmap_step("03", "Polish Resume Positioning", "Add measurable project outcomes and role-specific keywords to improve ATS strength.", "green")

    st.markdown("### Missing Skills")
    badge_list(missing, "danger", "No missing skills for this role")

    if missing:
        st.markdown("### Courses For Next Skills")
        st.dataframe(courses[courses["Skill"].isin(missing[:5])].head(10), use_container_width=True, hide_index=True)
