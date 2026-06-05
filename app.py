from pathlib import Path
from tempfile import NamedTemporaryFile

import altair as alt
import pandas as pd
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pdfplumber
except ImportError:  # The app still works for TXT resumes.
    pdfplumber = None

try:
    from docx import Document
except ImportError:  # Word upload is optional when dependency is unavailable.
    Document = None

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


def extract_docx_text(uploaded_file) -> str:
    if Document is None:
        return ""

    document = Document(uploaded_file)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]

    table_text = []
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    table_text.append(cell.text)

    return "\n".join(paragraphs + table_text)


def extract_plain_text(uploaded_file) -> str:
    raw_bytes = uploaded_file.getvalue()

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding, errors="ignore")
        except UnicodeError:
            continue

    return ""


def extract_resume_text(uploaded_file) -> tuple[str, str]:
    if uploaded_file is None:
        return "", ""

    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if suffix in {".txt", ".md", ".csv", ".rtf"}:
            return extract_plain_text(uploaded_file), ""

        if suffix == ".docx":
            text = extract_docx_text(uploaded_file)
            if text:
                return text, ""
            return "", "This Word file could not be read. Please try saving it as PDF or DOCX again."

        if suffix == ".doc":
            return "", "Old .doc files are not supported directly. Please save it as .docx or PDF and upload again."

        if suffix == ".pdf" and pdfplumber is not None:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            text_parts = []
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            text = "\n".join(text_parts).strip()
            if text:
                return text, ""
            return "", "This PDF does not contain readable text. It may be a scanned resume image."

        if suffix == ".pdf":
            return "", "PDF support is not available because pdfplumber is missing."

        fallback_text = extract_plain_text(uploaded_file).strip()
        if fallback_text:
            return fallback_text, ""

        return "", "This file type uploaded, but I could not extract readable resume text from it."

    except Exception:
        return "", "The file uploaded, but there was a problem reading it. Please try PDF, DOCX, or TXT format."


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


def semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )

    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
        return round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100, 1)
    except ValueError:
        return 0.0


def role_document(career_roles: pd.DataFrame, role: str) -> str:
    skills = role_required_skills(career_roles, role)
    return f"{role} " + " ".join(skills)


def semantic_recommendation_table(career_roles: pd.DataFrame, user_skills: list[str], resume_text: str = "") -> pd.DataFrame:
    user_profile = " ".join(user_skills) + " " + resume_text
    rows = []

    for role in sorted(career_roles["Role"].unique()):
        required = role_required_skills(career_roles, role)
        keyword_result = score_role(user_skills, required)
        semantic_score = semantic_similarity(user_profile, role_document(career_roles, role))
        final_score = round((keyword_result["score"] * 0.55) + (semantic_score * 0.45), 1)

        rows.append(
            {
                "Role": role,
                "Overall Match %": final_score,
                "Skill Match %": keyword_result["score"],
                "Semantic Fit %": semantic_score,
                "Missing Skills": ", ".join(keyword_result["missing"][:6]),
            }
        )

    return pd.DataFrame(rows).sort_values("Overall Match %", ascending=False)


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

    st.markdown("<div class='sidebar-note'>Build a stronger profile with semantic ATS scoring, role matching, skill gaps, learning paths, and market signals.</div>", unsafe_allow_html=True)


if menu == "Dashboard":
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Career Intelligence Platform</div>
            <h1>AI-Powered Career Decision System</h1>
            <div class="hero-action">Upload Analyze Advance</div>
            <p>Upload your resume, add a job description if available, and instantly understand your semantic ATS score, role fit, matched skills, missing skills, recommended courses, market trends, and a step-by-step roadmap to become job-ready.</p>
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
        feature_tile("CV", "Resume Scanner", "Reads PDF, Word, and text resumes for semantic ATS analysis.")
    with feature_col2:
        feature_tile("ATS", "Semantic ATS", "Compares your resume with a target role or job description using semantic similarity.")
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
    section_header(
        "Resume Analysis",
        "semantic ATS scanner",
        "Upload your resume, add an optional job description, and get a semantic ATS score with clear improvement guidance.",
    )

    input_col, jd_col = st.columns([1, 1])

    with input_col:
        uploaded_file = st.file_uploader(
            "Upload resume",
            type=["pdf", "docx", "doc", "txt", "md", "rtf", "csv"],
            help="Supported formats: PDF, DOCX, DOC, TXT, Markdown, RTF, and text-like files.",
        )
        target_role = st.selectbox("Target career role", sorted(career_roles["Role"].unique()))

    with jd_col:
        jd_file = st.file_uploader(
            "Upload job description",
            type=["pdf", "docx", "doc", "txt", "md", "rtf", "csv"],
            key="resume_analysis_jd_upload",
            help="Optional. Paste a JD below or upload one here for more accurate ATS scoring.",
        )
        pasted_jd = st.text_area(
            "Paste job description",
            height=150,
            placeholder="Paste job description here for semantic ATS comparison.",
        )

    resume_text, upload_message = extract_resume_text(uploaded_file)
    jd_text_from_file, jd_message = extract_resume_text(jd_file)
    jd_text = pasted_jd.strip() or jd_text_from_file

    if uploaded_file and not resume_text:
        st.warning(upload_message or "I could not read this file. Please upload PDF, DOCX, or TXT resume.")

    if jd_file and not jd_text_from_file and not pasted_jd.strip():
        st.warning(jd_message or "I could not read this job description file.")

    if resume_text:
        st.session_state["resume_text"] = resume_text
        st.session_state["resume_file_name"] = uploaded_file.name if uploaded_file else "Uploaded resume"

        detected = detect_skills(resume_text, catalog)
        st.session_state["resume_skills"] = detected
        required = role_required_skills(career_roles, target_role)
        role_skill_score = score_role(detected, required)
        role_semantic_score = semantic_similarity(resume_text, role_document(career_roles, target_role))

        if jd_text:
            jd_skills = detect_skills(jd_text, catalog)
            jd_gap = score_role(detected, jd_skills)
            jd_semantic_score = semantic_similarity(resume_text, jd_text)
            skill_score = jd_gap["score"] if jd_skills else role_skill_score["score"]
            ats_score = round((jd_semantic_score * 0.7) + (skill_score * 0.3), 1)
            matched_skills = jd_gap["matched"]
            missing_skills = jd_gap["missing"]
            semantic_label = "Resume vs JD meaning"
            skill_label = "JD skills covered"
        else:
            skill_score = role_skill_score["score"]
            ats_score = round((role_semantic_score * 0.7) + (skill_score * 0.3), 1)
            matched_skills = role_skill_score["matched"]
            missing_skills = role_skill_score["missing"]
            semantic_label = "Resume vs target role"
            skill_label = "Target-role skills covered"

        score_col, _ = st.columns([0.8, 1.2])
        with score_col:
            st.markdown("<div class='score-panel'>", unsafe_allow_html=True)
            st.metric("ATS Match Score", f"{ats_score}%")
            st.progress(min(int(ats_score), 100))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Skills Found In Resume")
        badge_list(detected, "info", "No known skills detected yet")

        matched_col, missing_col = st.columns(2)
        with matched_col:
            st.markdown("### Matched Skills")
            badge_list(matched_skills, "success", "No matched skills detected yet")
        with missing_col:
            st.markdown("### Missing / Weak Skills")
            badge_list(missing_skills, "danger", "No missing skills detected")

        st.markdown("### Smart ATS Suggestions")
        if ats_score >= 75:
            roadmap_step("01", "Strong ATS Alignment", "Your resume is semantically close to the role. Improve by adding measurable project outcomes and exact job-title language.", "green")
            roadmap_step("02", "Prepare Proof", "Be ready to explain projects, tools, and responsibilities that connect to this role.", "cyan")
        elif ats_score >= 45:
            roadmap_step("01", "Improve Role Alignment", "Your resume has partial fit. Add stronger project bullets, relevant tools, and missing skills from the list.", "violet")
            roadmap_step("02", "Rewrite Key Sections", "Tune your summary, skills, and project descriptions to match the target role or JD responsibilities.", "cyan")
        else:
            roadmap_step("01", "Build Missing Foundations", "The current resume is not close enough semantically. Learn the missing skills and add proof projects.", "red")
            roadmap_step("02", "Use A Better-Fit Target", "Try a target role closer to your current skill profile while you build toward this one.", "violet")

        preview_col, focus_col = st.columns([1, 1])
        with preview_col:
            st.markdown("### Resume Preview")
            st.text_area("Resume text", resume_text[:3500], height=260, label_visibility="collapsed")
        with focus_col:
            st.markdown("### Learning Focus")
            focus_skills = missing_skills[:5]
            if focus_skills:
                related_courses = courses[courses["Skill"].isin(focus_skills)]
                st.dataframe(related_courses.head(8), use_container_width=True, hide_index=True)
            else:
                st.success("Strong match. Focus on projects, proof of impact, and interview preparation.")

        if jd_text:
            st.markdown("### Job Description Preview")
            st.text_area("JD text", jd_text[:2500], height=180, label_visibility="collapsed")
    else:
        st.info("Upload a resume to generate semantic ATS analysis. Add a job description for the most accurate score.")


elif menu == "Career Recommendation":
    section_header("Career Recommendation", "semantic fit engine", "Select your skills and discover your strongest role matches using skill overlap and semantic similarity.")

    user_skills = st.multiselect("Your skills", sorted(career_roles["Skill"].unique()))
    resume_context = st.text_area(
        "Optional profile summary or resume text",
        height=140,
        placeholder="Paste resume summary, project details, or experience text for stronger semantic role matching.",
    )

    if user_skills or resume_context.strip():
        result = semantic_recommendation_table(career_roles, user_skills, resume_context)
        top = result.head(5)

        st.markdown("### Best Semantic Matches")
        st.dataframe(top, use_container_width=True, hide_index=True)

        chart = (
            alt.Chart(top)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("Overall Match %:Q", title="Overall match percentage", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Role:N", sort="-x", title=None),
                color=alt.value("#22d3ee"),
                tooltip=["Role", "Overall Match %", "Skill Match %", "Semantic Fit %"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Choose skills or paste profile text to generate semantic career matches.")


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
