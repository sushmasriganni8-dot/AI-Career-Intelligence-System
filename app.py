from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

try:
    import pdfplumber
except ImportError:  # The app still works for TXT resumes.
    pdfplumber = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ImportError:
    pdfminer_extract_text = None

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


def extract_docx_text(uploaded_file) -> tuple[str, str]:
    if Document is None:
        return "", "DOCX parser (python-docx) is not installed or loaded."

    try:
        document = Document(BytesIO(uploaded_file.getvalue()))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]

        table_text = []
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        table_text.append(cell.text)

        text = "\n".join(paragraphs + table_text).strip()
        if text:
            return text, ""
        return "", "The Word file is empty or contains no readable text."
    except Exception as e:
        return "", f"Error reading Word file: {e}"


def extract_plain_text(uploaded_file) -> str:
    raw_bytes = uploaded_file.getvalue()

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding, errors="ignore")
        except UnicodeError:
            continue

    return ""


def extract_pdf_text(uploaded_file) -> tuple[str, str]:
    pdf_bytes = uploaded_file.getvalue()
    text_parts = []
    errors = []

    if pdfium is not None:
        try:
            with pdfium.PdfDocument(pdf_bytes) as doc:
                for page in doc:
                    textpage = page.get_textpage()
                    text_parts.append(textpage.get_text_range() or "")
                    textpage.close()
            text = "\n".join(text_parts).strip()
            if text:
                return text, ""
            errors.append("pypdfium2 extracted no text")
        except Exception as e:
            errors.append(f"pypdfium2 error: {e}")
    else:
        errors.append("pypdfium2 is not installed/loaded")

    if pdfplumber is not None:
        try:
            text_parts_plumber = []
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text_parts_plumber.append(page.extract_text() or "")
            text = "\n".join(text_parts_plumber).strip()
            if text:
                return text, ""
            errors.append("pdfplumber extracted no text")
        except Exception as e:
            errors.append(f"pdfplumber error: {e}")
    else:
        errors.append("pdfplumber is not installed/loaded")

    if pdfminer_extract_text is not None:
        try:
            text = pdfminer_extract_text(BytesIO(pdf_bytes)).strip()
            if text:
                return text, ""
            errors.append("pdfminer extracted no text")
        except Exception as e:
            errors.append(f"pdfminer error: {e}")
    else:
        errors.append("pdfminer is not installed/loaded")

    error_summary = " | ".join(errors)
    return "", f"Could not extract text from PDF. Diagnostic details: {error_summary}"


def detect_uploaded_file_type(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if suffix:
        return suffix

    if raw_bytes.startswith(b"%PDF"):
        return ".pdf"

    if raw_bytes.startswith(b"PK") and b"word/" in raw_bytes[:5000]:
        return ".docx"

    return ".txt"


def extract_resume_text(uploaded_file) -> tuple[str, str]:
    if uploaded_file is None:
        return "", ""

    suffix = detect_uploaded_file_type(uploaded_file)

    try:
        if suffix in {".txt", ".md", ".csv", ".rtf"}:
            text = extract_plain_text(uploaded_file).strip()
            if text:
                return text, ""
            return "", "The plain text file is empty."

        if suffix == ".docx":
            return extract_docx_text(uploaded_file)

        if suffix == ".doc":
            return "", "Old .doc files are not supported directly. Please save it as .docx or PDF and upload again."

        if suffix == ".pdf":
            return extract_pdf_text(uploaded_file)

        fallback_text = extract_plain_text(uploaded_file).strip()
        if fallback_text:
            return fallback_text, ""

        return "", f"Unsupported file type: {suffix}. Please try PDF, DOCX, or TXT format."

    except Exception as e:
        return "", f"General error reading file: {e}. Please try PDF, DOCX, or TXT format."


SYNONYMS = {
    "aws": ["aws", "amazon web services", "amazon web service"],
    "azure": ["azure", "microsoft azure"],
    "ci/cd": ["ci/cd", "ci-cd", "continuous integration", "continuous deployment", "github actions", "jenkins"],
    "css": ["css", "css3"],
    "dsa": ["dsa", "data structures", "algorithms", "data structures and algorithms"],
    "data visualization": ["data visualization", "data-visualization", "dataviz", "tableau", "power bi", "matplotlib", "seaborn"],
    "deep learning": ["deep learning", "dl", "deeplearning"],
    "ethical hacking": ["ethical hacking", "penetration testing", "pen testing", "pen-testing", "white hat"],
    "gcp": ["gcp", "google cloud platform", "google cloud"],
    "github actions": ["github actions", "github-actions"],
    "html": ["html", "html5"],
    "javascript": ["javascript", "js", "ecmascript"],
    "kubernetes": ["kubernetes", "k8s"],
    "llm": ["llm", "large language model", "large language models", "llms"],
    "machine learning": ["machine learning", "ml", "machinelearning"],
    "nlp": ["nlp", "natural language processing"],
    "oop": ["oop", "object oriented programming", "object-oriented"],
    "rest apis": ["rest api", "rest apis", "restful api", "restful apis", "restful"],
    "react": ["react", "reactjs", "react.js"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "tailwind css": ["tailwind css", "tailwind"],
    "vector databases": ["vector database", "vector databases", "vector db", "vector dbs", "chromadb", "pinecone"],
    "web3": ["web3", "web 3", "web3.0"],
    "wireframing": ["wireframe", "wireframes", "wireframing"],
}

def detect_skills(text: str, catalog: dict[str, str]) -> list[str]:
    normalized_text = normalize(text)
    found = []

    for normalized_skill, display_skill in catalog.items():
        # Check direct match
        pattern = rf"(?<!\w){re.escape(normalized_skill)}(?!\w)"
        if re.search(pattern, normalized_text):
            found.append(display_skill)
            continue
        
        # Check synonyms
        syns = SYNONYMS.get(normalized_skill)
        if syns:
            matched = False
            for syn in syns:
                syn_norm = normalize(syn)
                pattern_syn = rf"(?<!\w){re.escape(syn_norm)}(?!\w)"
                if re.search(pattern_syn, normalized_text):
                    found.append(display_skill)
                    matched = True
                    break
            if matched:
                continue

    return sorted(set(found))


def analyze_resume_structure(text: str) -> dict[str, object]:
    normalized = text.lower()
    
    # Check for email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    has_email = bool(re.search(email_pattern, text))
    
    # Check for links (LinkedIn, GitHub)
    has_linkedin = "linkedin.com" in normalized
    has_github = "github.com" in normalized
    
    # Check for standard sections
    sections = {
        "Experience": ["experience", "work history", "employment", "professional background", "work experience"],
        "Education": ["education", "academic", "university", "college", "degree", "academic background"],
        "Skills": ["skills", "technical skills", "core competencies", "technologies", "key skills"],
        "Projects": ["projects", "personal projects", "key projects", "portfolio", "academic projects"],
    }
    
    found_sections = {}
    for sec_name, keywords in sections.items():
        found = False
        for kw in keywords:
            if rf"\b{kw}\b" in normalized:
                found = True
                break
        found_sections[sec_name] = found
        
    # Word count check
    words = text.split()
    word_count = len(words)
    if word_count < 300:
        word_score = 50
        word_msg = "Too short (under 300 words). Add detail to your experience."
    elif word_count > 1200:
        word_score = 70
        word_msg = "Too long (over 1200 words). Keep it concise (1-2 pages)."
    else:
        word_score = 100
        word_msg = "Perfect length (300-1200 words)."
        
    # Action verbs check
    action_verbs = [
        "led", "managed", "developed", "implemented", "designed", "created", 
        "optimized", "increased", "reduced", "analyzed", "built", "engineered",
        "formulated", "collaborated", "spearheaded", "accelerated", "integrated",
        "automated", "streamlined", "solved"
    ]
    found_verbs = [v for v in action_verbs if rf"\b{v}\b" in normalized]
    verb_count = len(found_verbs)
    if verb_count >= 8:
        verb_score = 100
        verb_msg = f"Excellent use of action verbs ({verb_count} found)."
    elif verb_count >= 4:
        verb_score = 80
        verb_msg = f"Good use of action verbs ({verb_count} found). Boost with more impact-focused terms."
    else:
        verb_score = 50
        verb_msg = f"Weak use of action verbs ({verb_count} found). Boost with action-driven terms."

    contact_score = 100 if (has_email and (has_linkedin or has_github)) else (75 if has_email else 30)
    section_score = (sum(found_sections.values()) / len(sections)) * 100
    
    overall_structure_score = round((contact_score * 0.25) + (section_score * 0.35) + (word_score * 0.20) + (verb_score * 0.20), 1)
    
    return {
        "overall_score": overall_structure_score,
        "has_email": has_email,
        "has_linkedin": has_linkedin,
        "has_github": has_github,
        "found_sections": found_sections,
        "word_count": word_count,
        "word_msg": word_msg,
        "verb_count": verb_count,
        "verb_msg": verb_msg,
        "found_verbs": sorted(list(set(found_verbs))),
    }


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


def analyze_semantic_matching(text_a: str, text_b: str) -> dict[str, object]:
    if not text_a.strip() or not text_b.strip():
        return {
            "score": 0.0,
            "vocab_count": 0,
            "top_terms": [],
            "feature_dims": 0
        }

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )

    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
        similarity = round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100, 1)
        
        # Get vocabulary terms and find overlap
        vocab = vectorizer.get_feature_names_out()
        
        # Multiply row vectors to see which features contribute most to similarity
        row_a = matrix[0].toarray()[0]
        row_b = matrix[1].toarray()[0]
        contributions = row_a * row_b
        
        # Sort contributions to find top matching terms
        term_contributions = []
        for idx, contrib in enumerate(contributions):
            if contrib > 0:
                term_contributions.append((vocab[idx], float(contrib)))
                
        # Sort by contribution value descending
        term_contributions.sort(key=lambda x: x[1], reverse=True)
        top_terms = [term[0] for term in term_contributions[:12]]
        
        return {
            "score": similarity,
            "vocab_count": len(vocab),
            "top_terms": top_terms,
            "feature_dims": matrix.shape[1]
        }
    except Exception:
        return {
            "score": 0.0,
            "vocab_count": 0,
            "top_terms": [],
            "feature_dims": 0
        }

def semantic_similarity(text_a: str, text_b: str) -> float:
    return analyze_semantic_matching(text_a, text_b)["score"]


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

    st.markdown("")
    st.markdown("", unsafe_allow_html=True)
    status_pdfium = "✅ pypdfium2 (Primary)" if pdfium is not None else "❌ pypdfium2 (Missing)"
    status_pdfplumber = "✅ pdfplumber (Fallback)" if pdfplumber is not None else "❌ pdfplumber (Missing)"
    status_pdfminer = "✅ pdfminer (Fallback)" if pdfminer_extract_text is not None else "❌ pdfminer (Missing)"
    status_docx = "✅ python-docx" if Document is not None else "❌ python-docx (Missing)"
    st.markdown(
        f"<div style='display: none;'>"
        f"{status_pdfium}<br>{status_pdfplumber}<br>{status_pdfminer}<br>{status_docx}"
        f"</div>",
        unsafe_allow_html=True,
    )


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
        st.dataframe(
            career_info,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Role": st.column_config.TextColumn("Career Role", help="Professional role title"),
                "Salary": st.column_config.TextColumn("Average Salary", help="Typical salary bracket"),
                "Demand": st.column_config.TextColumn("Demand Level", help="Current job market demand"),
                "Growth": st.column_config.TextColumn("Annual Growth", help="Projected industry growth rate"),
                "Industry": st.column_config.TextColumn("Sector / Industry", help="Core target sector"),
            }
        )

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
            type=None,
            help="Upload any resume file. Best supported: PDF, DOCX, TXT, Markdown, RTF, CSV.",
        )
        target_role = st.selectbox("Target career role", sorted(career_roles["Role"].unique()))

    with jd_col:
        jd_file = st.file_uploader(
            "Upload job description",
            type=None,
            key="resume_analysis_jd_upload",
            help="Optional. Upload any JD file or paste the job description below.",
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
        
        # Structure analysis
        structure_results = analyze_resume_structure(resume_text)
        structure_score = structure_results["overall_score"]

        if jd_text:
            jd_skills = detect_skills(jd_text, catalog)
            jd_gap = score_role(detected, jd_skills)
            semantic_results = analyze_semantic_matching(resume_text, jd_text)
            skill_score = jd_gap["score"] if jd_skills else role_skill_score["score"]
            matched_skills = jd_gap["matched"]
            missing_skills = jd_gap["missing"]
            semantic_label = "Resume vs Job Description Relevance"
            skill_label = "Job Description Skills Match"
        else:
            semantic_results = analyze_semantic_matching(resume_text, role_document(career_roles, target_role))
            skill_score = role_skill_score["score"]
            matched_skills = role_skill_score["matched"]
            missing_skills = role_skill_score["missing"]
            semantic_label = "Resume vs Target Role Relevance"
            skill_label = "Target Role Skills Match"

        semantic_score = semantic_results["score"]

        # Overall ATS Score (40% Skill Match, 40% Semantic Similarity, 20% Document Structure)
        ats_score = round((skill_score * 0.40) + (semantic_score * 0.40) + (structure_score * 0.20), 1)

        # Display overall score in a beautiful container
        score_color = "#10b981" if ats_score >= 75 else ("#f59e0b" if ats_score >= 50 else "#ef4444")
        st.markdown(
            f"""
            <div class="ats-overall-card" style="
                background: linear-gradient(135deg, #0f172a 0%, #080d1a 100%);
                border: 1px solid rgba(34, 211, 238, 0.1);
                border-radius: 12px;
                padding: 1.75rem;
                margin-bottom: 1.5rem;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            ">
                <p style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.5rem 0;">Overall ATS Match Strength</p>
                <h2 style="font-size: 3.5rem; font-weight: 800; color: {score_color}; margin: 0 0 0.75rem 0; line-height: 1;">{ats_score}%</h2>
                <div style="background: rgba(255,255,255,0.05); border-radius: 999px; height: 10px; width: 60%; margin: 0 auto 0.75rem auto; overflow: hidden;">
                    <div style="background: {score_color}; width: {min(int(ats_score), 100)}%; height: 100%; border-radius: 999px; transition: width 0.5s ease-in-out;"></div>
                </div>
                <p style="color: #e2e8f0; font-size: 0.95rem; max-width: 600px; margin: 0 auto; line-height: 1.5;">
                    {'Excellent! Your resume is highly optimized for this role and has passed major ATS filters.' if ats_score >= 75 else 
                     ('Good match, but has gaps. Tuning your skills list and formatting will make it highly competitive.' if ats_score >= 50 else 
                      'Weak alignment. You need to incorporate key skills and restructure the resume layout.')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Columns for detailed score metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            metric_card("Skill Match", f"{skill_score}%", "Core technical keyword overlap")
        with m_col2:
            metric_card("Semantic Fit", f"{semantic_score}%", "Contextual phrasing match")
        with m_col3:
            metric_card("Document Structure", f"{structure_score}%", "Formatting & layout checks")

        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["🎯 Skill Match & Gaps", "📋 ATS Layout & Formatting", "🚀 Action Verb Booster", "📄 Document Previews"])

        with tabs[0]:
            st.markdown("### Technical Skill Analysis")
            st.markdown("These are the core keywords and tools detected in your profile compared to target requirements.")
            
            st.markdown("#### Detected Skills in Resume")
            badge_list(detected, "info", "No known skills detected yet")

            col_match, col_miss = st.columns(2)
            with col_match:
                st.markdown("#### Matched Skills")
                badge_list(matched_skills, "success", "No matched skills detected yet")
            with col_miss:
                st.markdown("#### Missing / Weak Skills")
                badge_list(missing_skills, "danger", "No missing skills detected")
                
            st.markdown("---")
            st.markdown("#### Vector Space & Embedding Diagnostics")
            st.markdown("The ATS engine maps your resume and job requirements into high-dimensional vector embeddings to compute semantic similarity:")
            
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.markdown(f"• **Vocabulary Dimensions:** `{semantic_results['vocab_count']}` token features")
                st.markdown(f"• **Embedding Feature Dimensions:** `2 x {semantic_results['feature_dims']}` dense matrix")
            with e_col2:
                st.markdown("• **Top Shared Semantic Terms:**")
                if semantic_results["top_terms"]:
                    badge_list(semantic_results["top_terms"], "success")
                else:
                    st.markdown("*No significant overlapping terms found.*")

            if missing_skills:
                st.markdown("#### Recommended Learning Focus")
                st.markdown("Prioritize studying these missing skills to close your preparation gap:")
                focus_skills = missing_skills[:6]
                related_courses = courses[courses["Skill"].isin(focus_skills)]
                st.dataframe(related_courses.head(8), use_container_width=True, hide_index=True)

        with tabs[1]:
            st.markdown("### ATS Formatting & Structural Audit")
            st.markdown("ATS parsers search for specific sections and structural signals. Ensure these formatting rules are followed:")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Contact Information & Links")
                email_status = "✅ Email Address Found" if structure_results["has_email"] else "❌ Email Address Missing"
                linkedin_status = "✅ LinkedIn Profile Link Found" if structure_results["has_linkedin"] else "⚠️ LinkedIn Link Missing (Recommended)"
                github_status = "✅ GitHub Profile Link Found" if structure_results["has_github"] else "⚠️ GitHub Link Missing (Recommended)"
                
                st.markdown(f"<div style='font-size: 0.95rem; line-height: 2;'>• {email_status}<br>• {linkedin_status}<br>• {github_status}</div>", unsafe_allow_html=True)
                
                st.markdown("#### Document Length")
                st.markdown(f"• **Word Count:** {structure_results['word_count']} words")
                word_color = "#10b981" if structure_results["word_count"] in range(300, 1200) else "#f59e0b"
                st.markdown(f"<div style='padding: 0.5rem 0.75rem; background: rgba(255,255,255,0.03); border-left: 3px solid {word_color}; font-size: 0.9rem;'>{structure_results['word_msg']}</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("#### Key Sections Found")
                sec_map = structure_results["found_sections"]
                for sec, found in sec_map.items():
                    sec_status = "✅ Found" if found else "❌ Missing"
                    st.markdown(f"• **{sec}:** {sec_status}")
                    
                st.markdown("#### Action Verbs Strengths")
                verb_color = "#10b981" if structure_results["verb_count"] >= 8 else ("#f59e0b" if structure_results["verb_count"] >= 4 else "#ef4444")
                st.markdown(f"<div style='padding: 0.5rem 0.75rem; background: rgba(255,255,255,0.03); border-left: 3px solid {verb_color}; font-size: 0.9rem;'>{structure_results['verb_msg']}</div>", unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("### Action Verb Optimization")
            st.markdown("ATS algorithms rank resumes higher when they use strong, action-oriented verbs rather than passive statements like 'responsible for'.")
            
            st.markdown("#### Action Verbs Detected in Your Resume")
            if structure_results["found_verbs"]:
                badge_list(structure_results["found_verbs"], "cyan")
            else:
                st.warning("No standard action verbs detected. Try using words like 'developed', 'managed', or 'optimized'.")
                
            st.markdown("---")
            st.markdown("#### ATS Recommended Verbs to Boost Your Score")
            st.markdown("Replace passive descriptions with these industry-approved action verbs:")
            
            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                st.markdown("**Creation & Build:**")
                st.markdown("`engineered`, `architected`, `spearheaded`, `formulated`, `conceptualized` (e.g., *'Architected a distributed payment system...'* )")
            with v_col2:
                st.markdown("**Optimization & Scale:**")
                st.markdown("`optimized`, `streamlined`, `automated`, `accelerated`, `revitalized` (e.g., *'Streamlined CI/CD deployment pipelines...'* )")
            with v_col3:
                st.markdown("**Leadership & Value:**")
                st.markdown("`spearheaded`, `championed`, `orchestrated`, `delivered`, `maximized` (e.g., *'Orchestrated cross-functional database migrations...'* )")

        with tabs[3]:
            prev_col1, prev_col2 = st.columns(2)
            with prev_col1:
                st.markdown("#### Parsed Resume Plain Text")
                st.text_area("Resume plain text content", resume_text, height=300, label_visibility="collapsed", key="preview_resume_plain")
            with prev_col2:
                st.markdown("#### Job Description / Target Role Doc")
                if jd_text:
                    st.text_area("JD plain text content", jd_text, height=300, label_visibility="collapsed", key="preview_jd_plain")
                else:
                    st.text_area("Target role metadata plain text", role_document(career_roles, target_role), height=300, label_visibility="collapsed", key="preview_role_plain")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Smart ATS Suggestions")
        if ats_score >= 75:
            roadmap_step("01", "Strong ATS Alignment", "Your resume is semantically close to the role. Improve by adding measurable project outcomes and exact job-title language.", "green")
            roadmap_step("02", "Prepare Proof", "Be ready to explain projects, tools, and responsibilities that connect to this role.", "cyan")
        elif ats_score >= 50:
            roadmap_step("01", "Improve Role Alignment", "Your resume has partial fit. Add stronger project bullets, relevant tools, and missing skills from the list.", "violet")
            roadmap_step("02", "Rewrite Key Sections", "Tune your summary, skills, and project descriptions to match the target role or JD responsibilities.", "cyan")
        else:
            roadmap_step("01", "Build Missing Foundations", "The current resume is not close enough semantically. Learn the missing skills and add proof projects.", "red")
            roadmap_step("02", "Use A Better-Fit Target", "Try a target role closer to your current skill profile while you build toward this one.", "violet")
    else:
        st.info("Upload a resume to generate semantic ATS analysis. Add a job description for the most accurate score.")


elif menu == "Career Recommendation":
    section_header("Career Recommendation", "semantic fit engine", "Select your skills and discover your strongest role matches using skill overlap and semantic similarity.")

    default_skills = st.session_state.get("resume_skills", [])
    default_text = st.session_state.get("resume_text", "")
    
    user_skills = st.multiselect("Your skills", sorted(career_roles["Skill"].unique()), default=default_skills)
    resume_context = st.text_area(
        "Optional profile summary or resume text",
        value=default_text[:2000] if default_text else "",
        height=140,
        placeholder="Paste resume summary, project details, or experience text for stronger semantic role matching.",
    )

    if user_skills or resume_context.strip():
        result = semantic_recommendation_table(career_roles, user_skills, resume_context)
        top = result.head(5)

        st.markdown("### Best Semantic Matches")
        st.dataframe(
            top,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Role": st.column_config.TextColumn("Career Role", width="medium"),
                "Overall Match %": st.column_config.ProgressColumn("Overall Match", min_value=0, max_value=100, format="%d%%"),
                "Skill Match %": st.column_config.NumberColumn("Skill Keyword Match", format="%d%%"),
                "Semantic Fit %": st.column_config.NumberColumn("Semantic Phrase Fit", format="%d%%"),
                "Missing Skills": st.column_config.TextColumn("Missing Skills", width="large"),
            }
        )

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
    default_skills = st.session_state.get("resume_skills", [])
    user_skills = st.multiselect("Your skills", sorted(career_roles["Skill"].unique()), default=default_skills)

    required = role_required_skills(career_roles, role)
    gap = score_role(user_skills, required)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Readiness", f"{gap['score']}%", "Target role fit")
    with col2:
        metric_card("Matched", str(len(gap["matched"])), "Skills already covered")
    with col3:
        metric_card("Missing", str(len(gap["missing"])), "Skills to build next")

    col_req, col_miss = st.columns(2)
    with col_req:
        st.markdown("### Matched Skills")
        badge_list(gap["matched"], "success", "No matched skills yet")
    with col_miss:
        st.markdown("### Missing Skills")
        badge_list(gap["missing"], "danger", "No missing skills for this role")


elif menu == "Course Finder":
    section_header("Course Finder", "learning path", "Find courses that support the skill you want to build next.")

    skill = st.selectbox("Select skill", sorted(courses["Skill"].unique()))
    level = st.selectbox("Level", ["All"] + sorted(courses["Level"].dropna().unique()))

    filtered = courses[courses["Skill"] == skill]
    if level != "All":
        filtered = filtered[filtered["Level"] == level]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Skill": st.column_config.TextColumn("Target Skill", width="medium"),
            "Course": st.column_config.TextColumn("Course Title", width="large"),
            "Platform": st.column_config.TextColumn("Learning Platform", width="medium"),
            "Level": st.column_config.TextColumn("Difficulty Level", width="small"),
        }
    )


elif menu == "Market Trends":
    section_header("Market Trends", "skill demand", "Explore which skills are strongest in the current career dataset.")

    sorted_trends = skills_trends.sort_values("DemandScore", ascending=False)
    st.dataframe(
        sorted_trends,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Skill": st.column_config.TextColumn("Skill Keyword", width="medium"),
            "DemandScore": st.column_config.ProgressColumn("Market Demand Score", min_value=0, max_value=100, format="%d"),
            "Category": st.column_config.TextColumn("Technology Category", width="medium"),
        }
    )

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
    default_skills = st.session_state.get("resume_skills", [])
    user_skills = st.multiselect("Your current skills", sorted(career_roles["Skill"].unique()), default=default_skills)

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
        st.dataframe(
            courses[courses["Skill"].isin(missing[:5])].head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Skill": st.column_config.TextColumn("Target Skill", width="medium"),
                "Course": st.column_config.TextColumn("Course Title", width="large"),
                "Platform": st.column_config.TextColumn("Learning Platform", width="medium"),
                "Level": st.column_config.TextColumn("Difficulty Level", width="small"),
            }
        )
