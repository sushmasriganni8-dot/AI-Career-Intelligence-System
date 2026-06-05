# AI Career Intelligence System

AI Career Intelligence System is a Streamlit-based career analytics platform that helps users analyze resumes, measure ATS fit, discover suitable career roles, identify skill gaps, and find relevant courses.

## Features

- Dark premium dashboard UI
- Resume analysis for PDF, DOCX, TXT, and text-like files
- Semantic ATS match score against a target role or job description
- Job description comparison inside Resume Analysis
- Matched and missing skill detection
- Career recommendation based on selected skills
- Skill gap analysis with readiness percentage
- Course finder by skill and level
- Market trend visualization using demand scores
- Clean CSV-driven data structure for easy expansion

## Tech Stack

- Python
- Streamlit
- Pandas
- Altair
- PDFPlumber
- Scikit-learn
- Python-docx

## Project Structure

```text
AI-Career-Intelligence-System/
├── app.py
├── assets/
│   └── style.css
├── data/
│   ├── career_info.csv
│   ├── career_roles.csv
│   ├── courses.csv
│   └── skills_trends.csv
├── modules/
│   ├── ats_checker.py
│   ├── career_recommender.py
│   ├── resume_parser.py
│   └── skill_gap.py
├── requirements.txt
└── README.md
```

## How To Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Files

- `career_info.csv`: career roles, salary range, demand, growth, and industry
- `career_roles.csv`: role-to-skill mapping
- `courses.csv`: course recommendations by skill and level
- `skills_trends.csv`: demand scores for trending skills

## Future Enhancements

- AI-generated resume feedback
- Downloadable PDF career report
- User login and saved analysis history
- Larger role and skills dataset
- Interview preparation recommendations
