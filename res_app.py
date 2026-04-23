import streamlit as st
import pandas as pd
from resume_ats_core import (
    extract_text_from_pdf,
    preprocess,
    get_similarity,
    get_keywords,
    interpret
)

# UI CONFIG
st.set_page_config(page_title="ATS Resume Screener", layout="wide")

st.title("📄 AI Resume Screening System (ATS)")
st.markdown("Upload multiple resumes and match them with a job description.")

# INPUTS
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

job_desc = st.text_area("Paste Job Description")

# ANALYZE BUTTON
if st.button("Analyze Candidates"):

    if uploaded_files and job_desc:

        results = []

        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            resume_text = preprocess(resume_text)
            jd_text = preprocess(job_desc)

            score = get_similarity(resume_text, jd_text)
            keywords = get_keywords(resume_text, jd_text)
            match_level = interpret(score)

            results.append({
                "Candidate": file.name,
                "Score (%)": round(score, 2),
                "Match Level": match_level,
                "Keywords": ", ".join(keywords)
            })

        # Sort by score
        results = sorted(results, key=lambda x: x["Score (%)"], reverse=True)

        # BEST CANDIDATE
        best = results[0]
        st.success(f"🏆 Top Candidate: {best['Candidate']} ({best['Score (%)']}%)")

        st.subheader("📊 Ranked Candidates")

        # DISPLAY RESULTS
        for r in results:
            st.write(f"### {r['Candidate']}")
            st.write(f"Score: {r['Score (%)']}% | {r['Match Level']}")

            # Progress bar
            st.progress(int(r["Score (%)"]))

            st.write(f"🔑 Keywords: {r['Keywords']}")
            st.markdown("---")

        # DOWNLOAD CSV
        df = pd.DataFrame(results)

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Results CSV",
            data=csv,
            file_name="ats_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("Please upload resumes and enter job description")
