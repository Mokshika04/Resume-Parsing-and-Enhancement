import streamlit as st
import os
import json
import tempfile
import pprint
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & AI Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- 1. Resume Schema Definitions ---
class ExperienceItem(BaseModel):
    company: Optional[str]
    role: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]

class EducationItem(BaseModel):
    institution: Optional[str]
    degree: Optional[str]
    start_year: Optional[str]
    end_year: Optional[str]

class ResumeSchema(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    skills: List[str]
    education: List[EducationItem]
    experience: List[ExperienceItem]

# --- 2. Streamlit UI Config ---
st.set_page_config(page_title="ATS Resume Optimizer", layout="wide")
st.title(" AI Resume Parser & ATS Matcher")
st.markdown("Optimize your resume for a specific job description using Gemini 2.5 Flash.")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    google_api_key = st.text_input("Enter Google API Key", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    st.info("Upload your resume and paste the Job Description to begin.")

# --- 3. Input Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Step 1: Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF Resume", type="pdf")

with col2:
    st.subheader("Step 2: Paste Job Description")
    job_description = st.text_area("Job Description", height=250, placeholder="Paste the Job Description here...")

# --- 4. Main Processing Logic ---
if st.button(" Run Analysis", type="primary"):
    if not uploaded_file or not job_description:
        st.warning("Please upload a resume and provide a job description.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error("Please provide a Google API Key in the sidebar.")
    else:
        try:
            with st.spinner("Processing... This takes about 15-20 seconds."):
                # A. Handle File Upload (Save to Temp for PyPDFLoader)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                full_resume_text = "\n".join([doc.page_content for doc in docs])
                os.remove(tmp_path) # Clean up temp file

                # B. Initialize LLM
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

                # C. Parsing Resume
                parser = PydanticOutputParser(pydantic_object=ResumeSchema)
                parse_prompt = ChatPromptTemplate.from_template("""
                You are an expert ATS resume parser. Extract key information from the resume below.
                Return only JSON following this schema: {format_instructions}
                Resume text: ```{text}```
                """)
                messages = parse_prompt.format_messages(
                    text=full_resume_text,
                    format_instructions=parser.get_format_instructions()
                )
                response = llm.invoke(messages)
                parsed_resume = parser.parse(response.content)

                # D. Resume Enhancement
                enhance_prompt = ChatPromptTemplate.from_template("""
                You are an expert resume writer. Rewrite experience bullets to be action-based and quantified.
                Input JSON: ```json {resume_json} ```
                Return improved bullets in the SAME JSON format only.
                """)
                enhance_response = llm.invoke(enhance_prompt.format_messages(
                    resume_json=json.dumps(parsed_resume.model_dump())
                ))
                
                # Robust JSON Cleaning
                clean_enhance = enhance_response.content.strip().replace("```json", "").replace("```", "")
                enhanced_resume = json.loads(clean_enhance)

                # E. ATS Comparison
                ats_prompt = ChatPromptTemplate.from_template("""
                You are an ATS analyzer. Compare the resume with the Job Description.
                Return JSON with keys: ats_score, missing_keywords, recommended_keywords_to_add, suggested_improvements, tailored_summary.
                Resume: ```json {resume_json} ```
                Job Description: ```{jd}```
                Return JSON only.
                """)
                ats_response = llm.invoke(ats_prompt.format_messages(
                    resume_json=json.dumps(enhanced_resume),
                    jd=job_description
                ))
                clean_ats = ats_response.content.strip().replace("```json", "").replace("```", "")
                ats_output = json.loads(clean_ats)

            # --- 5. Displaying Results ---
            st.success("Analysis Complete!")
            
            # Top Metrics
            m_col1, m_col2 = st.columns(2)
            score = ats_output.get('ats_score', 0)
            m_col1.metric("ATS Match Score", f"{score}%")
            m_col2.progress(float(score) / 100)

            # Detailed Feedback Tabs
            tab1, tab2, tab3 = st.tabs(["💡 Improvements", "📊 Keyword Analysis", "✨ Enhanced Resume"])
            
            with tab1:
                st.subheader("Tailored Summary")
                st.write(ats_output.get("tailored_summary"))
                st.subheader("Actionable Improvements")
                for imp in ats_output.get("suggested_improvements", []):
                    st.write(f"- {imp}")

            with tab2:
                c1, c2 = st.columns(2)
                c1.write("**Missing Keywords**")
                c1.write(ats_output.get("missing_keywords"))
                c2.write("**Keywords to Add**")
                c2.write(ats_output.get("recommended_keywords_to_add"))

            with tab3:
                st.json(enhanced_resume)

        except Exception as e:
            st.error(f"Error: {str(e)}")