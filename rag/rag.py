import re
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
from openai import AsyncOpenAI
from supabase import create_client, Client
import streamlit as st
import PyPDF2  # For PDF files
import docx  # For Word documents
from tenacity import retry, wait_fixed, stop_after_attempt
from rag_agent import create_job, recommend_jobs
# Hardcoded Supabase and OpenAI credentials
SUPABASE_URL = "https://qlgpplquymkdfxvchhgb.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFsZ3BwbHF1eW1rZGZ4dmNoaGdiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODI1NzUzNCwiZXhwIjoyMDUzODMzNTM0fQ._nfKI6glaOmTFqn3Cl2erkT89hAHE37ts_I01rhVTdE"  # Replace with your Supabase API key
OPENAI_API_KEY = "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"  # Replace with your OpenAI API key

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI API client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Memory to store past interactions and feedback
memory = []

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI using a cost-efficient model."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text  # Cost-efficient embedding model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return []

def extract_email(text: str) -> str:
    """Extract email address from resume text using regex."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    return match.group(0) if match else "unknown@example.com"

def extract_resume_text(uploaded_file):
    """Extract text from uploaded resume file."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join(para.text for para in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type. Please upload a plain text, PDF, or Word document.")

async def insert_resume(candidate_name: str, email: str, resume_text: str, embedding: List[float]):
    """Insert or update resume in the Supabase database."""
    if not embedding:
        raise ValueError("Embedding cannot be empty.")

    resume_data = {
        "candidate_name": candidate_name,
        "email": email,
        "resume_text": resume_text,
        "embedding": embedding,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    response = supabase.table("resumes").upsert(resume_data, on_conflict="email").execute()
    if response.data:
        st.success(f"Resume uploaded and profile created for {candidate_name}!")
    else:
        st.error(f"Failed to upload resume for {candidate_name}.")

async def process_and_store_resume(candidate_name: str, email: str, uploaded_file):
    """Process a resume and store it in the Supabase database."""
    try:
        resume_text = extract_resume_text(uploaded_file)
        embedding = await get_embedding(resume_text)
        await insert_resume(candidate_name, email, resume_text, embedding)
    except Exception as e:
        st.error(f"Error processing resume: {e}")

async def retrieve_resumes(query_text: str, match_count: int = 10) -> List[dict]:
    """Retrieve resumes using vector similarity and additional filters."""
    query_embedding = await get_embedding(query_text)

    if not query_embedding:
        st.error("Failed to generate embedding for search query.")
        return []

    # Fetch resumes using vector similarity
    response = supabase.rpc("match_resumes", {
        "query_embedding": query_embedding,
        "match_count": match_count * 2  # Fetch more candidates for re-ranking
    }).execute()

    return response.data if response.data else []

async def reason_over_resumes(query_text: str, resumes: List[dict]) -> str:
    """Use GPT-3.5 to reason over retrieved resumes and generate insights (cost-efficient)."""
    system_prompt = """
    You are an AI assistant that analyzes resumes and provides insights.
    Given a list of resumes and a search query, summarize why these candidates are a good fit.
    Be concise and focus on key skills, experiences, and qualifications.
    """
    resume_texts = "\n\n".join([resume.get("resume_text", "")[:1000] for resume in resumes])  # Limit text length

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Cost-efficient model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query_text}\nResumes:\n{resume_texts}"}
            ],
            max_tokens=300  # Limit tokens to reduce cost
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error reasoning over resumes: {e}")
        return ""

async def plan_next_steps(query_text: str, resumes: List[dict]) -> str:
    """Use GPT-3.5 to decide the next steps based on the retrieved resumes (cost-efficient)."""
    system_prompt = """
    You are an AI assistant that helps with resume searches.
    Given a search query and a list of retrieved resumes, decide the next steps:
    - If the resumes are a good match, summarize why.
    - If the resumes are not a good match, suggest how to refine the search or ask the user for clarification.
    """
    resume_texts = "\n\n".join([resume.get("resume_text", "")[:1000] for resume in resumes])  # Limit text length

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Cost-efficient model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query_text}\nResumes:\n{resume_texts}"}
            ],
            max_tokens=300  # Limit tokens to reduce cost
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error planning next steps: {e}")
        return ""

async def store_feedback(query_text: str, feedback: str):
    """Store user feedback in memory for future improvement."""
    memory.append({
        "query": query_text,
        "feedback": feedback,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

def streamlit_ui():
    st.title("AI-Powered Job Recommendation System with Feedback Loop")

    page = st.sidebar.radio("Select an option", ("Upload Resume", "Search Resumes", "Create Job", "Recommend Jobs"))

    if page == "Upload Resume":
        st.header("Upload Resume & Create Profile")

        with st.form(key="profile_form"):
            candidate_name = st.text_input("Full Name")
            email = st.text_input("Email")
            uploaded_file = st.file_uploader("Upload Resume", type=["txt", "pdf", "docx"])
            submit_button = st.form_submit_button("Upload Resume")

            if submit_button and uploaded_file is not None:
                asyncio.run(process_and_store_resume(candidate_name, email, uploaded_file))

    elif page == "Search Resumes":
        st.header("Search Resumes with Feedback Loop")

        search_query = st.text_input("Enter search query (e.g., Python developer with 5 years of experience)")
        match_count = st.slider("Number of recommendations", 1, 20, 5)

        if st.button("Search Resumes"):
            if search_query:
                st.write("Searching for matching profiles...")
                resumes = asyncio.run(retrieve_resumes(search_query, match_count))

                if resumes:
                    # Reason over the retrieved resumes
                    insights = asyncio.run(reason_over_resumes(search_query, resumes))
                    st.write("### Insights from GPT-3.5")
                    st.write(insights)

                    # Plan next steps
                    next_steps = asyncio.run(plan_next_steps(search_query, resumes))
                    st.write("### Next Steps")
                    st.write(next_steps)

                    # Collect user feedback
                    feedback = st.text_input("Was this search helpful? Please provide feedback:")
                    if feedback:
                        asyncio.run(store_feedback(search_query, feedback))
                        st.success("Thank you for your feedback!")
                else:
                    st.warning("No matching profiles found.")

    elif page == "Create Job":
        st.header("Create a New Job Posting")

        with st.form(key="job_form"):
            title = st.text_input("Job Title")
            description = st.text_area("Job Description")
            requirements = st.text_area("Job Requirements")
            submit_button = st.form_submit_button("Create Job")

            if submit_button:
                asyncio.run(create_job(title, description, requirements))

    elif page == "Recommend Jobs":
        st.header("Get Job Recommendations with Feedback Loop")

        email = st.text_input("Enter your email to get job recommendations")
        if st.button("Recommend Jobs"):
            if email:
                st.write("Fetching job recommendations...")
                recommended_jobs = asyncio.run(recommend_jobs(email))

                if recommended_jobs:
                    for job in recommended_jobs:
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 2px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <h3 style="color: #2E7D32;">{job.get('title', 'Unknown')}</h3>
                                <p><strong>Description:</strong> {job.get('description', 'N/A')}</p>
                                <p><strong>Requirements:</strong> {job.get('requirements', 'N/A')}</p>
                                <p><strong>Relevance Score:</strong> {job.get('relevance_score', 0):.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Collect user feedback
                    feedback = st.text_input("Were these recommendations helpful? Please provide feedback:")
                    if feedback:
                        asyncio.run(store_feedback(email, feedback))
                        st.success("Thank you for your feedback!")
                else:
                    st.warning("No job recommendations found.")

if __name__ == "__main__":
    streamlit_ui()