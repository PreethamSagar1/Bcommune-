import os
import json
import re
import asyncio
from datetime import datetime, timezone
from typing import List
from openai import AsyncOpenAI
from supabase import create_client, Client
import streamlit as st
import PyPDF2  # For PDF files
import docx  # For Word documents
from tenacity import retry, wait_fixed, stop_after_attempt

# Hardcoded Supabase and OpenAI credentials
SUPABASE_URL = "https://qlgpplquymkdfxvchhgb.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFsZ3BwbHF1eW1rZGZ4dmNoaGdiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODI1NzUzNCwiZXhwIjoyMDUzODMzNTM0fQ._nfKI6glaOmTFqn3Cl2erkT89hAHE37ts_I01rhVTdE"  # Replace with your Supabase API key
OPENAI_API_KEY = "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"  # Replace with your OpenAI API key

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI API client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
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

async def search_resumes(query_text: str, match_count: int = 10) -> List[dict]:
    """
    Search for resumes using vector similarity in Supabase.
    Re-rank top candidates using GPT-4.
    """
    query_embedding = await get_embedding(query_text)

    if not query_embedding:
        st.error("Failed to generate embedding for search query.")
        return []

    # Step 1: Initial search using vector similarity
    response = supabase.rpc("match_resumes", {
        "query_embedding": query_embedding,
        "match_count": match_count * 2  # Fetch more candidates for re-ranking
    }).execute()

    candidates = response.data if response.data else []

    if not candidates:
        return []

    # Step 2: Re-rank top candidates using GPT-4
    reranked_candidates = await rerank_candidates(query_text, candidates[:20])  # Re-rank top 20 candidates
    return reranked_candidates[:match_count]  # Return top N candidates

async def rerank_candidates(query_text: str, candidates: List[dict]) -> List[dict]:
    """
    Re-rank candidates by analyzing their full resume text using GPT-4.
    """
    reranked = []
    for candidate in candidates:
        resume_text = candidate.get("resume_text", "")
        if not resume_text:
            continue

        # Use GPT-4 to evaluate relevance of the full resume text to the query
        relevance_score = await evaluate_relevance(query_text, resume_text)
        reranked.append({**candidate, "relevance_score": relevance_score})

    # Sort candidates by relevance score
    reranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return reranked

async def evaluate_relevance(query_text: str, resume_text: str) -> float:
    """
    Evaluate the relevance of a resume to the query using GPT-4.
    Returns a relevance score between 0 and 1, where 1 is highly relevant.
    """
    system_prompt = """
    You are an AI assistant that evaluates the relevance of a resume to a given query.
    Return ONLY a numeric score between 0 and 1, where:
    - 1 means the resume is highly relevant to the query.
    - 0 means the resume is not relevant at all.
    Do not include any additional text or explanation in your response.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query_text}\nResume Text: {resume_text[:4000]}"}
            ],
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        
        # Debugging: Print the raw response from GPT-4
        st.write(f"GPT-4 Response: {score_text}")
        
        # Extract numeric score from the response
        try:
            score = float(score_text)
            if 0 <= score <= 1:
                return score
            else:
                st.error(f"Invalid score range: {score}. Expected a value between 0 and 1.")
                return 0.0
        except ValueError:
            st.error(f"Failed to parse score: {score_text}. Expected a numeric value.")
            return 0.0
    except Exception as e:
        st.error(f"Error evaluating relevance: {e}")
        return 0.0

def streamlit_ui():
    st.title("AI-Powered Resume Search System")

    page = st.sidebar.radio("Select an option", ("Upload Resume", "Search Resumes"))

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
        st.header("Search Resumes")

        search_query = st.text_input("Enter search query (e.g., Python developer with 5 years of experience)")

        # Add a slider for selecting the number of recommendations
        max_profiles = 20  # Maximum number of profiles to recommend
        match_count = st.slider("Number of recommendations", 1, max_profiles, 5)

        if st.button("Search Resumes"):
            if search_query:
                st.write("Searching for matching profiles...")
                search_results = asyncio.run(search_resumes(search_query, match_count=match_count))

                if search_results:
                    for resume in search_results:
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 2px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <h3 style="color: #2E7D32;">{resume.get('candidate_name', 'Unknown')}</h3>
                                <p><strong>Email:</strong> {resume.get('email', 'N/A')}</p>
                                <p><strong>Relevance Score:</strong> {resume.get('relevance_score', 0):.2f}</p>
                                <p><strong>Resume Excerpt:</strong> {resume.get('resume_text', '')[:500]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No matching profiles found.")

if __name__ == "__main__":
    streamlit_ui()