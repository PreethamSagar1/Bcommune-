import re
import asyncio
from datetime import datetime, timezone
from typing import List
from openai import AsyncOpenAI
from supabase import create_client, Client
import PyPDF2  # For PDF files
import docx  # For Word documents
from sentence_transformers import SentenceTransformer  # For embeddings
from tenacity import retry, wait_fixed, stop_after_attempt

# Hardcoded Supabase and OpenAI credentials
SUPABASE_URL = "https://anqosdckbtplhzynospu.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFucW9zZGNrYnRwbGh6eW5vc3B1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODIzMjY0OCwiZXhwIjoyMDUzODA4NjQ4fQ.tFFzJuh8MWNjkC4O_InDPFLJHiQgkKCJagh5Qgjx4Is" # Replace with your Supabase API key
OPENAI_API_KEY = "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"  # Replace with your OpenAI API key

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI API client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using Sentence Transformers."""
    try:
        # Generate the embedding
        embedding = embedding_model.encode(text)
        # Convert the NumPy array to a list of floats
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


# @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
# async def get_embedding(text: str) -> List[float]:
#     """Get embedding vector from OpenAI."""
#     try:
#         response = await openai_client.embeddings.create(
#             model="text-embedding-3-small", input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return []

def extract_email(text: str) -> str:
    """Extract email address from resume text using regex."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    return match.group(0) if match else "unknown@example.com"

def extract_resume_text(uploaded_file):
    """Extract text from uploaded resume file."""
    if uploaded_file.content_type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.content_type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        return text
    elif uploaded_file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
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
        print(f"Resume uploaded and profile created for {candidate_name}!")
    else:
        print(f"Failed to upload resume for {candidate_name}.")

async def search_resumes(query_text: str, match_count: int = 10) -> List[dict]:
    """
    Search for resumes using vector similarity in Supabase.
    Re-rank top candidates using GPT-4.
    """
    query_embedding = await get_embedding(query_text)

    if not query_embedding:
        print("Failed to generate embedding for search query.")
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
        
        # Extract numeric score from the response
        try:
            score = float(score_text)
            if 0 <= score <= 1:
                return score
            else:
                print(f"Invalid score range: {score}. Expected a value between 0 and 1.")
                return 0.0
        except ValueError:
            print(f"Failed to parse score: {score_text}. Expected a numeric value.")
            return 0.0
    except Exception as e:
        print(f"Error evaluating relevance: {e}")
        return 0.0

async def create_job(title: str, description: str, requirements: str):
    """Create a new job posting in the Supabase database."""
    job_data = {
        "title": title,
        "description": description,
        "requirements": requirements,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    response = supabase.table("jobs").insert(job_data).execute()
    if response.data:
        print(f"Job '{title}' created successfully!")
    else:
        print(f"Failed to create job '{title}'.")

async def recommend_jobs(email: str, match_count: int = 5) -> List[dict]:
    """
    Recommend jobs for a candidate based on their resume using Agantic RAG.
    """
    # Fetch the candidate's resume
    response = supabase.table("resumes").select("*").eq("email", email).execute()
    if not response.data:
        print(f"No resume found for email: {email}")
        return []

    candidate = response.data[0]
    resume_text = candidate.get("resume_text", "")
    if not resume_text:
        print("Resume text is empty.")
        return []

    # Step 1: Fetch all jobs from the database
    jobs_response = supabase.table("jobs").select("*").execute()
    jobs = jobs_response.data if jobs_response.data else []

    if not jobs:
        print("No jobs available for recommendation.")
        return []

    # Step 2: Use Agantic RAG to rank jobs based on the candidate's resume
    recommended_jobs = await rerank_jobs(resume_text, jobs)
    return recommended_jobs[:match_count]

async def rerank_jobs(resume_text: str, jobs: List[dict]) -> List[dict]:
    """
    Re-rank jobs based on the candidate's resume using GPT-4.
    """
    reranked = []
    for job in jobs:
        job_description = job.get("description", "")
        job_requirements = job.get("requirements", "")

        # Use GPT-4 to evaluate relevance of the job to the resume
        relevance_score = await evaluate_relevance(resume_text, f"{job_description}\n{job_requirements}")
        reranked.append({**job, "relevance_score": relevance_score})

    # Sort jobs by relevance score
    reranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return reranked