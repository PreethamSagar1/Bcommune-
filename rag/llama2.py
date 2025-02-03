import streamlit as st
from langchain.llms import Ollama
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from llama_parse import LlamaParse
from supabase import create_client
import os

# Hardcoded Supabase
SUPABASE_URL = "https://qlgpplquymkdfxvchhgb.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFsZ3BwbHF1eW1rZGZ4dmNoaGdiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODI1NzUzNCwiZXhwIjoyMDUzODMzNTM0fQ._nfKI6glaOmTFqn3Cl2erkT89hAHE37ts_I01rhVTdE"  # Replace with your Supabase API key
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# LlamaParse for Resume Parsing
parser = LlamaParse(api_key="llx-r49wysXuKBXrwqHfrun8jCxIbkG1yssnXRLR4IRriy3lnAna")

# Qdrant for Vector Storage
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
qdrant = Qdrant(collection_name="resumes", embeddings=embeddings, location="qdrant-cloud-url")

# Ollama (Mistral-7B) for RAG-based Queries
llm = Ollama(model_name="mistral")

def parse_resume(file):
    """Parses resume PDF and extracts structured data."""
    parsed_data = parser.parse(file.read())
    return parsed_data

def store_resume(parsed_resume):
    """Stores parsed resume in Supabase and indexes in Qdrant."""
    response = supabase.table("resumes").insert(parsed_resume).execute()
    qdrant.add_texts(texts=[parsed_resume['content']])
    return response

def match_jobs(query):
    """Retrieves relevant resumes from Qdrant and generates insights using Mistral-7B."""
    docs = qdrant.similarity_search(query)
    response = llm.generate(f"Match the following job role to the most relevant resumes: {query}\n\nResumes: {docs}")
    return response

# Streamlit UI
st.title("AI-Powered Resume Matcher")

st.sidebar.header("Upload Resume")
uploaded_file = st.sidebar.file_uploader("Upload your resume (PDF)", type=["pdf"])
if uploaded_file:
    parsed_resume = parse_resume(uploaded_file)
    store_resume(parsed_resume)
    st.sidebar.success("Resume uploaded and processed successfully!")

st.header("Job Role Matching")
job_query = st.text_input("Enter Job Role Description:")
if st.button("Find Matching Resumes"):
    results = match_jobs(job_query)
    st.write(results)
