from __future__ import annotations as _annotations
import streamlit as st
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import json
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import create_client, Client
from typing import List
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Load environment variables
llm = os.getenv('LLM_MODEL', 'gpt-4-turbo')  # Updated to a valid model name
model = OpenAIModel(llm)
logfire.configure(send_to_logfire='if-token-present')

# Supabase and OpenAI clients
supabase_client: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define necessary dependencies
@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# System prompt for the agent
system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

# Function to extract embeddings using OpenAI API
async def get_embedding(text: str) -> List[float]:
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return a zero vector if embedding fails

# Function to extract metadata from text
async def extract_metadata(text: str) -> dict:
    system_prompt = """Extract structured metadata (skills, experience, keywords) from the text."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo",  # Updated to a valid model name
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text[:1000]}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}

# Function to insert/update resumes into Supabase with user details
async def insert_resume(user_name: str, email: str, resume_text: str, embedding: List[float], metadata: dict):
    resume_data = {
        "candidate_name": user_name,
        "email": email,
        "resume_text": resume_text,
        "metadata": metadata,
        "embedding": embedding,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Insert or update resume in Supabase
    response = supabase_client.table("resumes").upsert(resume_data, on_conflict="email").execute()
    if response.data:
        print(f"Inserted/Updated resume for {user_name} with email {email}")
    else:
        print(f"Failed to insert/update resume for {user_name} with email {email}")

# Function to process and store uploaded resume
async def process_and_store_resume(user_name: str, email: str, resume_text: str):
    embedding = await get_embedding(resume_text)
    metadata = await extract_metadata(resume_text)
    await insert_resume(user_name, email, resume_text, embedding, metadata)

# Function to retrieve relevant resumes from Supabase based on user query
@pydantic_ai_expert.tool
async def retrieve_relevant_resumes(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    try:
        query_embedding = await get_embedding(user_query)
        
        # Call the `match_resumes` function in Supabase
        result = ctx.deps.supabase.rpc(
            'match_resumes',
            {
                'query_embedding': query_embedding,
                'match_count': 5,  # Adjust based on how many matches you want to retrieve
                'filter': {}  # Add any metadata filter here if needed
            }
        ).execute()
        
        if not result.data:
            return "No relevant profiles found."
            
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['candidate_name']} ({doc['email']})

{doc['resume_text']}
"""
            formatted_chunks.append(chunk_text)
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving resumes: {e}")
        return f"Error retrieving resumes: {str(e)}"

# Main Streamlit interface
def main():
    st.title("Resume Upload and Search")

    # Create Resume upload section
    with st.form("upload_form"):
        st.header("Upload Resume")
        user_name = st.text_input("Full Name")
        email = st.text_input("Email")
        resume_file = st.file_uploader("Upload Resume", type=["txt", "pdf", "docx"])
        
        if st.form_submit_button("Upload Resume"):
            if user_name and email and resume_file:
                resume_text = resume_file.read().decode("utf-8")  # Assuming text-based resume files
                asyncio.run(process_and_store_resume(user_name, email, resume_text))
                st.success(f"Resume for {user_name} uploaded successfully!")
            else:
                st.error("Please provide all the required fields.")

    # Create Resume search section
    with st.form("search_form"):
        st.header("Search for Profiles")
        search_query = st.text_input("Search Query (e.g., skills or experience)")

        if st.form_submit_button("Search Resumes"):
            if search_query:
                response = asyncio.run(retrieve_relevant_resumes(pydantic_ai_expert, search_query))
                st.text_area("Search Results", value=response, height=400)
            else:
                st.error("Please enter a query to search for profiles.")

if __name__ == "__main__":
    main()