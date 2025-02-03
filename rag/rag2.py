import os
import streamlit as st
from llama_parser import LlamaParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile


api_key = "llx-r49wysXuKBXrwqHfrun8jCxIbkG1yssnXRLR4IRriy3lnAna"  # Replace with your actual API key

# Initialize the SentenceTransformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings

# Initialize the Llama Parser
parser = LlamaParser(api_key=api_key)

def parse_resume(file_path):
    """
    Parse the resume using Llama Parser and return key details.
    Extracts:
    - Name
    - Highlighted Skills
    - Experience
    - Full Resume Text (for embeddings)
    """
    try:
        parsed_data = parser.parse(file_path)
        name = parsed_data.get("name", "Unknown")
        skills = ", ".join(parsed_data.get("skills", []))  # Convert list to string
        experience = parsed_data.get("experience", "Not mentioned")
        resume_text = " ".join(parsed_data.values())  # Convert all extracted data into a single string
        return name, skills, experience, resume_text.strip()
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None, None, None  # Skip resumes that fail to parse

# Function to convert text to embeddings
def get_embedding(text):
    """
    Convert text to a vector embedding.
    """
    return model.encode(text, convert_to_numpy=True)  # Convert text to numerical vector

def match_resumes_with_prompt(resume_files, recruiter_prompt):
    """
    Match resumes with the recruiter prompt based on cosine similarity.
    """
    # Convert recruiter prompt to an embedding
    prompt_embedding = get_embedding(recruiter_prompt).reshape(1, -1)  # Ensure it's 2D

    resume_embeddings = []
    resume_data = []

    for file_path in resume_files:
        name, skills, experience, resume_text = parse_resume(file_path)
        if resume_text:  # Ensure text was extracted
            embedding = get_embedding(resume_text).reshape(1, -1)  # Ensure it's 2D
            resume_embeddings.append(embedding)
            resume_data.append((name, skills, experience, file_path))  # Store candidate details

    # Convert embeddings list to a NumPy array
    resume_embeddings = np.array([embedding.flatten() for embedding in resume_embeddings])

    # Compute cosine similarity between recruiter query and each resume
    similarity_scores = cosine_similarity(prompt_embedding, resume_embeddings)

    # Rank resumes based on similarity scores (highest to lowest)
    ranked_resumes = sorted(zip(resume_data, similarity_scores.flatten()), key=lambda x: x[1], reverse=True)

    return ranked_resumes

# Streamlit UI for uploading resumes and entering a query
def streamlit_ui():
    st.title("Resume Query Matcher")

    # Upload resumes
    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
    
    if uploaded_files:
        # Save the uploaded resumes to temporary files
        resume_files = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.read())
                resume_files.append(tmp_file.name)

        # Get recruiter query from user input
        recruiter_prompt = st.text_input("Enter your recruiter query")

        if recruiter_prompt:
            # Match resumes with query
            ranked_resumes = match_resumes_with_prompt(resume_files, recruiter_prompt)

            # Display the results
            st.subheader("Query Results")

            for (name, skills, experience, file_path), score in ranked_resumes:
                st.write(f"*Name:* {name}")
                st.write(f"*Skills:* {skills}")
                st.write(f"*Experience:* {experience}")
                st.write(f"*Resume File:* {os.path.basename(file_path)}")
                st.write(f"*Similarity Score:* {score:.4f}")
                st.write("---")

# Run the app
if __name__ == "_main_":
    streamlit_ui()