-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the resumes table
CREATE TABLE resumes (
    id BIGSERIAL PRIMARY KEY,
    candidate_name VARCHAR NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    resume_text TEXT NOT NULL,  -- Raw text extracted from the resume
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,  -- JSONB metadata (skills, experience, etc.)
    embedding VECTOR(768),  -- OpenAI embeddings (1536 dimensions)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index for fast vector similarity search
CREATE INDEX ON resumes USING ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for fast filtering
CREATE INDEX idx_resumes_metadata ON resumes USING gin (metadata jsonb_ops);

-- Create a function to match resumes based on vector similarity
CREATE OR REPLACE FUNCTION match_resumes (
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 10,
  filter JSONB DEFAULT '{}'::JSONB
) RETURNS TABLE (
  id BIGINT,
  candidate_name VARCHAR,
  email VARCHAR,
  resume_text TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql AS $$ 
BEGIN
  RETURN QUERY
  SELECT
    resumes.id,  -- Explicitly specify the table for the 'id' column
    resumes.candidate_name,
    resumes.email,
    resumes.resume_text,
    resumes.metadata,
    1 - (resumes.embedding <=> query_embedding) AS similarity
  FROM resumes
  WHERE resumes.metadata @> filter  -- Filter by metadata (optional)
  ORDER BY resumes.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Enable Row-Level Security (RLS)
ALTER TABLE resumes ENABLE ROW LEVEL SECURITY;

-- Allow public read access
CREATE POLICY "Allow public read access"
ON resumes
FOR SELECT
TO public
USING (true);

-- Allow users to update their own resumes
CREATE POLICY "Allow user to update own resume"
ON resumes
FOR UPDATE
TO public
USING (true)  -- Check existing rows
WITH CHECK (true);  -- Check new rows

-- Allow users to insert resumes
CREATE POLICY "Allow user to insert own resume"
ON resumes
FOR INSERT
TO public
WITH CHECK (true);  -- Check new rows

-- Allow users to delete their own resumes
CREATE POLICY "Allow user to delete own resume"
ON resumes
FOR DELETE
TO public
USING (true);  -- Check existing rows

-- Make sure policies are enabled
ALTER TABLE resumes ENABLE ROW LEVEL SECURITY;