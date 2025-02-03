import os
import json
import asyncio
from datetime import datetime, timezone
from urllib.parse import urlparse
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from tenacity import retry, wait_fixed, stop_after_attempt
from supabase import create_client, Client

# Hardcoded Supabase and OpenAI credentials
SUPABASE_URL = "https://qlgpplquymkdfxvchhgb.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFsZ3BwbHF1eW1rZGZ4dmNoaGdiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODI1NzUzNCwiZXhwIjoyMDUzODMzNTM0fQ._nfKI6glaOmTFqn3Cl2erkT89hAHE37ts_I01rhVTdE"  # Replace with your Supabase API key
OPENAI_API_KEY = "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"  # Replace with your OpenAI API key

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI API key
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))  # Retry up to 3 times with a 2-second delay
async def get_embedding(text: str):
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))  # Retry up to 3 times with a 2-second delay
async def extract_metadata(text: str):
    """Use OpenAI to extract structured metadata."""
    system_prompt = """Extract structured metadata (skills, experience, keywords) from the text and return it as a JSON object."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo",  # Updated to a valid model name
            messages=[ 
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:1000]}  # Limit input size
            ],
            response_format={ "type": "json_object" }
        )
        metadata = json.loads(response.choices[0].message.content)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata is not a valid JSON object.")
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        raise

async def insert_resume(url, text, embedding, metadata):
    """Insert or update resume in the Supabase database."""
    if embedding is None or metadata is None:
        raise ValueError("Embedding or metadata cannot be None.")
    
    email = f"{urlparse(url).hostname}@example.com"  # Placeholder email, replace as necessary
    resume_data = {
        "candidate_name": "Unknown",
        "email": email,
        "resume_text": text,
        "metadata": metadata,
        "embedding": embedding,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Insert or update resume in Supabase
    response = supabase.table("resumes").upsert(resume_data, on_conflict="email").execute()
    if response.data:
        print(f"Inserted/Updated resume for {url}")
    else:
        print(f"Failed to insert/update resume for {url}")

async def process_and_store_document(url, markdown):
    """Process a document and store it in the Supabase database."""
    embedding = await get_embedding(markdown)
    metadata = await extract_metadata(markdown)
    await insert_resume(url, markdown, embedding, metadata)

async def search_resumes_by_embedding(query_text, match_count=10, filter_json={}):
    """Search for resumes using vector similarity in Supabase."""
    query_embedding = await get_embedding(query_text)
    
    # Call the `match_resumes` function in Supabase
    response = supabase.rpc("match_resumes", {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "filter": filter_json  # Use 'filter' instead of 'filter_json'
    }).execute()
    
    if response.data:
        return response.data
    else:
        print("No results found.")
        return []

async def crawl_parallel(urls, max_concurrent=5):
    """Crawl multiple URLs in parallel."""
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    async def process_url(url):
        async with semaphore:
            result = await crawler.arun(url=url, config=crawl_config)
            if result.success:
                await process_and_store_document(url, result.markdown_v2.raw_markdown)
            else:
                print(f"Failed: {url} - {result.error_message}")
    
    await asyncio.gather(*[process_url(url) for url in urls])
    await crawler.close()

async def main():
    """Main entry point to crawl and store resumes."""
    urls = ["https://ai.pydantic.dev/docs/example"]  # Replace with actual URLs
    await crawl_parallel(urls)
    
    # Example search query
    search_query = "Pydantic AI framework"
    filter_criteria = {}  # Add any metadata filter here if needed
    search_results = await search_resumes_by_embedding(search_query, match_count=5, filter_json=filter_criteria)
    print("Search Results:")
    for resume in search_results:
        print(resume)

if __name__ == "__main__":
    asyncio.run(main())