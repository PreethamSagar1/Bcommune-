from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Resume, Job
import asyncio
from .utils import (
    extract_resume_text, get_embedding, insert_resume, search_resumes,
    create_job, recommend_jobs
)

async def upload_resume(request):
    if request.method == 'POST':
        candidate_name = request.POST.get('candidate_name')
        email = request.POST.get('email')
        uploaded_file = request.FILES.get('uploaded_file')

        if candidate_name and email and uploaded_file:
            resume_text = extract_resume_text(uploaded_file)
            embedding = await get_embedding(resume_text)
            await insert_resume(candidate_name, email, resume_text, embedding)
            return redirect('upload_resume')
    return render(request, 'upload_resume.html')

async def search_resumes_view(request):
    if request.method == 'POST':
        search_query = request.POST.get('search_query')
        match_count = int(request.POST.get('match_count', 5))

        if search_query:
            search_results = await search_resumes(search_query, match_count)
            return render(request, 'search_resumes.html', {'search_results': search_results})
    return render(request, 'search_resumes.html')

async def create_job_view(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description')
        requirements = request.POST.get('requirements')

        if title and description and requirements:
            await create_job(title, description, requirements)
            return redirect('create_job')
    return render(request, 'create_job.html')

async def recommend_jobs_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')

        if email:
            recommended_jobs = await recommend_jobs(email)
            return render(request, 'recommend_jobs.html', {'recommended_jobs': recommended_jobs})
    return render(request, 'recommend_jobs.html')

def findnewtalent(request):
    return render(request, 'findnewtalent.html')