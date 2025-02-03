from django.urls import path
from .views import upload_resume, create_job_view,recommend_jobs_view,findnewtalent,search_resumes_view

urlpatterns = [
    path("upload_resume/", upload_resume, name="upload_resume"),
    path("create_job/", create_job_view, name="create_job"),
    path("recommend_jobs/", recommend_jobs_view, name="recommend_jobs"),
    path("findnewtalent/",findnewtalent,name="findnewtalent"),
    path('search_resumes_view',search_resumes_view,name="search_resumes_view"),
]
