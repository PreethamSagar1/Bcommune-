from django.db import models
import json

class Resume(models.Model):
    candidate_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    resume_text = models.TextField()
    embedding = models.JSONField()  # Store embedding as JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.candidate_name

class Job(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    requirements = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Feedback(models.Model):
    query = models.TextField()
    feedback = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback for {self.query}"