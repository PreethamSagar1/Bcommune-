from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone
from django.conf import settings 
from django.contrib.auth import get_user_model
from pydantic import ValidationError

class CustomUser(AbstractUser):
    USER_TYPE_CHOICES = (
        ('individual', 'Individual'),
        ('company', 'Company'),
    )
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='company') 
    company_name = models.CharField(max_length=255, null=True, blank=True)
    company_website = models.URLField(max_length=200, null=True, blank=True)
    industry = models.CharField(max_length=100, null=True, blank=True)
    company_size = models.CharField(max_length=20, null=True, blank=True)
    company_type = models.CharField(max_length=255, null=True, blank=True)
    designation = models.CharField(max_length=255, null=True, blank=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    bcommune_profile = models.URLField(null=True, blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    about_company = models.TextField(null=True, blank=True)  # About the Company
    founding_year = models.PositiveIntegerField(null=True, blank=True)  # Establishment Year
    headquarters = models.CharField(max_length=255, null=True, blank=True)  # Headquarters Location
    branches = models.TextField(null=True, blank=True) 
    business_registration_number = models.CharField(max_length=100, null=True, blank=True)
    legal_structure = models.CharField(
        max_length=50,
        choices=[
            ('LLP', 'LLP'),
            ('Pvt Ltd', 'Private Limited'),
            ('Partnership', 'Partnership'),
            ('Sole Proprietorship', 'Sole Proprietorship'),
            ('Others', 'Others'),
        ],
        null=True,
        blank=True
    )
    core_services = models.TextField(null=True, blank=True)
    social_media_links = models.JSONField(default=dict, blank=True)  # Example: {"facebook": "", "linkedin": "", "twitter": ""}
    portfolio = models.TextField(null=True, blank=True)
    

    
class Idea(models.Model):
    title = models.CharField(max_length=200)
    patent_number = models.CharField(max_length=100, blank=True, null=True)
    brief_description = models.TextField()
    application_number = models.CharField(max_length=100, blank=True, null=True)
    problem_statement = models.TextField()
    solution = models.TextField()
    visibility = models.CharField(max_length=10, choices=[('public', 'Public'), ('private', 'Private')])
    details = models.TextField(blank=True, null=True)
    fund = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    category = models.CharField(max_length=100)
    photo = models.ImageField(upload_to='idea_photos/', blank=True, null=True)
    video = models.FileField(upload_to='idea_videos/', blank=True, null=True)
    team_info = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    email = models.EmailField(null=True, blank=True)
    def __str__(self):
        return self.title
    
    @property
    def likes(self):
        return self.likedislike_set.filter(like=True).count()

    @property
    def dislikes(self):
        return self.likedislike_set.filter(dislike=True).count()
    
    def get_like_status(self, user):
        """Get the like status for a specific user"""
        if not user.is_authenticated:
            return None
        try:
            like_obj = self.likedislike_set.get(user=user)
            if like_obj.like:
                return 'liked'
            elif like_obj.dislike:
                return 'disliked'
        except LikeDislike.DoesNotExist:
            pass
        return None

    @property
    def total_reactions(self):
        """Get total number of reactions"""
        return self.likes + self.dislikes
    
class Job(models.Model):
    title = models.CharField(max_length=200)
    company = models.CharField(max_length=200)
    company_user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    location = models.CharField(max_length=200)
    description = models.TextField()
    requirements = models.CharField(max_length=200)
    salary = models.CharField(max_length=200, default="Not Specified")
    posted_date = models.DateTimeField(default=timezone.now)
    # New fields for matching
    required_skills = models.TextField(help_text="Enter required skills separated by commas", default='')
    min_experience = models.PositiveIntegerField(default=0)
    required_qualification = models.CharField(max_length=50, default='bachelor',choices=[
        ('bachelor', 'Bachelors'),
        ('master', 'Masters'),
        ('phd', 'PhD'),
        ('diploma', 'Diploma'),
    ])

    def __str__(self):
        return f"{self.title} at {self.company}"
    
class Internship(models.Model):
    title = models.CharField(max_length=200)
    company = models.CharField(max_length=200)
    company_user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    location = models.CharField(max_length=200)
    description = models.TextField()
    requirements = models.CharField(max_length=200)  # or TextField()
    salary = models.CharField(max_length=200, default="Not Specified") 
    duration = models.CharField(max_length=100, default="Not Specified")
    posted_date = models.DateTimeField(default=timezone.now)
    # Add other fields as needed

    def __str__(self):
        return f"{self.title} at {self.company}"
    
class Project(models.Model):
    company = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()
    project_type = models.CharField(max_length=100)
    industry = models.CharField(max_length=100)
    budget = models.DecimalField(max_digits=15, decimal_places=2)
    timeline = models.DateField()
    location = models.CharField(max_length=200, blank=True, null=True)
    expertise_required = models.TextField()
    payment_terms = models.TextField()
    nda_required = models.BooleanField(default=False)
    confidentiality_required = models.BooleanField(default=False)
    ip_rights_required = models.BooleanField(default=False)
    custom_field = models.CharField(max_length=200, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class IndividualProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True) # 3
    tagline = models.CharField(max_length=255, blank=True, null=True)
    availability_status = models.CharField(max_length=50, choices=[
        ('actively_looking', 'Actively Looking'),
        ('not_available', 'Not Available'),
        ('open_to_offers', 'Open to Offers')
    ], default='open_to_offers') #3
    about_me = models.TextField(blank=True, null=True)
    key_skills = models.JSONField(blank=True, null=True)  # Store as a list of skills 1
    industries_of_interest = models.JSONField(blank=True, null=True)  # Store as a list 2
    desired_role = models.CharField(max_length=255, blank=True, null=True)
    desired_location = models.CharField(max_length=255, blank=True, null=True)
    work_type = models.CharField(max_length=50, choices=[
        ('remote', 'Remote'),
        ('hybrid', 'Hybrid'),
        ('on_site', 'On-Site')
    ], default='remote') #4
    salary_expected = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True) 

    # Work Experience Section
    current_position_title = models.CharField(max_length=255, blank=True, null=True)
    current_company = models.CharField(max_length=255, blank=True, null=True)
    current_duration = models.CharField(max_length=255, blank=True, null=True)
    key_responsibilities = models.TextField(blank=True, null=True)

    # Education Section
    qualification = models.CharField(max_length=255, blank=True, null=True)#5
    institution = models.CharField(max_length=255, blank=True, null=True)
    year_of_completion = models.IntegerField(blank=True, null=True)
    certifications = models.JSONField(blank=True, null=True)  # Store as a list 6

    # Portfolio/Projects Section
    projects = models.JSONField(blank=True, null=True)  # Store projects as a list of dicts

    # Achievements Section
    awards = models.JSONField(blank=True, null=True)  # Store awards as a list
    publications = models.JSONField(blank=True, null=True)  # Store publications as a list

    # Interests Section
    hobbies = models.JSONField(blank=True, null=True)  # Store as a list

    def __str__(self):
        return f"{self.user.username}'s Profile"
    
class Bid(models.Model):
    # Existing core fields
    project = models.ForeignKey('Project', on_delete=models.CASCADE, related_name='bids')
    bidder = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              limit_choices_to={'user_type': 'company'})
    amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name="Proposed Price")
    
    # Make new required fields nullable with blank=True for forms
    payment_terms = models.TextField(verbose_name="Payment Terms", null=True, blank=True)
    estimated_timeline = models.IntegerField(
        help_text="Estimated days to complete",
        null=True,
        blank=True
    )
    preferred_start_date = models.DateField(null=True, blank=True)
    project_approach = models.TextField(verbose_name="Project Approach", null=True, blank=True)
    team_resources = models.TextField(verbose_name="Team/Resources Description", null=True, blank=True)
    company_profile = models.TextField(verbose_name="Company Profile", null=True, blank=True)
    
    # Optional fields
    portfolio_links = models.TextField(verbose_name="Portfolio/Previous Work Links", blank=True, null=True)
    client_testimonials = models.TextField(verbose_name="Client Testimonials/References", blank=True, null=True)
    minimum_budget_expectation = models.DecimalField(
        max_digits=15, 
        decimal_places=2, 
        null=True, 
        blank=True,
        verbose_name="Minimum Budget Expectation"
    )
    additional_details = models.TextField(blank=True, null=True)
    
    # File fields
    proposal_document = models.FileField(
        upload_to='bid_proposals/',
        null=True,
        blank=True,
        verbose_name="Proposal Document"
    )
    
    
    # Custom fields
    custom_fields = models.JSONField(default=dict, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['project', 'bidder']
        
    def __str__(self):
        return f"Bid by {self.bidder.company_name} on {self.project}"


class FreelanceProject(models.Model):
    CATEGORY_CHOICES = [
        ('Development', 'Development'),
        ('Design', 'Design'),
        ('Marketing', 'Marketing'),
        ('Writing', 'Writing'),
        ('Data Entry', 'Data Entry'),
        ('Other', 'Other'),
    ]

    EXPERIENCE_LEVEL_CHOICES = [
        ('Beginner', 'Beginner'),
        ('Intermediate', 'Intermediate'),
        ('Expert', 'Expert'),
    ]

    PAYMENT_TERMS_CHOICES = [
        ('Milestone-based', 'Milestone-based'),
        ('Hourly', 'Hourly'),
        ('Fixed-price', 'Fixed-price'),
    ]

    DURATION_CHOICES = [
        ('Less than 1 month', 'Less than 1 month'),
        ('1-3 months', '1-3 months'),
    ]

    title = models.CharField(max_length=200)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    description = models.TextField()
    skills_required = models.TextField()  # Store as comma-separated values
    duration = models.CharField(max_length=50, choices=DURATION_CHOICES)
    budget = models.CharField(max_length=50)
    payment_terms = models.CharField(max_length=50, choices=PAYMENT_TERMS_CHOICES)
    experience_level = models.CharField(max_length=50, choices=EXPERIENCE_LEVEL_CHOICES)
    location_preference = models.CharField(max_length=200, blank=True, null=True)
    language_requirements = models.CharField(max_length=200, blank=True, null=True)
    attachments = models.FileField(upload_to='attachments/', blank=True, null=True)
    company_name = models.CharField(max_length=200)
    company_description = models.TextField()
    point_of_contact_name = models.CharField(max_length=100)
    point_of_contact_email = models.EmailField()
    point_of_contact_phone = models.CharField(max_length=15)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
    
# models.py
class FreelanceBid(models.Model):
    project = models.ForeignKey(FreelanceProject, on_delete=models.CASCADE, related_name='bids')
    bidder = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    portfolio_links = models.TextField(help_text="Add your portfolio links (one per line)")
    previous_works = models.TextField(help_text="Describe your relevant previous works")
    expected_pay = models.DecimalField(max_digits=10, decimal_places=2)
    timeline = models.CharField(max_length=100)
    execution_plan = models.TextField(help_text="Describe how you plan to execute this project")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Bid by {self.name} for {self.project.title}"
    

class JobApplication(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    job = models.ForeignKey('Job', on_delete=models.CASCADE)
    name = models.CharField(max_length=255, default='Unknown')
    email = models.EmailField(max_length=254, default='example@example.com') 
    phone_number = models.CharField(max_length=15)
    degree = models.CharField(max_length=50)
    percentage = models.FloatField()
    work_experience = models.PositiveIntegerField()
    resume = models.FileField(upload_to='resumes/')
    applied_at = models.DateTimeField(auto_now_add=True)
    # New field for matching
    skills = models.TextField(default='', help_text="Enter your skills separated by commas")

    def __str__(self):
        return f"{self.user.name} - {self.job.title}"
# your_app/matchers.py

from fuzzywuzzy import fuzz

class JobMatcher:
    def __init__(self):
        self.weights = {
            'skills_match': 0.40,
            'experience_match': 0.30,
            'education_match': 0.30
        }
    
    def calculate_match(self, application, job):
        """Main method to calculate overall match score"""
        scores = {
            'skills_match': self._calculate_skills_score(application.skills, job.required_skills),
            'experience_match': self._calculate_experience_score(
                application.work_experience, 
                job.min_experience
            ),
            'education_match': self._calculate_education_score(
                application.degree, 
                job.required_qualification
            )
        }
        
        weighted_score = sum(
            scores[key] * self.weights[key] 
            for key in scores
        )
        
        return {
            'overall_score': round(weighted_score, 1),
            'detailed_scores': scores,
            'matching_skills': self._get_matching_skills(application.skills, job.required_skills),
            'missing_skills': self._get_missing_skills(application.skills, job.required_skills)
        }
    
    def _calculate_skills_score(self, applicant_skills, job_skills):
        """Calculate skill match percentage using fuzzy matching"""
        if not job_skills:
            return 0
            
        applicant_skills = set(s.strip().lower() for s in applicant_skills.split(','))
        job_skills = set(s.strip().lower() for s in job_skills.split(','))
        
        if not job_skills:
            return 0
            
        matches = sum(
            max(fuzz.ratio(j_skill, a_skill) for a_skill in applicant_skills) > 80
            for j_skill in job_skills
        )
        return (matches / len(job_skills)) * 100
    
    def _calculate_experience_score(self, applicant_exp, required_exp):
        """Calculate experience match percentage"""
        if not required_exp:
            return 100  # No required experience, perfect match
    
    # Case when applicant's experience is much greater than required
        if applicant_exp >= required_exp * 1.5:
        # Penalize slightly, but ensure it doesn't drop too much
            return max(50, 100 * (required_exp * 1.5) / applicant_exp)  # Set a minimum threshold of 50%
    
    # Case when applicant's experience is equal or slightly greater than required
        if applicant_exp >= required_exp:
            return 100
    
    # Case when applicant's experience is less than required
        return (applicant_exp / required_exp) * 100
    
    def _calculate_education_score(self, applicant_edu, required_edu):
        """Calculate education match percentage"""
        education_levels = {
            'high school': 1,
            'diploma': 2,
            'bachelors': 3,
            'bachelor': 3,
            'masters': 4,
            'master': 4,
            'phd': 5,
            'doctorate': 5
        }
    
        
        applicant_level = education_levels.get(applicant_edu.lower(), 0)
        required_level = education_levels.get(required_edu.lower(), 0)
        
        if applicant_level >= required_level:
            return 100
        return (applicant_level / required_level) * 100 if required_level else 0
    
    def _get_matching_skills(self, applicant_skills, job_skills):
        """Get list of matching skills"""
        if not applicant_skills or not job_skills:
            return []
            
        applicant_skills = [s.strip().lower() for s in applicant_skills.split(',')]
        job_skills = [s.strip().lower() for s in job_skills.split(',')]
        
        matching = []
        for job_skill in job_skills:
            if any(fuzz.ratio(job_skill, app_skill) > 80 for app_skill in applicant_skills):
                matching.append(job_skill)
        
        return matching
    
    def _get_missing_skills(self, applicant_skills, job_skills):
        """Get list of missing required skills"""
        if not job_skills:
            return []
            
        applicant_skills = [s.strip().lower() for s in applicant_skills.split(',')]
        job_skills = [s.strip().lower() for s in job_skills.split(',')]
        
        missing = []
        for job_skill in job_skills:
            if not any(fuzz.ratio(job_skill, app_skill) > 80 for app_skill in applicant_skills):
                missing.append(job_skill)
        
        return missing
    
class SavedApplication(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    job_application = models.ForeignKey(JobApplication, on_delete=models.CASCADE)
    overall_match_score = models.FloatField(default=0.0)
    matching_skills = models.TextField(blank=True, help_text="Comma-separated matching skills")
    missing_skills = models.TextField(blank=True, help_text="Comma-separated missing skills")
    category = models.CharField(max_length=100, default="Shortlisted")
    saved_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} saved {self.job_application.job.title}"
    
class LikeDislike(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    idea = models.ForeignKey(Idea, on_delete=models.CASCADE)
    like = models.BooleanField(default=False)
    dislike = models.BooleanField(default=False)

    class Meta:
        unique_together = ('user', 'idea')  # Ensures a user can only like or dislike once
    
    def __str__(self):
        return f"{self.user.username} - {self.idea.title} ({'Like' if self.like else 'Dislike'})"
    
    def clean(self):
        """Ensure only one of like or dislike is True"""
        if self.like and self.dislike:
            raise ValidationError("Cannot like and dislike simultaneously")

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

# assessments/models.py
import json

class Question(models.Model):
    QUESTION_TYPES = [
        ('aptitude', 'Aptitude'),
        ('coding', 'Coding'),
        ('general', 'General'),
    ]

    DIFFICULTY_LEVELS = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]

    question_text = models.TextField()
    answer = models.CharField(max_length=255)
    question_type = models.CharField(max_length=50, choices=QUESTION_TYPES)
    difficulty = models.CharField(max_length=50, choices=DIFFICULTY_LEVELS)

    def __str__(self):
        return f"Question: {self.question_text[:50]}..."

class Assessment(models.Model):
    total_questions = models.IntegerField()
    aptitude_questions = models.IntegerField()
    coding_questions = models.IntegerField()
    questions = models.JSONField()  # This will store a list of question IDs and their order in the assessment
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Assessment ID: {self.id}, Total Questions: {self.total_questions}"

    def save_assessment(self, assessment_data):
        """ Helper function to save the generated assessment data in JSON format """
        self.questions = json.dumps(assessment_data)  # Store question IDs and their order
        self.save()