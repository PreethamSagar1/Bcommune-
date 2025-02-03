from django.db import IntegrityError
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, get_user_model
from django.contrib.auth.decorators import login_required
from .models import Idea, Job, Project, CustomUser, IndividualProfile, Bid, Internship,FreelanceProject, FreelanceBid, JobApplication, JobMatcher, SavedApplication,LikeDislike
from .models import Assessment
from users.forms import CompanySignupForm, IndividualSignupForm,IndividualProfileForm, CompanyProfileForm, BidForm,FreelanceProjectForm,FreelanceBidForm,JobApplicationForm
from django.contrib.auth import logout
from django.http import JsonResponse
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
from .filters import JobApplicationFilter
import json


def logout_view(request):
    if request.user.is_authenticated:
        if request.user.user_type == 'individual':
            logout(request)
            return redirect('individual_login')  # Change 'individual_login' to the actual URL name for individual login
        elif request.user.user_type == 'company':
            logout(request)
            return redirect('company_login')  # Change 'company_login' to the actual URL name for company login
    return redirect('home') 

def home(request):
    return render(request, 'home.html')

def individual_signup(request):
    if request.method == 'POST':
        form = IndividualSignupForm(request.POST)
        if form.is_valid():
            if form.cleaned_data['password'] != form.cleaned_data['confirm_password']:
                messages.error(request, "Passwords do not match.")
                return redirect('individual_signup')

            if CustomUser.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists.")
                return redirect('individual_signup')

            try:
                user = CustomUser.objects.create_user(
                    username=form.cleaned_data['email'],
                    email=form.cleaned_data['email'],
                    password=form.cleaned_data['password'],
                    first_name=form.cleaned_data['name'],
                    user_type='individual'
                )

                # Create associated profile
                IndividualProfile.objects.create(user=user)

                messages.success(request, "Account created successfully!")
                return redirect('individual_login')
            except Exception as e:
                messages.error(request, f"Error creating account: {str(e)}")
                return redirect('individual_signup')
    else:
        form = IndividualSignupForm()
    return render(request, 'individual_signup.html', {'form': form})

def individual_login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        
        user = authenticate(request, username=email, password=password)
        
        if user is not None and user.user_type == 'individual':
            login(request, user)
            return redirect('individual_dashboard')
        else:
            messages.error(request, "Invalid credentials or wrong user type.")
            return redirect('individual_login')
    
    return render(request, 'individual_login.html')

@login_required
def individual_dashboard(request):
    if request.user.user_type != 'individual':
        return redirect('individual_login')

    # Fetch all ideas
    ideas = Idea.objects.all().order_by('-created_at')[:3]

    # Fetch jobs and order by the most recent
    jobs = Job.objects.all().order_by('-posted_date')[:3]

    # Fetch top 3 most recent freelance projects
    freelance_projects = FreelanceProject.objects.all().order_by('-created_at')[:3]

    # Render the dashboard
    return render(
        request,
        'individual_dashboard.html',
        {
            'ideas': ideas,
            'jobs': jobs,
            'freelance_projects': freelance_projects,
        }
    )

def company_signup(request):
    if request.method == 'POST':
        form = CompanySignupForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                messages.success(request, "Company account created successfully!")
                return redirect('company_login')
            except Exception as e:
                messages.error(request, f"Error creating account: {str(e)}")
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = CompanySignupForm()
    return render(request, 'company_signup.html', {'form': form})

def company_login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        
        user = authenticate(request, username=email, password=password)
        
        if user is not None and user.user_type == 'company':
            login(request, user)
            return redirect('company_dashboard')
        else:
            messages.error(request, "Invalid credentials or wrong user type.")
    return render(request, 'company_login.html')


# View for Company Dashboard (requires login)
@login_required
def company_dashboard(request):
    if request.user.user_type != 'company':
        return redirect('company_login')
    
    ideas = Idea.objects.all().order_by('-created_at')[:3]
    recent_projects = Project.objects.exclude(company=request.user).order_by('-created_at')[:3]
    my_projects = Project.objects.filter(company=request.user).order_by('-created_at')[:3]
    
    context = {
        'ideas': ideas,
        'recent_projects': recent_projects,
        'my_projects': my_projects,
        'company_name': request.user.company_name,
    }
    return render(request, 'company_dashboard.html', context)
def ideaform(request):
    return render(request, 'ideaform.html')

def submit_idea(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        patent_number = request.POST.get('patent_number')
        brief_description = request.POST.get('brief_description')
        application_number = request.POST.get('application_number')
        problem_statement = request.POST.get('problem_statement')
        solution = request.POST.get('solution')
        visibility = request.POST.get('visibility')
        details = request.POST.get('details')
        fund = request.POST.get('fund')
        category = request.POST.get('category')
        photo = request.FILES.get('photo')
        video = request.FILES.get('video')
        team_info = request.POST.get('team_info')
        user_email = request.POST.get('user_email')

        # Save to database
        idea = Idea(
            title=title,
            patent_number=patent_number,
            brief_description=brief_description,
            application_number=application_number,
            problem_statement=problem_statement,
            solution=solution,
            visibility=visibility,
            details=details,
            fund=fund,
            category=category,
            photo=photo,
            video=video,
            team_info=team_info,
            email = user_email,
        )
        idea.save()

        return JsonResponse({'message': 'Idea submitted successfully!'})

    return JsonResponse({'error': 'Invalid request method!'}, status=400)


def ideas_and_invest(request):
    return render(request, 'ideasandinvest.html')




@login_required
def myprojects(request):
    # Check if the user is a company user
    other_company_projects = Project.objects.exclude(company=request.user).order_by('-created_at')
    
    # Get projects posted by the current company (for portfolio section)
    my_projects = Project.objects.filter(company=request.user).prefetch_related('bids').order_by('-created_at')
    
    context = {
        'projects': other_company_projects,
        'my_projects': my_projects
    }
    return render(request, 'myprojects.html', context)
@login_required
def post_project(request):
    if request.method == 'POST':
        # Create new project from form data
        project = Project(
            company=request.user,
            title=request.POST['title'],
            description=request.POST['description'],
            project_type=request.POST['type'],
            industry=request.POST['industry'],
            budget=request.POST['budget'],
            timeline=request.POST['timeline'],
            location=request.POST['location'],
            expertise_required=request.POST['expertise'],
            payment_terms=request.POST['payment-terms'],
        )
        project.save()
        messages.success(request, 'Project posted successfully!')
        return redirect('myprojects')
    
    return render(request, 'myprojectform.html')
    
    
def myprojectform(request):
    return render(request, 'myprojectform.html') 

@login_required
def delete_project(request, project_id):
    try:
        project = Project.objects.get(id=project_id, company=request.user)
        project.delete()
        messages.success(request, 'Project deleted successfully!')
    except Project.DoesNotExist:
        messages.error(request, 'Project not found or you do not have permission to delete it.')
    return redirect('myprojects')

@login_required
def edit_project(request, project_id):
    try:
        project = get_object_or_404(Project, id=project_id, company=request.user)
        
        if request.method == 'POST':
            project.title = request.POST['title']
            project.description = request.POST['description']
            project.project_type = request.POST['type']
            project.industry = request.POST['industry']
            project.budget = request.POST['budget']
            project.timeline = request.POST['timeline']  # This will now be in YYYY-MM-DD format
            project.location = request.POST['location']
            project.expertise_required = request.POST['expertise']
            project.payment_terms = request.POST['payment-terms']
            
            # Handle boolean fields
            project.nda_required = 'nda_required' in request.POST
            project.confidentiality_required = 'confidentiality_required' in request.POST
            project.ip_rights_required = 'ip_rights_required' in request.POST
            
            # Handle optional field
            project.custom_field = request.POST.get('custom_field', '')
            
            project.save()
            messages.success(request, 'Project updated successfully!')
            return redirect('myprojects')
            
        return render(request, 'edit_project.html', {'project': project})
    except Exception as e:
        messages.error(request, f'Error editing project: {str(e)}')
        return redirect('myprojects')
    

def explore_all_ideas(request):
    ideas = Idea.objects.filter(visibility='public').order_by('-created_at')
    return render(request, 'explore_all_ideas.html', {'ideas': ideas})

def idea_detail(request, idea_id):
    idea = get_object_or_404(Idea, id=idea_id)
    context = {
        'idea': idea,
        'user_reaction': idea.get_like_status(request.user) if request.user.is_authenticated else None
    }
    return render(request, 'idea_detail.html', context)

def edit_job(request, job_id):
    job = get_object_or_404(Job, id=job_id, company_user=request.user)
    
    if request.method == 'POST':
        try:
            # Validate minimum experience
            min_experience = request.POST.get('min_experience')
            if not min_experience or not min_experience.isdigit():
                raise ValidationError('Minimum experience must be a valid number')
            
            # Validate required skills
            required_skills = request.POST.get('skills', '').strip()
            if not required_skills:
                raise ValidationError('Required skills cannot be empty')
            # Clean and validate skills format
            skills_list = [skill.strip() for skill in required_skills.split(',')]
            skills_list = [skill for skill in skills_list if skill]  # Remove empty entries
            if not skills_list:
                raise ValidationError('At least one valid skill must be provided')
            
            # Validate qualification
            qualification = request.POST.get('required_qualification')
            valid_qualifications = ['bachelor', 'master', 'phd']
            if not qualification or qualification.lower() not in valid_qualifications:
                raise ValidationError('Invalid qualification selected')
            
            # Update job fields
            job.title = request.POST.get('job_title')
            job.company = request.POST.get('company_name')
            job.location = request.POST.get('job_location')
            job.description = request.POST.get('job_description')
            job.requirements = request.POST.get('job_type')
            job.salary = request.POST.get('salary')
            job.required_skills = ','.join(skills_list)
            job.required_qualification = qualification.lower()
            job.min_experience = int(min_experience)
            
            # Perform model validation
            job.full_clean()
            job.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Job updated successfully!'
            })
            
        except ValidationError as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e),
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': 'An unexpected error occurred while updating the job.',
            }, status=500)
    
    return render(request, 'edit_job.html', {'job': job})
@login_required
@require_http_methods(["POST"])
def delete_job(request, job_id):
    try:
        job = get_object_or_404(Job, id=job_id, company_user=request.user)
        job.delete()
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)




@login_required
def explore_projects(request):
    # Fetch projects excluding those belonging to the logged-in company
    other_company_projects = Project.objects.exclude(company=request.user).order_by('-created_at')
    
    # Handle AJAX request for project details
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        project_id = request.GET.get('project_id')
        if project_id:
            try:
                project = Project.objects.get(id=project_id)
                data = {
                    'title': project.title,
                    'description': project.description,
                    'project_type': project.project_type,
                    'industry': project.industry,
                    'budget': float(project.budget),
                    'timeline': project.timeline.strftime('%Y-%m-%d'),
                    'location': project.location or 'Not specified',
                    'expertise_required': project.expertise_required,
                    'payment_terms': project.payment_terms,
                    'nda_required': project.nda_required,
                    'confidentiality_required': project.confidentiality_required,
                    'ip_rights_required': project.ip_rights_required,
                    'custom_field': getattr(project, 'custom_field', None),  # Safely handle custom_field
                    'created_at': project.created_at.strftime('%Y-%m-%d %H:%M:%S')
                }
                return JsonResponse(data)
            except Project.DoesNotExist:
                return JsonResponse({'error': 'Project not found'}, status=404)
        else:
            return JsonResponse({'error': 'Invalid project ID'}, status=400)

    # Get selected project ID from URL parameter
    selected_project_id = request.GET.get('selected_project')

    # Determine the project to display first
    first_project = None
    if selected_project_id:
        try:
            first_project = Project.objects.get(id=selected_project_id)
        except Project.DoesNotExist:
            first_project = other_company_projects.first()
    else:
        first_project = other_company_projects.first()
    
    # Context for rendering the template
    context = {
        'projects': other_company_projects,
        'first_project': first_project,
        'selected_project_id': selected_project_id,
    }
    return render(request, 'explore_projects.html', context)

@login_required
def edit_individual_profile(request):
    try:
        profile = IndividualProfile.objects.get(user=request.user)
    except IndividualProfile.DoesNotExist:
        profile = IndividualProfile(user=request.user)

    if request.method == 'POST':
        form = IndividualProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('individual_profile')  # Replace with the desired redirect URL
    else:
        form = IndividualProfileForm(instance=profile)

    return render(request, 'edit_individual_profile.html', {'form': form})

@login_required
def individual_profile(request):
    profile = get_object_or_404(IndividualProfile, user=request.user)
    return render(request, 'individual_profile.html', {'profile': profile})


@login_required
def company_profile(request):
    user = request.user
    if user.user_type != 'company':  # Check the user_type field
        return redirect('home')  # Redirect if the user is not a company.

    if request.method == 'POST':
        form = CompanyProfileForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            form.save()
            return redirect('company_profile')
    else:
        form = CompanyProfileForm(instance=user)

    return render(request, 'company_profile.html', {'form': form, 'user': user})

@login_required
def faq(request):
    return render(request,'faq.html')

@login_required
def place_project_bid(request, project_id):
    # Get the project or return 404
    project = get_object_or_404(Project, id=project_id)
    
    # Check if user is a company
    if request.user.user_type != 'company':
        messages.error(request, "Only companies can place bids!")
        return redirect('explore_projects')
    
    # Prevent project owner from bidding on their own project
    if project.company == request.user:
        messages.error(request, "You cannot bid on your own project!")
        return redirect('explore_projects')
    
    # Check if the bidder has already placed a bid
    if Bid.objects.filter(project=project, bidder=request.user).exists():
        messages.error(request, "You have already placed a bid on this project.")
        return redirect('explore_projects')
    
    if request.method == 'POST':
        form = BidForm(request.POST, request.FILES)
        if form.is_valid():
            bid = form.save(commit=False)
            bid.project = project
            bid.bidder = request.user
            
            # Handle custom fields
            custom_fields = {}
            for key, value in request.POST.items():
                if key.startswith('custom_field_name_'):
                    field_num = key.split('_')[-1]
                    field_name = value
                    field_value = request.POST.get(f'custom_field_value_{field_num}')
                    if field_name and field_value:
                        custom_fields[field_name] = field_value
            
            bid.custom_fields = custom_fields
            
            try:
                bid.save()
                messages.success(request, 'Your bid has been submitted successfully!')
                return redirect('explore_projects')
            except IntegrityError:
                messages.error(request, "An error occurred while saving your bid. Please try again.")
        else:
            messages.error(request, "There was an error in your bid form. Please correct it and try again.")
    else:
        form = BidForm()
    
    context = {
        'form': form,
        'project': project,
    }
    return render(request, 'bidform.html', context)

def myportfolio(request):
    if not request.user.is_authenticated:
        return redirect('company_login')
    
    # Fetch all jobs for the currently logged-in company user
    recent_jobs = Job.objects.filter(company_user=request.user).order_by('-posted_date')[:3]
    recent_projects = Project.objects.filter(company=request.user).order_by('-created_at')[:3]
    recent_ideas = Idea.objects.all().order_by('-created_at')[:3]
    recent_internships = Internship.objects.filter(company_user=request.user).order_by('-posted_date')[:3]
    # Add this line to fetch recent freelance projects
    recent_freelance_projects = FreelanceProject.objects.filter(user=request.user).order_by('-created_at')[:3]
    
    return render(request, 'myportfolio.html', {
        'jobs': recent_jobs,
        'projects': recent_projects,
        'ideas': recent_ideas,
        'internships': recent_internships,
        'freelance_projects': recent_freelance_projects  # Add this line
    })


def all_jobs(request):
    if not request.user.is_authenticated:
        return redirect('login')

    jobs = Job.objects.filter(company_user=request.user)
    
    # Get the job_id from URL parameters
    selected_job_id = request.GET.get('job_id')
    
    return render(request, 'all_jobs.html', {
        'jobs': jobs,
        'selected_job_id': selected_job_id
    })

def all_projects(request):
    if not request.user.is_authenticated:
        return redirect('company_login')
    
    projects = Project.objects.filter(company=request.user).order_by('-created_at')
    selected_project_id = request.GET.get('project_id')
    
    return render(request, 'all_projects.html', {
        'projects': projects,
        'selected_project_id': selected_project_id
    })

def all_ideas(request):
    if not request.user.is_authenticated:
        return redirect('company_login')
    
    ideas = Idea.objects.all().order_by('-created_at')
    selected_idea_id = request.GET.get('idea_id')
    
    return render(request, 'all_ideas.html', {
        'ideas': ideas,
        'selected_idea_id': selected_idea_id
    })

def myinternships(request):
    if not request.user.is_authenticated:
        return redirect('company_login')
    company_internships = Internship.objects.filter(company_user=request.user).order_by('-posted_date')
    return render(request,'myinternships.html',{'internships':company_internships})

def post_internship(request):
    if request.method == 'POST':
        try:
            internship = Internship.objects.create(
                title=request.POST.get('job_title'),  # Matches corrected form
                company=request.POST.get('company_name'),
                location=request.POST.get('job_location'),  # Matches corrected form
                description=request.POST.get('job_description'),  # Matches corrected form
                requirements=request.POST.get('job_type'),
                duration=request.POST.get('duration'),
                salary=request.POST.get('salary'),
                company_user=request.user
            )
            return JsonResponse({'status': 'success', 'message': 'Internship posted successfully!'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return render(request, 'post_internship.html')


def myjobs(request):
    if not request.user.is_authenticated:
        return redirect('company_login')
    
    # Get jobs for the current company user
    company_jobs = Job.objects.filter(company_user=request.user).order_by('-posted_date')
    return render(request, 'myjobs.html', {'jobs': company_jobs})

def post_job(request):
    if request.method == 'POST':
        try:
            job = Job.objects.create(
                title=request.POST.get('job_title'),
                company=request.POST.get('company_name'),
                location=request.POST.get('job_location'),
                description=request.POST.get('job_description'),
                requirements=request.POST.get('job_type'),
                salary=request.POST.get('salary'),
                company_user=request.user,  # Add this line
                required_skills=request.POST.get('required_skills', ''),  # New field
                min_experience=request.POST.get('min_experience', 0),  # New field
                required_qualification=request.POST.get('required_qualification', 'bachelor')
            )
            return JsonResponse({'status': 'success', 'message': 'Job posted successfully!'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return render(request, 'create_job.html')


@login_required
@require_http_methods(["POST"])
def delete_internship(request, internship_id):
    try:
        internship = get_object_or_404(Internship, id=internship_id, company_user=request.user)
        internship.delete()
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
def edit_internship(request, internship_id):  # Changed parameter name to internship_id
    internship = get_object_or_404(Internship, id=internship_id, company_user=request.user)
    
    if request.method == 'POST':
        try:
            internship.title = request.POST.get('job_title')  # Updated with internship variable
            internship.company = request.POST.get('company_name')
            internship.location = request.POST.get('job_location')
            internship.description = request.POST.get('job_description')
            internship.requirements = request.POST.get('job_type')
            internship.duration = request.POST.get('duration')
            internship.salary = request.POST.get('salary')
            internship.save()
            return JsonResponse({'status': 'success', 'message': 'Internship updated successfully!'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return render(request, 'edit_internship.html', {'internship': internship})

def freelance(request):
    if not request.user.is_authenticated:
        return redirect('company_login')
    projects = FreelanceProject.objects.filter(user=request.user)
    return render(request, 'freelance.html', {'projects': projects})

def post_freelance(request):
    if request.method == 'POST':
        form = FreelanceProjectForm(request.POST, request.FILES)
        if form.is_valid():
            project_data = form.cleaned_data
            request.session['project_data'] = project_data  # Save form data in session for preview
            return redirect('preview_freelance_project')
    else:
        form = FreelanceProjectForm()

    return render(request, 'post_freelance.html', {'form': form})

def preview_freelance_project(request):
    project_data = request.session.get('project_data', None)
    if not project_data:
        return redirect('post_project')

    if request.method == 'POST':  # If confirmed, save the project
        form = FreelanceProjectForm(project_data)
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user  # Associate project with logged-in user
            project.save()
            del request.session['project_data']  # Clear session
            return redirect('project_success')

    return render(request, 'preview_freelance_project.html', {'project_data': project_data})

def project_success(request):
    return render(request, 'project_success.html')

@login_required
@require_http_methods(["POST"])
def delete_freelance(request, project_id):
    """
    Delete a freelance project by its ID.
    """
    try:
        project = get_object_or_404(FreelanceProject, id=project_id, user=request.user)
        project.delete()
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@login_required
def edit_freelance(request, project_id):
    """
    Edit a freelance project by its ID.
    """
    project = get_object_or_404(FreelanceProject, id=project_id, user=request.user)

    if request.method == 'POST':
        try:
            # Bind the form to the POST data and the existing instance
            form = FreelanceProjectForm(request.POST, request.FILES, instance=project)
            if form.is_valid():
                form.save()
                return JsonResponse({'status': 'success', 'message': 'Freelance project updated successfully!'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid data submitted.'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    # Render the edit form with the existing project data
    form = FreelanceProjectForm(instance=project)
    return render(request, 'edit_freelance.html', {'form': form, 'project': project})

@login_required
def view_bids(request, project_id):
    # Fetch the project and its associated bids
    project = get_object_or_404(Project, id=project_id)
    
    # Ensure user has permission to view bids (project owner or admin)
    if project.company != request.user and not request.user.is_staff:
        messages.error(request, "You don't have permission to view these bids.")
        return redirect('explore_projects')
    
    # Get bids with related bidder information
    bids = project.bids.select_related('bidder').order_by('-created_at')
    
    context = {
        "project": project,
        "bids": bids,
    }
    return render(request, "view_bids.html", context)

def all_internships(request):
    internships = Internship.objects.filter(company_user=request.user).order_by('-posted_date')
    
    return render(request, 'all_internships.html', {'internships': internships})

def all_freelance_projects(request):
    freelance_projects = FreelanceProject.objects.all().order_by('-created_at')
    return render(request, 'all_freelance_projects.html', {'freelance_projects': freelance_projects})


def all_freelance_projects_company(request):
    # Ensure the user is authenticated and fetch projects belonging to the logged-in user
    if not request.user.is_authenticated:
        return redirect('login')  # Redirect to login page if not authenticated

    # Filter projects by the logged-in user (assuming company users are linked via the `user` field)
    freelance_projects = FreelanceProject.objects.filter(user=request.user).order_by('-created_at')

    return render(request, 'all_freelance_projects_company.html', {'freelance_projects': freelance_projects})



@login_required
def place_freelance_bid(request, project_id):
    project = get_object_or_404(FreelanceProject, id=project_id)
    
    # Check if user has already placed a bid
    existing_bid = FreelanceBid.objects.filter(project=project, bidder=request.user).first()
    if existing_bid:
        return redirect('view_my_freelance_bid', project_id=project_id)
    
    if request.method == 'POST':
        form = FreelanceBidForm(request.POST)
        if form.is_valid():
            bid = form.save(commit=False)
            bid.project = project
            bid.bidder = request.user
            bid.save()
            messages.success(request, 'Your bid has been submitted successfully!')
            return redirect('all_freelance_projects')
    else:
        form = FreelanceBidForm(initial={
            'name': request.user.get_full_name(),
            'email': request.user.email
        })
    
    return render(request, 'place_freelance_bid.html', {
        'form': form,
        'project': project
    })

@login_required
def view_my_freelance_bid(request, project_id):
    project = get_object_or_404(FreelanceProject, id=project_id)
    bid = get_object_or_404(FreelanceBid, project=project, bidder=request.user)
    return render(request, 'view_my_freelance_bid.html', {
        'bid': bid,
        'project': project
    })

def view_freelance_bids(request, project_id):
    # Fetch the project and its bids
    project = get_object_or_404(FreelanceProject, id=project_id)
    bids = FreelanceBid.objects.filter(project=project).order_by('-created_at')

    return render(request, 'view_freelance_bids.html', {
        'project': project,
        'bids': bids,
    })

def individual_alljobs(request):
    if not request.user.is_authenticated:
        return redirect('login')
    
    # Fetch all jobs and order by most recent first
    jobs = Job.objects.all().order_by('-posted_date')  # Assuming `posted_date` is the field for job posting date

    # Get the job_id from URL parameters (if any)
    selected_job_id = request.GET.get('job_id')

    return render(request, 'individual_alljobs.html', {
        'jobs': jobs,
        'selected_job_id': selected_job_id
    })

def apply_job(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    
    if request.method == 'POST':
        form = JobApplicationForm(request.POST, request.FILES)
        if form.is_valid():
            application = form.save(commit=False)
            application.job = job
            application.user = request.user
            application.name = f"{request.user.first_name} {request.user.last_name}"  # Get the user's name
            application.email = request.user.email
            application.save()
            job.applications_count = JobApplication.objects.filter(job=job).count()
            job.save()
            messages.success(request, 'You have successfully applied for the job!')
            return redirect('job_success')  # Redirect to a success page
    
    else:
        form = JobApplicationForm()
        
    job.applications_count = JobApplication.objects.filter(job=job).count()
    return render(request, 'apply_jobs.html', {'job': job, 'form': form})


def job_success(request):
    return render(request,'job_success.html')

@login_required
def view_applications(request, job_id):
    # Get the job and verify ownership
    job = get_object_or_404(Job, id=job_id, company_user=request.user)
    
    # Get all applications for this job
    applications = JobApplication.objects.filter(job=job)
    
    # Apply filters
    application_filter = JobApplicationFilter(request.GET, queryset=applications)
    filtered_applications = application_filter.qs
    
    # Calculate match scores for filtered applications
    matcher = JobMatcher()
    applications_with_scores = []
    
    for application in filtered_applications:
        match_results = matcher.calculate_match(application, job)
        applications_with_scores.append({
            'application': application,
            'match_score': match_results['overall_score'],
            'detailed_scores': match_results['detailed_scores'],
            'matching_skills': match_results['matching_skills'],
            'missing_skills': match_results['missing_skills']
        })
    
    # Sort applications by match score (highest first)
    applications_with_scores.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Add sorting by filter parameters if specified
    sort_by = request.GET.get('sort_by')
    if sort_by:
        if sort_by == 'date':
            applications_with_scores.sort(
                key=lambda x: x['application'].applied_at, 
                reverse=True
            )
        elif sort_by == 'education':
            applications_with_scores.sort(
                key=lambda x: x['application'].percentage, 
                reverse=True
            )
        elif sort_by == 'experience':
            applications_with_scores.sort(
                key=lambda x: x['application'].work_experience, 
                reverse=True
            )
    
    context = {
        'job': job,
        'applications': applications_with_scores,
        'filter': application_filter,
        'sort_by': sort_by,
        'total_applications': applications.count(),
        'filtered_count': len(applications_with_scores)
    }
    
    return render(request, 'view_applications.html', context)

@login_required
def save_application(request, app_id):
    if request.method != 'POST':
        return JsonResponse({'message': "Invalid request method!"}, status=400)

    job_application = get_object_or_404(JobApplication, id=app_id)
    
    # Parse the JSON data from request body
    try:
        data = json.loads(request.body)
        category = data.get('category', 'Shortlisted')
    except json.JSONDecodeError:
        category = 'Shortlisted'

    # Check if the application is already saved
    saved_app = SavedApplication.objects.filter(
        user=request.user, 
        job_application=job_application
    ).first()

    if saved_app:
        return JsonResponse({'message': "Application already saved!"})
    
    # Calculate match scores
    matcher = JobMatcher()
    match_result = matcher.calculate_match(job_application, job_application.job)

    # Create new saved application
    saved_app = SavedApplication.objects.create(
        user=request.user,
        job_application=job_application,
        overall_match_score=match_result['overall_score'],
        matching_skills=', '.join(match_result['matching_skills']),
        missing_skills=', '.join(match_result['missing_skills']),
        category=category
    )

    return JsonResponse({'message': "Application saved successfully!"})

@login_required
def saved_applications(request):
    saved_apps = SavedApplication.objects.filter(user=request.user).order_by('-saved_at')
    
    # Filter by category if provided
    category = request.GET.get('category')
    if category:
        saved_apps = saved_apps.filter(category=category)
    
    # Sorting by match score or saved time
    sort_by = request.GET.get('sort', 'saved_at')
    saved_apps = saved_apps.order_by(f"-{sort_by}")
    
    context = {
        'saved_apps': saved_apps,
        'categories': saved_apps.values_list('category', flat=True).distinct(),
    }
    return render(request, 'saved_applications.html', context)

@login_required
def update_category(request, saved_id):
    if request.method == 'POST':
        saved_app = get_object_or_404(SavedApplication, id=saved_id, user=request.user)
        new_category = request.POST.get('category')
        if new_category:
            saved_app.category = new_category
            saved_app.save()
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'success'})
    return redirect('saved_applications')

def like_idea(request, idea_id):
    idea = get_object_or_404(Idea, id=idea_id)
    user = request.user

    # Check if user already liked or disliked this idea
    obj, created = LikeDislike.objects.get_or_create(user=user, idea=idea)
    
    # If the user already liked, toggle off. If disliked, toggle to like.
    if obj.dislike:
        obj.dislike = False
        obj.like = True
    elif obj.like:
        obj.like = False
    else:
        obj.like = True

    obj.save()
    
    return JsonResponse({'likes': idea.likes,'dislikes': idea.dislikes,'message': 'Like toggled successfully!'}, status=200)

def dislike_idea(request, idea_id):
    idea = get_object_or_404(Idea, id=idea_id)
    user = request.user

    # Check if user already liked or disliked this idea
    obj, created = LikeDislike.objects.get_or_create(user=user, idea=idea)
    
    # If the user already disliked, toggle off. If liked, toggle to dislike.
    if obj.like:
        obj.like = False
        obj.dislike = True
    elif obj.dislike:
        obj.dislike = False
    else:
        obj.dislike = True

    obj.save()
    
    return JsonResponse({'likes': idea.likes,'dislikes': idea.dislikes,'message': 'Dislike toggled successfully!'}, status=200)

def assessmentc(request):
    return render(request,'assessment/assessmentc.html')

def create_own_assessment(request):
    if request.method == 'POST':
        total_questions = int(request.POST['total_questions'])
        aptitude_questions = int(request.POST['aptitude_questions'])
        coding_questions = int(request.POST['coding_questions'])

        if aptitude_questions + coding_questions != total_questions:
            return HttpResponse("The sum of aptitude and coding questions must equal total questions", status=400)

        file = request.FILES['input_file']
        if file.name.endswith('.json'):
            questions = json.load(file)
        elif file.name.endswith('.docx'):
            questions = read_questions_from_docx(file)
        else:
            return HttpResponse("Unsupported file type", status=400)

        assessment = generate_assessment(questions, total_questions, aptitude_questions, coding_questions)
        save_assessment(assessment)
        return render(request, 'assessment/success.html', {'assessment': assessment})

    return render(request, 'assessment/create_own.html')

def prefer_bcommune_assessment(request):
    if request.method == 'POST':
        total_questions = int(request.POST['total_questions'])
        aptitude_questions = int(request.POST['aptitude_questions'])
        coding_questions = int(request.POST['coding_questions'])
        difficulty = request.POST['difficulty']

        if aptitude_questions + coding_questions != total_questions:
            return HttpResponse("The sum of aptitude and coding questions must equal total questions", status=400)

        with open('static/js/q&a.json', 'r') as file:
            questions = json.loads(file.read())

        filtered_questions = filter_questions_by_difficulty(questions, difficulty)
        assessment = generate_assessment(filtered_questions, total_questions, aptitude_questions, coding_questions)
        save_assessment(assessment)
        return render(request, 'assessment/success.html', {'assessment': assessment})

    return render(request, 'assessment/prefer_bcommune.html')

def view_assessment(request):
    with open('assessment.json', 'r') as file:
        assessment = json.load(file)

    if request.method == 'POST':
        user_answers = request.POST
        score = calculate_score(assessment, user_answers)
        return render(request, 'assessment/score.html', {'score': score, 'total': len(assessment)})

    return render(request, 'assessment/view_assessment.html', {'assessment': assessment})

# Helper functions
def read_questions_from_docx(file):
    # Logic to parse questions from a Word document
    pass

import random

def generate_assessment(questions, total, aptitude, coding):
    aptitude_questions = [q for q in questions if q['type'] == 'aptitude'][:aptitude]
    coding_questions = [q for q in questions if q['type'] == 'coding'][:coding]

    remaining_slots = total - len(aptitude_questions) - len(coding_questions)
    additional_questions = [q for q in questions if q not in aptitude_questions + coding_questions]
    selected_additional = additional_questions[:remaining_slots]

    assessment = aptitude_questions + coding_questions + selected_additional
    random.shuffle(assessment)

    for idx, question in enumerate(assessment):
        question['id'] = idx + 1

    return assessment

def save_assessment(assessment):
    with open('assessment.json', 'w') as file:
        json.dump(assessment, file)

def filter_questions_by_difficulty(questions, difficulty):
    return [q for q in questions if q['difficulty'] == difficulty]

def calculate_score(assessment, user_answers):
    score = 0
    for idx, question in enumerate(assessment):
        question_id = str(question.get('id', idx))
        if user_answers.get(question_id) == question['answer']:
            score += 1
    return score

def success(request):
    return(request,'success.html')

# AI-search-&-recommender
