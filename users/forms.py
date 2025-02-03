from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from .models import CustomUser, IndividualProfile, Bid, FreelanceProject, FreelanceBid, JobApplication
CustomUser = get_user_model()
class CompanySignupForm(UserCreationForm):
    INDUSTRY_CHOICES = [
        ('Agriculture', 'Agriculture'),
        ('Automotive', 'Automotive'),
        ('Banking', 'Banking'),
        ('Construction', 'Construction'),
        ('Consumer Goods', 'Consumer Goods'),
        ('Education', 'Education'),
        ('Energy', 'Energy'),
        ('Entertainment', 'Entertainment'),
        ('Finance', 'Finance'),
        ('Healthcare', 'Healthcare'),
        ('Hospitality', 'Hospitality'),
        ('Information Technology', 'Information Technology'),
        ('Insurance', 'Insurance'),
        ('Logistics', 'Logistics'),
        ('Manufacturing', 'Manufacturing'),
        ('Media', 'Media'),
        ('Pharmaceuticals', 'Pharmaceuticals'),
        ('Real Estate', 'Real Estate'),
        ('Retail', 'Retail'),
        ('Telecommunications', 'Telecommunications'),
        ('Transportation', 'Transportation'),
        ('Utilities', 'Utilities'),
        ('Other', 'Other'),
    ]

    COMPANY_SIZE_CHOICES = [
        ('1-10', '1-10'),
        ('11-50', '11-50'),
        ('51-200', '51-200'),
        ('201-500', '201-500'),
        ('501-1000', '501-1000'),
        ('1000+', '1000+'),
    ]

    company_name = forms.CharField(max_length=255, widget=forms.TextInput(attrs={'class': 'form-control'}))
    company_website = forms.URLField(max_length=200, widget=forms.URLInput(attrs={'class': 'form-control'}))
    industry = forms.ChoiceField(choices=INDUSTRY_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    company_size = forms.ChoiceField(choices=COMPANY_SIZE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    company_type = forms.CharField(max_length=255, widget=forms.TextInput(attrs={'class': 'form-control'}))
    person_name = forms.CharField(max_length=255, widget=forms.TextInput(attrs={'class': 'form-control'}))
    designation = forms.CharField(max_length=255, widget=forms.TextInput(attrs={'class': 'form-control'}))
    company_mail = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    phone_number = forms.CharField(max_length=15, widget=forms.TextInput(attrs={'class': 'form-control', 'pattern': '[0-9]{10}'}))
    bcommune_profile = forms.URLField(widget=forms.URLInput(attrs={'class': 'form-control'}))
    is_company = forms.BooleanField(initial=True, widget=forms.HiddenInput())
    class Meta:
        model = CustomUser
        fields = ['username', 'password1', 'password2', 'company_name', 'company_website', 'industry', 
                 'company_size', 'company_type', 'person_name', 'designation', 'company_mail', 
                 'phone_number', 'bcommune_profile']
    def clean(self):
        cleaned_data = super().clean()
        company_mail = cleaned_data.get('company_mail')
        
        if company_mail:
            if CustomUser.objects.filter(email=company_mail).exists():
                raise forms.ValidationError("This email is already registered.")
            cleaned_data['username'] = company_mail
            
        return cleaned_data    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.username = self.cleaned_data['company_mail']
        user.email = self.cleaned_data['company_mail']
        user.is_company = True
        user.user_type = 'company'
        user.company_name = self.cleaned_data['company_name']
        user.company_website = self.cleaned_data['company_website']
        user.industry = self.cleaned_data['industry']
        user.company_size = self.cleaned_data['company_size']
        user.company_type = self.cleaned_data['company_type']
        user.designation = self.cleaned_data['designation']
        user.phone_number = self.cleaned_data['phone_number']
        user.bcommune_profile = self.cleaned_data['bcommune_profile']
        
        if commit:
            user.save()
        return user
    
class IndividualSignupForm(forms.Form):
    name = forms.CharField(max_length=255)
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)


class IndividualProfileForm(forms.ModelForm):
    class Meta:
        model = IndividualProfile
        fields = [
            'profile_picture', 'location', 'tagline', 'availability_status', 'about_me',
            'key_skills', 'industries_of_interest', 'desired_role', 'desired_location',
            'work_type', 'salary_expected', 'current_position_title', 'current_company',
            'current_duration', 'key_responsibilities', 'qualification', 'institution',
            'year_of_completion', 'certifications', 'projects', 'awards', 'publications', 'hobbies'
        ]
        widgets = {
            'about_me': forms.Textarea(attrs={'rows': 4}),
            'key_skills': forms.TextInput(attrs={'placeholder': 'e.g., Python, Django, React'}),
            'industries_of_interest': forms.TextInput(attrs={'placeholder': 'e.g., Tech, Marketing'}),
            'certifications': forms.Textarea(attrs={'rows': 2}),
            'projects': forms.Textarea(attrs={'rows': 3}),
            'awards': forms.Textarea(attrs={'rows': 2}),
            'publications': forms.Textarea(attrs={'rows': 2}),
            'hobbies': forms.TextInput(attrs={'placeholder': 'e.g., Reading, Traveling'}),
        }

class CompanyProfileForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = [
            'company_name', 'phone_number', 'email', 'company_website', 'profile_picture',
            'about_company', 'industry', 'founding_year', 'company_size', 'headquarters', 'branches',
            'business_registration_number', 'legal_structure', 'core_services', 'social_media_links', 'portfolio'
        ]
        widgets = {
            'company_name': forms.TextInput(attrs={'class': 'form-control'}),
            'phone_number': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'company_website': forms.URLInput(attrs={'class': 'form-control'}),
            'profile_picture': forms.FileInput(attrs={'class': 'form-control'}),
            'about_company': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'industry': forms.TextInput(attrs={'class': 'form-control'}),
            'founding_year': forms.NumberInput(attrs={'class': 'form-control'}),
            'company_size': forms.Select(choices=[
                ('1-10', '1-10 employees'),
                ('11-50', '11-50 employees'),
                ('51-200', '51-200 employees'),
                ('201-500', '201-500 employees'),
                ('501-1000', '501-1000 employees'),
                ('1000+', '1000+ employees'),
            ], attrs={'class': 'form-control'}),
            'headquarters': forms.TextInput(attrs={'class': 'form-control'}),
            'branches': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            # New fields for the company profile section
            'business_registration_number': forms.TextInput(attrs={'class': 'form-control'}),
            'legal_structure': forms.Select(choices=[
                ('LLP', 'LLP'),
                ('Pvt Ltd', 'Private Limited'),
                ('Partnership', 'Partnership'),
                ('Sole Proprietorship', 'Sole Proprietorship'),
                ('Others', 'Others'),
            ], attrs={'class': 'form-control'}),
            'core_services': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'social_media_links': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'portfolio': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }



class BidForm(forms.ModelForm):
    class Meta:
        model = Bid
        exclude = ['project', 'bidder', 'created_at', 'updated_at']
        widgets = {
            'preferred_start_date': forms.DateInput(attrs={'type': 'date'}),
            'project_approach': forms.Textarea(attrs={'rows': 4}),
            'payment_terms': forms.Textarea(attrs={'rows': 3}),
            'team_resources': forms.Textarea(attrs={'rows': 4}),
            'company_profile': forms.Textarea(attrs={'rows': 4}),
            'portfolio_links': forms.Textarea(attrs={'rows': 2}),
            'client_testimonials': forms.Textarea(attrs={'rows': 3}),
        }
        
class FreelanceProjectForm(forms.ModelForm):
    class Meta:
        model = FreelanceProject
        fields = [
            'title', 'category', 'description', 'skills_required', 'duration', 
            'budget', 'payment_terms', 'experience_level', 'location_preference', 
            'language_requirements', 'attachments', 'company_name', 'company_description', 
            'point_of_contact_name', 'point_of_contact_email', 'point_of_contact_phone'
        ]
        widgets = {
            'description': forms.Textarea(attrs={'rows': 5}),
            'skills_required': forms.TextInput(attrs={'placeholder': 'Comma-separated skills'}),
        }

class FreelanceBidForm(forms.ModelForm):
    class Meta:
        model = FreelanceBid
        fields = ['name', 'email', 'phone', 'portfolio_links', 'previous_works', 
                 'expected_pay', 'timeline', 'execution_plan']
        widgets = {
            'portfolio_links': forms.Textarea(attrs={'rows': 4}),
            'previous_works': forms.Textarea(attrs={'rows': 6}),
            'execution_plan': forms.Textarea(attrs={'rows': 8}),
        }

class JobApplicationForm(forms.ModelForm):
    DEGREE_CHOICES = [
    ('bachelor', 'Bachelor\'s'),
    ('master', 'Masters'),
    ('phd', 'PhD'),
    ('diploma', 'Diploma'),
]

    degree = forms.ChoiceField(
        choices=DEGREE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'}),
        required=True
    )

    skills = forms.CharField(
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Enter your skills separated by commas'}),
        help_text="Enter your skills separated by commas"
    )

    class Meta:
        model = JobApplication
        fields = ['phone_number', 'email','degree', 'percentage', 'work_experience', 'resume', 'skills']
