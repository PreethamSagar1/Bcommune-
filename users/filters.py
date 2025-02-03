# filters.py
import django_filters
from django.db.models import Q
from .models import JobApplication, SavedApplication

class JobApplicationFilter(django_filters.FilterSet):
    # Match Score Filter (now using the related SavedApplication model)
    match_score_min = django_filters.NumberFilter(method='filter_match_score', label='Minimum Match Score')
    match_score_max = django_filters.NumberFilter(method='filter_match_score', label='Maximum Match Score')
    
    # Education Filters
    degree = django_filters.ChoiceFilter(
    choices=[
    ('bachelor', 'Bachelors'),
    ('master', 'Masters'),
    ('phd', 'PhD'),
    ('diploma', 'Diploma'),
]
)
    percentage_min = django_filters.NumberFilter(field_name='percentage', lookup_expr='gte')
    percentage_max = django_filters.NumberFilter(field_name='percentage', lookup_expr='lte')
    
    # Experience Filter
    experience = django_filters.ChoiceFilter(
        field_name='work_experience',
        choices=[
            ('0-2', '0-2 years'),
            ('3-5', '3-5 years'),
            ('5+', '5+ years'),
        ],
        method='filter_experience'
    )
    
    # Skills Filter
    skills = django_filters.CharFilter(method='filter_skills', label='Required Skills')
    
    # Application Date Filter
    applied_date = django_filters.DateFromToRangeFilter(field_name='applied_at')
    
    # Status Filter (now using SavedApplication's category)
    category = django_filters.ChoiceFilter(
        method='filter_category',
        choices=[
            ('shortlisted', 'Shortlisted'),
            ('reviewed', 'Reviewed'),
            ('rejected', 'Rejected'),
        ]
    )

    class Meta:
        model = JobApplication
        fields = ['degree', 'category']
    
    def filter_match_score(self, queryset, name, value):
        if name == 'match_score_min':
            return queryset.filter(savedapplication__overall_match_score__gte=value)
        elif name == 'match_score_max':
            return queryset.filter(savedapplication__overall_match_score__lte=value)
        return queryset
    
    def filter_experience(self, queryset, name, value):
        if value == '0-2':
            return queryset.filter(work_experience__lte=2)
        elif value == '3-5':
            return queryset.filter(work_experience__gte=3, work_experience__lte=5)
        elif value == '5+':
            return queryset.filter(work_experience__gt=5)
        return queryset
    
    def filter_skills(self, queryset, name, value):
        if not value:
            return queryset
        
        skills = [skill.strip().lower() for skill in value.split(',')]
        q_objects = Q()
        
        for skill in skills:
            q_objects |= Q(skills__icontains=skill)
        
        return queryset.filter(q_objects)
    
    def filter_category(self, queryset, name, value):
        if value:
            return queryset.filter(savedapplication__category__iexact=value)
        return queryset
