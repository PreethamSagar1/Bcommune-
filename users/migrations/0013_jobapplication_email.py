# Generated by Django 5.1.4 on 2025-01-19 18:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0012_alter_job_required_qualification'),
    ]

    operations = [
        migrations.AddField(
            model_name='jobapplication',
            name='email',
            field=models.EmailField(blank=True, max_length=254, null=True),
        ),
    ]
