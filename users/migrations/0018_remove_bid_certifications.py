# Generated by Django 5.1.4 on 2025-01-22 06:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0017_remove_bid_proposal_bid_certifications_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='bid',
            name='certifications',
        ),
    ]
