# Generated by Django 5.1.4 on 2025-01-21 12:08

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0015_idea_email'),
    ]

    operations = [
        migrations.CreateModel(
            name='LikeDislike',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('like', models.BooleanField(default=False)),
                ('dislike', models.BooleanField(default=False)),
                ('idea', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.idea')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'unique_together': {('user', 'idea')},
            },
        ),
    ]
