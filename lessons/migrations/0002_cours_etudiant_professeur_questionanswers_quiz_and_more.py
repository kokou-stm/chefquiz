# Generated by Django 5.1.3 on 2024-12-25 01:27

import django.db.models.deletion
import lessons.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('lessons', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Cours',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('description', models.CharField(max_length=255)),
                ('file', models.FileField(upload_to=lessons.models.clean_filename)),
            ],
        ),
        migrations.CreateModel(
            name='Etudiant',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero_de_carte', models.CharField(blank=True, max_length=20, null=True, unique=True)),
                ('phone', models.CharField(blank=True, max_length=15, null=True)),
                ('nom', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=100)),
                ('username', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Professeur',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero_de_carte', models.CharField(blank=True, max_length=20, null=True, unique=True)),
                ('phone', models.CharField(blank=True, max_length=15, null=True)),
                ('nom', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=100)),
                ('username', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='QuestionAnswers',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_text', models.CharField(max_length=500)),
                ('required_time', models.FloatField(default=60)),
                ('user_answer', models.CharField(blank=True, max_length=500, null=True)),
                ('great_answer', models.CharField(max_length=500)),
                ('score', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Quiz',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quiz_title', models.CharField(max_length=255)),
                ('quiz_description', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='VerificationCode',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.CharField(default='', max_length=6, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='verification_code', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.DeleteModel(
            name='Courses',
        ),
        migrations.AddField(
            model_name='cours',
            name='etudiant',
            field=models.ManyToManyField(to='lessons.etudiant'),
        ),
        migrations.AddField(
            model_name='cours',
            name='professeur',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='lessons.professeur'),
        ),
        migrations.AddField(
            model_name='quiz',
            name='course',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='quizzes', to='lessons.cours'),
        ),
        migrations.AddField(
            model_name='questionanswers',
            name='quiz',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='questions', to='lessons.quiz'),
        ),
    ]