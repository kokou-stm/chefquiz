from django.conf.urls.static import static
from django.conf import settings
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('upload_cours/', views.upload_cours, name='upload_cours'),
    path('quiz/<int:course_id>', views.quiz, name='quiz_creator'),
    path('quiz/', views.quiz, name='quiz'),
    path('code/', views.code, name='code'),
    path('cours/', views.listecours, name='lescours'),
    path('faquestion/', views.faquestion, name='faquestion'),
    path('ask_ai/<int:course_id>', views.ask_ia, name='ask_ai'),
    path('login/', views.connection, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.deconnexion, name='logout'),
    path('forgotpassword/', views.forgotpassword, name='forgotpassword'),
    path('updatepassword/<str:token>/<str:uid>/', views.updatepassword, name='updatepassword'),  
    path('course/<int:course_id>/', views.course_details, name='course_details'),
    path('quiz_details/<int:quiz_id>/', views.quiz_details, name='quiz_details'),
    path('quiz_details/<int:quiz_id>/', views.quiz_details, name='display_quiz'),
    path('quiz_score/<int:quiz_id>/', views.quiz_score, name='quiz_score'),
    path('contact/', views.contact, name='contact'),
    path('boat/', views.boat, name='boat'),
      
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

      