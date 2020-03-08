from django.shortcuts import render
from django.http import HttpResponse
import requests
from .models import Post

# Create your views here.

posts = [
    {
        'author_id':'1',
        'title':'Blog Post 1',
        'content': 'First post content',
        'date_posted':'August 18,2018'
    },
    {
        'author_id':'1',
        'title':'Blog Post 2',
        'content': 'Second post content',
        'date_posted':'August 19,2018'
    }
]
def home(request):
    context={
        "posts":Post.objects.all()
    }
    return render(request,"pages/home.html",context)

def about(request):
    return render(request,"pages/about.html",{"title":"About"})
