from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm
from bitstealer.utils import request_binance


# Create your views here.
def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            return redirect('home')
    else:
        form = UserRegisterForm()
    return render(request, 'pages/register.html', {'form': form})


@login_required
def profile(request):

    # current_user=request.user
    # orders=current_user.orders.all()
    # print(orders)
    return render(request,'pages/profile.html')

