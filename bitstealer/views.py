import json
from django.db import connection

from .utils import *
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from bitstealer.coin import Coin
from .models import *
from django.core.serializers import serialize

# Create your views here.


def show_trade_page(request):
    return render(request,"pages/trade-list.html")


def home(request):
    return render(request,"pages/home.html")


def get_trade_list(request):
    response = None
    return JsonResponse({'data':response})


@login_required
def trader(request):
    traddder(request.user,Coin(base=request.GET['base'].upper(), quote=request.GET['quote'].upper(), qty=request.GET['qty']))
    return JsonResponse({'success':request.GET['base']})


def orders(request):
    return render(request,'pages/orders.html')



# SELECT DISTINCT
# min(price) as low,
# max(firstvalue) as open,
# max(lastvalue) as close,
# max(price) as high,
# date_trunc('minute', "time") AS minute,
# count(*) AS count
# FROM   (select *, first_value(price) over (partition by date_trunc('minute', "time") order by time) as firstvalue,
#                                                                                                        last_value(price) over (partition by date_trunc('minute', "time") order by time) as lastvalue
# from public.bitstealer_recenttradelist
# ) list
# GROUP  BY minute
# Order BY minute