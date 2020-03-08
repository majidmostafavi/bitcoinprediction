from django.urls import path
from . import views

urlpatterns = [
    path('', views.home,name='home'),
    path('trade-page', views.show_trade_page,name='getTradePage'),
    path('get-trade-list', views.get_trade_list,name='getTradeList'),
    path('orders', views.orders,name='orders'),
    path('trader', views.trader,name='trader'),
]
