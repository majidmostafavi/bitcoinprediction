from django.contrib import admin

from .models import Trade, Order, ExchangeInfo

# Register your models here.

admin.site.register(Trade)
admin.site.register(Order)
admin.site.register(ExchangeInfo)
