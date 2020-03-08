from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Account(models.Model):
    user = models.OneToOneField(User, on_delete=None)
    api_key = models.CharField(max_length=255)
    api_secret = models.CharField(max_length=255)
    maker_commission = models.CharField(max_length=255, blank=True, null=True)
    taker_commission = models.FloatField(null=True, blank=True, default=None)
    buyer_commission = models.FloatField(null=True, blank=True, default=None)
    seller_commission = models.FloatField(null=True, blank=True, default=None)
    can_trade = models.BooleanField(default=True)
    can_withdraw = models.BooleanField(default=True)
    can_deposit = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


