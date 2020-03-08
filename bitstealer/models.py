from django.db import models

from django.contrib.auth.models import User


# Create your models here.
class Trade(models.Model):
    symbol = models.CharField(max_length=20)
    price = models.DecimalField(max_digits=14, decimal_places=8)
    qty = models.DecimalField(max_digits=14, decimal_places=8)
    time = models.DateTimeField()
    is_buyer_maker = models.BooleanField()

    class Meta:
        unique_together = ('symbol', 'price', 'qty', 'time')


class ExchangeInfo(models.Model):
    symbol = models.CharField(max_length=20)
    status = models.CharField(max_length=255)
    base_asset = models.CharField(max_length=255)
    quote_asset = models.CharField(max_length=255)
    tick_size = models.FloatField()
    sts = models.FloatField()


class Balance(models.Model):
    user = models.ForeignKey(User,related_name='balances', on_delete=models.CASCADE)
    asset = models.CharField(max_length=64),
    free = models.DecimalField(max_digits=14, decimal_places=8),
    locked = models.DecimalField(max_digits=14, decimal_places=8)


class Order(models.Model):
    user = models.ForeignKey(User, related_name='orders', on_delete=models.CASCADE)
    symbol = models.CharField(max_length=20)
    order_id = models.CharField(max_length=255)
    status = models.CharField(max_length=255, blank=True, default=None)
    price = models.DecimalField(max_digits=14, decimal_places=8)
    qty = models.DecimalField(max_digits=14, decimal_places=8)
    time = models.DateTimeField()
    side = models.CharField(max_length=20)
    low = models.DecimalField(max_digits=14, decimal_places=8)
    high = models.DecimalField(max_digits=14, decimal_places=8)

    def __str__(self):
        return f"{self.symbol} {self.side} {self.price}"


class HourlyCandle(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateTimeField()
    open = models.DecimalField(max_digits=36, decimal_places=8)
    close = models.DecimalField(max_digits=36, decimal_places=8)
    low = models.DecimalField(max_digits=36, decimal_places=8)
    high = models.DecimalField(max_digits=36, decimal_places=8)
    volume = models.DecimalField(max_digits=36, decimal_places=8)
    coin_marketcap = models.BigIntegerField(null=True)
    marketcap = models.BigIntegerField(null=True)


class DailyCandle(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateTimeField()
    open = models.DecimalField(max_digits=36, decimal_places=8)
    close = models.DecimalField(max_digits=36, decimal_places=8)
    low = models.DecimalField(max_digits=36, decimal_places=8)
    high = models.DecimalField(max_digits=36, decimal_places=8)
    volume = models.DecimalField(max_digits=36, decimal_places=8, null=True)
    coin_marketcap = models.BigIntegerField(null=True)
    marketcap = models.BigIntegerField(null=True)


class MinCandle(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateTimeField()
    open = models.DecimalField(max_digits=36, decimal_places=8)
    close = models.DecimalField(max_digits=36, decimal_places=8)
    low = models.DecimalField(max_digits=36, decimal_places=8)
    high = models.DecimalField(max_digits=36, decimal_places=8)
    volume = models.DecimalField(max_digits=36, decimal_places=8, null=True)
    coin_marketcap = models.BigIntegerField(null=True)
    marketcap = models.BigIntegerField(null=True)


class Min3Candle(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateTimeField()
    open = models.DecimalField(max_digits=36, decimal_places=8)
    close = models.DecimalField(max_digits=36, decimal_places=8)
    low = models.DecimalField(max_digits=36, decimal_places=8)
    high = models.DecimalField(max_digits=36, decimal_places=8)
    volume = models.DecimalField(max_digits=36, decimal_places=8, null=True)
    coin_marketcap = models.BigIntegerField(null=True)
    marketcap = models.BigIntegerField(null=True)


class Min5Candle(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateTimeField()
    open = models.DecimalField(max_digits=36, decimal_places=8)
    close = models.DecimalField(max_digits=36, decimal_places=8)
    low = models.DecimalField(max_digits=36, decimal_places=8)
    high = models.DecimalField(max_digits=36, decimal_places=8)
    volume = models.DecimalField(max_digits=36, decimal_places=8, null=True)
    coin_marketcap = models.BigIntegerField(null=True)
    marketcap = models.BigIntegerField(null=True)


class Gold(models.Model):
    date = models.DateTimeField()
    price = models.DecimalField(max_digits=14, decimal_places=8)


class Oil(models.Model):
    date = models.DateTimeField()
    price = models.DecimalField(max_digits=14, decimal_places=8)


class BitmexTrade(models.Model):
    symbol = models.CharField(max_length=20)
    price = models.DecimalField(max_digits=14, decimal_places=8)
    qty = models.DecimalField(max_digits=14, decimal_places=8)
    time = models.DateTimeField()
    is_buyer_maker = models.BooleanField()


class BitmxMinCandle(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateTimeField()
    open = models.DecimalField(max_digits=36, decimal_places=8)
    close = models.DecimalField(max_digits=36, decimal_places=8)
    low = models.DecimalField(max_digits=36, decimal_places=8)
    high = models.DecimalField(max_digits=36, decimal_places=8)
    volume = models.DecimalField(max_digits=36, decimal_places=8)
    coin_marketcap = models.BigIntegerField(null=True)
    marketcap = models.BigIntegerField(null=True)