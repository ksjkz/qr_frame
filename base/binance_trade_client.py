import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import yaml
import requests as req
from binance.client import Client


def get_client():
    with open("config/binance_api.yaml", "r", encoding="utf-8") as file:
          config = yaml.safe_load(file)
    key = config['binance_api']['api_key']
    secret = config['binance_api']['api_secret']
    return Client(key, secret)

