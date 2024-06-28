import os
import requests
from flask import Flask, request, render_template
from flask_migrate import Migrate
from flask_minify import Minify
from sys import exit
from textblob import TextBlob
import tweepy
import numpy as np
from datetime import datetime, timedelta
import json

from apps.config import config_dict
from apps import create_app, db

DEBUG = (os.getenv('DEBUG', 'False') == 'True')
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    app_config = config_dict[get_config_mode.capitalize()]
except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)
    
if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG)             )
    app.logger.info('Page Compression = ' + 'FALSE' if DEBUG else 'TRUE' )
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT )

# Twitter API setup
auth = tweepy.OAuthHandler(os.getenv('TWITTER_API_KEY'), os.getenv('TWITTER_API_SECRET_KEY'))
auth.set_access_token(os.getenv('TWITTER_ACCESS_TOKEN'), os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
api = tweepy.API(auth)

# Bot Sentinel API setup
BOT_SENTINEL_URL = "https://botsentinel.com/api/v1/accounts/check"

def get_tweets(query):
    tweets = api.search_tweets(q=query, count=100, result_type='recent', lang='en')
    return tweets

def calculate_engagement(tweet):
    user_followers = tweet.user.followers_count
    if user_followers > 0:
        engagement = (tweet.favorite_count + tweet.retweet_count + tweet.reply_count) / user_followers
        return engagement
    return 0

def analyze_sentiment(tweet_text):
    analysis = TextBlob(tweet_text)
    return analysis.sentiment.polarity

def detect_bots(usernames):
    response = requests.post(BOT_SENTINEL_URL, json={'accounts': usernames})
    return response.json()

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return render_template('index.html', error='Please enter a search query')

    tweets = get_tweets(query)
    tweet_data = []

    usernames = [tweet.user.screen_name for tweet in tweets]
    bot_check_results = detect_bots(usernames)

    for tweet in tweets:
        engagement = calculate_engagement(tweet)
        sentiment = analyze_sentiment(tweet.text)
        is_bot = bot_check_results.get(tweet.user.screen_name, {}).get('is_bot', False)
        tweet_data.append({
            'text': tweet.text,
            'user': tweet.user.screen_name,
            'engagement': engagement,
            'sentiment': sentiment,
            'is_bot': is_bot
        })

    return render_template('results.html', tweets=tweet_data)

@app.route('/predict', methods=['POST'])
def predict():
    query = request.form.get('query')
    if not query:
        return render_template('index.html', error='Please enter a search query')

    tweets = get_tweets(query)
    engagements = [calculate_engagement(tweet) for tweet in tweets if not detect_bots([tweet.user.screen_name])[0]['is_bot']]
    
    # Assuming Llama3 is set up with an API endpoint
    llama3_url = "https://llama3.api/predict"
    historical_data = get_historical_data(query)
    payload = {
        'engagements': engagements,
        'historical_data': historical_data
    }
    prediction = requests.post(llama3_url, json=payload).json()

    return render_template('prediction.html', prediction=prediction)

def get_historical_data(crypto_symbol):
    # Fetch historical price data from CoinMarketCap or CoinGecko
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart"
    params = {'vs_currency': 'usd', 'days': '30'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['prices']
    return []

if __name__ == "__main__":
    app.run()
