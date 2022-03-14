from optparse import Option
from posixpath import split
from urllib import request
import streamlit as st 
import pandas as pd 
import os 
import numpy as np 
import hvplot.pandas
import math
import time
from pathlib import Path
from datetime import date
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import Counter
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Hour
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib
import requests

# Basic functionalities
import json


# datetime manipulation
import datetime as dt
from time import sleep
from datetime import timedelta

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Deep learning model persistence
from tensorflow.keras.models import model_from_json

# Graphing

import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots



header = st.container() 
dataset1 = st.container()
dataset2 = st.container() 
dataset3 = st.container()
dataset4 = st.container()
dataset5 = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Social Sentiment Meme Stock Picker and Technical Analyser")
    st.write("Developed by members of Group-1 for Project 2 of the Monash University FinTech Bootcamp 2021-2022")


def create_alpaca_connection():    
    api = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key,
    base_url = 'https://paper-api.alpaca.markets',
    api_version = "v2"
    )
    return api
def get_sentiment_df():
    # Create headers variable containing API Key 
    headers = {
        'Authorization' : f'Token {ss_key}',
        'Accept': 'application/json',
    }
    # Get data for daily stock sentiment from API
    sentiment_response = requests.get('https://socialsentiment.io/api/v1/stocks/sentiment/daily/', headers=headers)
    
    # Create dict from sentiment_response['results'] and then a data frame from this dict 
    sentiment_dict = sentiment_response.json()['results']
    sentiment_df = pd.DataFrame.from_dict(sentiment_dict)
    
    # Determine how many lines and then pages are in the sentiment response
    line_count = sentiment_response.json()['count']
    page_count = int(line_count / 50) + (line_count % 50 > 0)
    page=2
    
    # Loop through each page and gather the sentiment data from the API
    while page <= page_count:
        # Loop through all pages available from API and construct dataframe for sentiment data
        sentiment_url = "https://socialsentiment.io/api/v1/stocks/sentiment/daily/?page=%s"%page
        sentiment_response = requests.get(sentiment_url, headers=headers)
        sentiment_dict = sentiment_response.json()['results']
        sentiment_df_loop = pd.json_normalize(sentiment_dict)
        sentiment_df = pd.concat([sentiment_df, sentiment_df_loop], axis=0)
        page += 1
        time.sleep(1)

    # Reset the index on the sentiment df and drop the old index
    sentiment_df.reset_index(inplace=True, drop=True)
    
    # Output the sentiment data for today to a csv file
    path = (f'../Resources/sentiment_{date.today()}.csv')
    sentiment_df.to_csv(path)
    
    return sentiment_df

def get_sentiment_trending_df():
    # Create headers variable containing API Key 
    headers = {
        'Authorization' : f'Token {ss_key}',
        'Accept': 'application/json',
    }
    # Get data for trending stock sentiment from API
    trending_response = requests.get('https://socialsentiment.io/api/v1/stocks/trending/daily/', headers=headers)
    
    # Create dict from trending_response and then a data frame from this dict 
    trending_dict = trending_response.json()
    trending_df = pd.DataFrame.from_dict(trending_dict)
    
    # Output the sentiment data for today to a csv file
    path = (f'../Resources/sentiment_trending_{date.today()}.csv')
    trending_df.to_csv(path)
    
    return trending_df

def create_technical_analysis_df():
    # Create the Alpaca API object, specifying use of the paper trading account:
    api = create_alpaca_connection()
    
    # Set the list of tickers to the top class stock
    tickers = top_class_stocks.index
    # Set get data from API to DataFrame
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    previous_days = today - timedelta(days=365)
    beg_date = previous_days
    end_date = yesterday
    timeframe='1Day'
    start =  pd.Timestamp(f'{beg_date} 09:30:00-0400', tz='America/New_York').replace(hour=9, minute=30, second=0).astimezone('GMT').isoformat()[:-6]+'Z'
    end =  pd.Timestamp(f'{end_date} 16:00:00-0400', tz='America/New_York').replace(hour=16, minute=0, second=0).astimezone('GMT').isoformat()[:-6]+'Z'
    portfolio_df = api.get_bars(tickers, timeframe, start=start, end=end).df
   
    
    # Calculate the 5,15 period high and low rolling SMAs and add each to the portfolio dataframe
    portfolio_df['Lowest_5D'] = portfolio_df.groupby('symbol')['low'].transform(lambda x: x.rolling(window = 12).min())
    portfolio_df['High_5D'] = portfolio_df.groupby('symbol')['high'].transform(lambda x: x.rolling(window = 12).max())
    portfolio_df['Lowest_15D'] = portfolio_df.groupby('symbol')['low'].transform(lambda x: x.rolling(window = 26).min())
    portfolio_df['High_15D'] = portfolio_df.groupby('symbol')['high'].transform(lambda x: x.rolling(window = 26).max())
    
    # Calculate Stochastic Indicators and add each to the portfolio dataframe
    portfolio_df['Stochastic_5'] = ((portfolio_df['close'] - portfolio_df['Lowest_5D'])/(portfolio_df['High_5D'] - portfolio_df['Lowest_5D']))*100
    portfolio_df['Stochastic_15'] = ((portfolio_df['close'] - portfolio_df['Lowest_15D'])/(portfolio_df['High_15D'] - portfolio_df['Lowest_15D']))*100
    portfolio_df['Stochastic_%D_5'] = portfolio_df['Stochastic_5'].rolling(window = 5).mean()
    portfolio_df['Stochastic_%D_15'] = portfolio_df['Stochastic_5'].rolling(window = 15).mean()
    portfolio_df['Stochastic_Ratio'] = portfolio_df['Stochastic_%D_5']/portfolio_df['Stochastic_%D_15']

    # Calculate the TP,sma, mad, cci, previous_close and TR then add each to the portfolio dataframe
    portfolio_df['TP'] = (portfolio_df['high'] + portfolio_df['low'] + portfolio_df['close']) / 3
    portfolio_df['sma'] = portfolio_df.groupby('symbol')['TP'].transform(lambda x: x.rolling(window=26).mean())
    portfolio_df['mad'] = portfolio_df['TP'].rolling(window=26).apply(lambda x: pd.Series(x).mad()) #Calculates Mean Absolute Deviation of 'TP' using a 21 period and returns a pandas series
    portfolio_df['CCI'] = (portfolio_df['TP'] - portfolio_df['sma']) / (0.015 * portfolio_df['mad'])
    portfolio_df['prev_close'] = portfolio_df.groupby('symbol')['close'].shift(1)
    portfolio_df['Actual Returns'] = portfolio_df.groupby('symbol')['close'].pct_change()
    portfolio_df['TR'] = np.maximum((portfolio_df['high'] - portfolio_df['low']),
                                np.maximum(abs(portfolio_df['high'] - portfolio_df['prev_close']), 
                                abs(portfolio_df['prev_close'] - portfolio_df['low'])))

    # Calculate the ATR12 and 26 and add each to the portfolio dataframe
    for i in portfolio_df['symbol'].unique():
        ATR_12 = []
        ATR_26 = []
        TR_data = portfolio_df[portfolio_df.symbol == i].copy()
        portfolio_df.loc[portfolio_df.symbol==i,'ATR_12'] = (TR_data['TR']).rolling(window=12).mean()
        portfolio_df.loc[portfolio_df.symbol==i,'ATR_26'] = (TR_data['TR']).rolling(window=26).mean()
    portfolio_df['ATR_Ratio'] = portfolio_df['ATR_12'] / portfolio_df['ATR_26']
        
    # Reset then set the index on the dataframe then output to csv file for later use
    portfolio_df.reset_index(inplace=True)
    portfolio_df.set_index(['symbol', 'timestamp'], inplace=True)
    path = (f'../Resources/portfolio_indicators_{date.today()}.csv')
    portfolio_df.to_csv(path)
    
    return portfolio_df

def create_returns_df():
    # Create the Alpaca API object, specifying use of the paper trading account:
    api = create_alpaca_connection()
    
    # Set the list of tickers to the top class stock
    tickers = top_class_stocks.index
    # Set get data from API to DataFrame
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    previous_days = today - timedelta(days=30)
    beg_date = previous_days
    end_date = yesterday
    timeframe='1Day'
    start =  pd.Timestamp(f'{beg_date} 09:30:00-0400', tz='America/New_York').replace(hour=9, minute=30, second=0).astimezone('GMT').isoformat()[:-6]+'Z'
    end =  pd.Timestamp(f'{end_date} 16:00:00-0400', tz='America/New_York').replace(hour=16, minute=0, second=0).astimezone('GMT').isoformat()[:-6]+'Z'
    portfolio_df = api.get_bars(tickers, timeframe, start=start, end=end).df

    # Pull prices from the ALPACA API
    data = api.get_bars(tickers, timeframe, start=start, end=end).df
    
    close_df = pd.DataFrame(index=data.index)

    for ticker in tickers:
        vector = data.loc[data["symbol"] == ticker].close
        close_df[ticker] = vector

    close_df.dropna(axis=1, how='all', inplace=True)

    # Use Pandas' forward fill function to fill missing values (be sure to set inplace=True)
    close_df.ffill(inplace=True)
    
    # Define a variable to set prediction period
    forecast = 1

    # Compute the pct_change for 1 min 
    returns = close_df.pct_change(periods=forecast)

    # Shift the returns to convert them to forward returns
    returns = returns.shift(-(forecast))
    returns.dropna(inplace=True)
    path = (f'../Resources/returns_{date.today()}.csv')
    returns.to_csv(path)
    
    return returns

with dataset1:
    load_dotenv()
    ss_key = os.getenv("SS_API_KEY")
    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

    st.subheader('Welcome to the Memestock picker. This app uses a special algorithim which analyses sentiment across social media and generates a portfolio that is predicted to perform well.')
    
    col1, col2, = st.columns(2)
    
with col1:
    st.header("Latest sentiment")
    sentiment_df = get_sentiment_df()
    sentiment_df.head()
    st.dataframe(sentiment_df)

with col2:
    st.header("Trending sentiment")
    sentiment_trending_df = get_sentiment_trending_df()
    sentiment_trending_df.head()
    st.dataframe(sentiment_trending_df)

with dataset2:

    col3, col4, = st.columns([3,1])

with col3:
    st.header("Trending stock sentiment scores")
    sentiment_trending_plot_df = sentiment_trending_df.set_index("stock")
    sentiment_trending_plot_df["score"].plot(
    kind='bar',
    x='stock',
    y='score', 
    title = "Trending Stock Sentiment Scores",
    figsize=(20,10))
    st.bar_chart(sentiment_trending_plot_df)

with col4:
    st.subheader("Top 3 trending stocks")
    stock_name = pd.DataFrame(sentiment_trending_df['stock'])
    st.dataframe(stock_name.head(3))

with dataset3:
    x = sentiment_trending_df.set_index("stock")
    
    x_scaled = StandardScaler().fit_transform(x)
    st.subheader("Principal component analysis")
    pca = PCA(n_components=3)
    sentiment_pca = pca.fit_transform(x_scaled)
    pcs_df = pd.DataFrame(
    data=sentiment_pca, columns=["PC 1","PC 2","PC 3"], index=x.index)
    st.dataframe(pcs_df.head(10))

    # Determine the optimal value for k using k= 1-11
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pcs_df)
    inertia.append(km.inertia_)

st.header("Elbow curve")
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
st.line_chart(df_elbow)

# Initialize the K-Means model
model = KMeans(n_clusters=5, random_state=0)

# Fit the model
model.fit(pcs_df)

# Predict clusters
predictions = model.predict(pcs_df)

# Create a new DataFrame including predicted clusters and trending sentiment features
st.header("Predicted clusters")
clustered_df = pd.DataFrame({
    "score": x.score,
    "positive_score": x.positive_score,
    "negative_score": x.negative_score,
    "activity": x.activity,
    "activity_avg_7_days": x.activity_avg_7_days,
    "activity_avg_14_days": x.activity_avg_14_days,
    "activity_avg_30_days": x.activity_avg_30_days,
    "score_avg_7_days": x.score_avg_7_days,
    "score_avg_14_days": x.score_avg_14_days,
    "score_avg_30_days": x.score_avg_30_days,
    "PC 1": pcs_df['PC 1'],
    "PC 2": pcs_df['PC 2'],
    "PC 3": pcs_df['PC 3'],
    "Class": model.labels_,
    },
    index=x.index
)
st.dataframe(clustered_df.head())

st.header("Enhanced visualisation of predicted clusters")
# Plotting the 3D-Scatter with x="PC 1", y="PC 2" and z="PC 3"
fig = px.scatter_3d(
    clustered_df,
    x="PC 1",
    y="PC 2",
    z="PC 3",
    hover_name='score',
    hover_data= ['activity'],
    height=600,
    color="Class"
)
st.plotly_chart(fig)

st.header("Top class stocks")
st.subheader("Top stocks based on score")
    
# Sort the clustered_df by score and order the DataFrame selecting the top 10
top_stocks = clustered_df.sort_values("score", ascending=False).head(10)                   
# Select the class most represented in the top 10 stocks
top_class = top_stocks["Class"].mode()
top_class_stocks = top_stocks.loc[top_stocks["Class"] == top_class[0]]
# View the top class stocks
st.dataframe(top_class_stocks)

with dataset5:
    st.title("Technical Analysis")
    st.header("Trading data")
    portfolio_df = create_technical_analysis_df()
    st.dataframe(portfolio_df.head(5))
    
    technicals = portfolio_df[["Stochastic_Ratio","CCI","ATR_Ratio","close"]]
    st.header("Winning stock pick")
    st.dataframe(technicals.tail())


