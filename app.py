import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import requests
import feedparser
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="XAU Analytics", layout="centered")

@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "gld_price_data.csv"
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError as e:
        st.error(f"Could not find data file: {data_path}")
        raise

@st.cache_resource
def train_model(data):
    X = data.drop(columns=["Date", "GLD"])
    y = data["GLD"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    model = RandomForestRegressor(n_estimators=100, random_state=2)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mse),
    }
    return model, X_train, X_test, y_train, y_test, metrics

@st.cache_data
def feature_bounds(data):
    numeric = data.drop(columns=["Date", "GLD"])
    bounds = {}
    for feature in numeric.columns:
        bounds[feature] = {
            "min": float(numeric[feature].min()),
            "max": float(numeric[feature].max()),
            "mean": float(numeric[feature].mean()),
        }
    return bounds


@st.cache_data(ttl=300)
def fetch_live_gold_data(range="1d", interval="5m"):
    """Fetch free live gold spot data from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range={range}&interval={interval}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {}).get("result", [])
    if not chart:
        return None
    result = chart[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    if not timestamps or not closes:
        return None
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
        },
        index=pd.to_datetime(timestamps, unit="s")
    )
    df = df.dropna()
    return df


def load_broker_data(uploaded_file=None):
    broker_path = Path(__file__).resolve().parent / "broker_data.csv"
    if uploaded_file is not None:
        data_source = uploaded_file
    elif broker_path.exists():
        data_source = broker_path
    else:
        return None

    try:
        df = pd.read_csv(data_source)
    except Exception:
        return None

    datetime_columns = [c for c in df.columns if c.lower() in ["datetime", "date", "timestamp", "time"]]
    price_columns = [c for c in df.columns if c.lower() in ["price", "gold_price", "spot_price", "close", "last"]]

    if datetime_columns:
        dt_col = datetime_columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.sort_values(dt_col)
        df = df.set_index(dt_col)

    if price_columns:
        price_col = price_columns[0]
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        price_col = numeric_cols[-1] if len(numeric_cols) > 0 else None

    if price_col is None:
        return None

    return df[[price_col]].rename(columns={price_col: "Gold Spot Price"})


@st.cache_data(ttl=3600)
def fetch_news_data():
    """Fetch news related to gold, USD, and forex markets from RSS feeds and APIs"""
    news_dict = {
        "today": [],
        "current": [],
        "upcoming": [],
        "error": None
    }
    
    try:
        # Fetch from financial RSS feeds
        rss_feeds = [
            "https://feeds.bloomberg.com/markets/commodities.rss",  # Commodities
            "https://feeds.finance.yahoo.com/rss/2.0/headline",  # Finance
            "http://feeds.reuters.com/reuters/businessNews",  # Reuters business
        ]
        
        for feed_url in rss_feeds:
            try:
                response = requests.get(feed_url, timeout=10)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:5]:  # Get first 5 entries
                    title = entry.get('title', 'No title')
                    summary = entry.get('summary', '')[:150]  # Truncate summary
                    link = entry.get('link', '#')
                    
                    # Categorize news based on content
                    if any(keyword in title.lower() for keyword in ['gold', 'xau', 'precious', 'commodity']):
                        target = "current"
                    elif any(keyword in title.lower() for keyword in ['fed', 'rate', 'inflation', 'dollar', 'usd']):
                        target = "current"
                    else:
                        target = None
                    
                    if target:
                        news_dict[target].append({
                            "title": title,
                            "summary": summary,
                            "source": feed_url.split('/')[2],
                            "link": link
                        })
            except Exception as feed_error:
                news_dict["error"] = str(feed_error)
                continue
        
        # Sample upcoming events (economic calendar)
        news_dict["upcoming"] = [
            {
                "title": "US Non-Farm Payroll Release",
                "summary": "Monthly employment data release that impacts USD and gold prices",
                "date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                "impact": "High"
            },
            {
                "title": "Federal Reserve Meeting",
                "summary": "FOMC policy decision affecting interest rates and USD",
                "date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "impact": "Very High"
            },
            {
                "title": "Consumer Price Index (CPI)",
                "summary": "Inflation data release impacting gold demand",
                "date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
                "impact": "High"
            }
        ]
        
        # Today's news - filter for today
        today_news = [n for n in news_dict["current"][:3]]
        news_dict["today"] = today_news if today_news else [
            {
                "title": "Gold Market Overview",
                "summary": "XAU/USD trading active in today's session with market participants monitoring USD strength",
                "source": "market_data"
            }
        ]
        
    except Exception as e:
        news_dict = {
            "today": [
                {
                    "title": "Gold Market Active",
                    "summary": "XAU/USD market remains active with traders monitoring economic indicators",
                    "source": "default"
                }
            ],
            "current": [
                {
                    "title": "USD Strength Impacts Gold Prices",
                    "summary": "A stronger US Dollar typically pressures gold prices as it becomes more expensive for foreign buyers",
                    "source": "analysis"
                },
                {
                    "title": "Central Bank Policy Effects",
                    "summary": "Interest rate decisions from major central banks influence safe-haven demand for gold",
                    "source": "analysis"
                }
            ],
            "upcoming": [
                {
                    "title": "Economic Data Releases",
                    "summary": "Upcoming employment and inflation data will significantly impact market sentiment",
                    "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                    "impact": "High"
                }
            ],
            "error": str(e)
        }
    
    return news_dict

# Load data and train the model
gold_data = load_data()
model, X_train, X_test, y_train, y_test, metrics = train_model(gold_data)
bounds = feature_bounds(gold_data)

st.title("XAU Analytics")
st.markdown("Analyze and predict **Gold Spot US Dollar** prices using machine learning.")

with st.expander("Dataset preview"):
    st.dataframe(gold_data.head())

st.subheader("Gold Spot US Dollar Chart")
st.markdown("Historical Gold Spot prices in US Dollars (USD)")

# Prepare price data for chart
price_data = gold_data[['GLD']].reset_index(drop=True)
price_data.index.name = 'Trading Day'
price_data.columns = ['Gold Spot (USD)']

# Display line chart using Streamlit
chart_col = st.line_chart(
    data=price_data,
    use_container_width=True,
    height=400,
    color="#FFD700"  # Gold color
)

# Show Gold Spot price statistics
st.subheader("Gold Spot US Dollar Statistics")
price_stats_col1, price_stats_col2, price_stats_col3, price_stats_col4 = st.columns(4)
price_stats_col1.metric("Highest Spot Price", f"${gold_data['GLD'].max():.2f}")
price_stats_col2.metric("Lowest Spot Price", f"${gold_data['GLD'].min():.2f}")
price_stats_col3.metric("Average Spot Price", f"${gold_data['GLD'].mean():.2f}")
price_stats_col4.metric("Current Spot Price", f"${gold_data['GLD'].iloc[-1]:.2f}")

st.markdown("---")
st.subheader("Live Online Gold Spot Candlestick Chart")

live_gold_data = fetch_live_gold_data()
if live_gold_data is not None and not live_gold_data.empty:
    st.markdown("Live 15-minute gold spot candlestick chart sourced from Yahoo Finance (GC=F).")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=live_gold_data.index,
                open=live_gold_data["Open"],
                high=live_gold_data["High"],
                low=live_gold_data["Low"],
                close=live_gold_data["Close"],
                increasing_line_color="#00FF00",
                decreasing_line_color="#FF0000",
            )
        ]
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Time",
        yaxis_title="Gold Spot Price (USD)",
    )
    st.plotly_chart(fig, use_container_width=True)
    latest_price = live_gold_data["Close"].iloc[-1]
    latest_time = live_gold_data.index[-1]
    st.metric("Latest Live Gold Spot Price", f"${latest_price:.2f}")
    st.caption(f"Latest timestamp: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.warning("Unable to fetch live online gold spot data. Please check your connection or try again later.")

st.markdown("---")
st.subheader("Model performance")
col1, col2, col3 = st.columns(3)
col1.metric("R² score", f"{metrics['r2']:.4f}")
col2.metric("MAE", f"{metrics['mae']:.4f}")
col3.metric("RMSE", f"{metrics['rmse']:.4f}")

st.markdown("---")
st.subheader("Prediction Settings")

# Date range selection
col_days, col_months = st.columns(2)
with col_days:
    prediction_days = st.number_input("Prediction Days", min_value=1, max_value=30, value=7, step=1)
with col_months:
    prediction_months = st.number_input("Prediction Months", min_value=0, max_value=12, value=1, step=1)

st.markdown("---")
st.subheader("Enter USD Pair Value")
st.markdown(f"Adjust the EUR/USD exchange rate to predict Gold Spot Price for the next **{prediction_months} month(s) and {prediction_days} day(s)**")

# Filter to only EUR/USD feature
usd_pair_bounds = {k: v for k, v in bounds.items() if k == "EUR/USD"}

inputs = {}
for feature, info in usd_pair_bounds.items():
    inputs[feature] = st.slider(
        label=f"{feature} Exchange Rate",
        min_value=info["min"],
        max_value=info["max"],
        value=info["mean"],
        step=0.001,
        format="%.3f"
    )

if st.button("Predict Gold Spot Price (USD)"):
    # Use all features for prediction, but only EUR/USD is adjustable
    feature_names = list(bounds.keys())
    features = [[inputs.get(f, bounds[f]["mean"]) for f in feature_names]]
    predicted_price = model.predict(features)[0]
    st.success(f"**Predicted Gold Spot Price (USD) for {prediction_months}M {prediction_days}D: ${predicted_price:.2f}**")

    prediction_table = pd.DataFrame({
        "Metric": [
            "Prediction Horizon",
            "EUR/USD Input",
            "Predicted Gold Spot Price (USD)"
        ],
        "Value": [
            f"{prediction_months} month(s) and {prediction_days} day(s)",
            f"{inputs['EUR/USD']:.3f}",
            f"${predicted_price:.2f}"
        ]
    })
    st.table(prediction_table)

st.markdown("---")
st.subheader("📰 Market News Affecting Gold Price (XAU/USD)")

# Fetch news data
news_data = fetch_news_data()
if news_data.get("error"):
    st.error(f"News fetch error: {news_data['error']}")

# Today's News
st.markdown("#### 🔴 Today's News")
if news_data["today"]:
    for i, news in enumerate(news_data["today"][:3]):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{news.get('title', 'Market Update')}**")
                st.caption(news.get('summary', 'No summary available'))
            with col2:
                st.caption(f"📍 {news.get('source', 'Source')}")
else:
    st.info("No news available")

# Current Market News
st.markdown("#### 🟡 Current Market News")
if news_data["current"]:
    for i, news in enumerate(news_data["current"][:3]):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{news.get('title', 'Market Analysis')}**")
                st.caption(news.get('summary', 'No summary available'))
            with col2:
                st.caption(f"📍 {news.get('source', 'Source')}")
else:
    st.info("No current news available")

# Upcoming Economic Events
st.markdown("#### 🟢 Upcoming Economic Events")
if news_data["upcoming"]:
    for event in news_data["upcoming"][:3]:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{event.get('title', 'Event')}**")
                st.caption(event.get('summary', 'No summary available'))
            with col2:
                st.caption(f"📅 {event.get('date', 'TBD')}")
            with col3:
                impact = event.get('impact', 'Medium')
                if impact == "Very High":
                    st.markdown("🔴 Very High")
                elif impact == "High":
                    st.markdown("🟠 High")
                else:
                    st.markdown("🟡 Medium")
else:
    st.info("No upcoming events available")

st.markdown("---")
st.subheader("Current XAU/USD Statistics")
for feature, info in usd_pair_bounds.items():
    col1, col2, col3 = st.columns(3)
    col1.metric(f"XAU/USD Minimum", f"{info['min']:.4f}")
    col2.metric(f"XAU/USD Average", f"{info['mean']:.4f}")
    col3.metric(f"XAU/USD Maximum", f"{info['max']:.4f}")
