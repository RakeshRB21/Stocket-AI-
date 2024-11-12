import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import google.generativeai as genai
import altair as alt
import numpy as np
import mplfinance as mpf
from datetime import datetime
from streamlit_card import card
import textwrap
import html
import re



#1. Initial Setup
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

#2. API keys
FINNHUB_API_KEY = "cs62oupr01qv8tfqeqhgcs62oupr01qv8tfqeqi0"
NEWS_API_KEY = "773f803ad36b4f298c176361f14741bd"
GEMINI_API_KEY = "AIzaSyBNTOCOxNpXdMxGmrMCihud-o3-c7uYt5A"

#3. Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

#4. Custom Theme and Styling
st.set_page_config(page_title="Stocket Dashboard", page_icon="📈", layout="wide")

#5. Custom CSS for improved aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    .css-1d391kg {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #FF69B4;
        color: white;
        border-radius: 50px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF1493;
    }
    .stSelectbox>div>div {
        background-color: #FFFFFF;
        color: #000000;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #FF69B4;
    }
    .sidebar .block-container {
        padding-top: 2rem;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 1rem;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] a {
        color: #000000;
        text-decoration: none;
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] a:hover {
        background-color: #FF69B4;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] a svg {
        margin-right: 0.5rem;
    }
    .news-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    .news-item {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        flex: 1 1 calc(33.333% - 20px);
    }
    .news-item:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .news-item h3 {
        margin-bottom: 10px;
        color: #FF69B4;
    }
    .news-item p {
        margin-bottom: 10px;
        color: #333333;
    }
    .news-item a {
        color: #FF69B4;
        text-decoration: none;
    }
    .news-item a:hover {
        text-decoration: underline;
    }
    .news-item .news-meta {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .news-item .news-meta span {
        transition: color 0.3s ease, text-shadow 0.3s ease;
    }
    .news-item .news-meta span:hover {
        color: #FF69B4;
        text-shadow: 0 0 5px #FF69B4;
    }
    .heatmap-container {
        margin-top: 20px;
    }
    .comparison-container {
        margin-top: 20px;
    }
    .explanation-container {
        margin-top: 20px;
    }
    .sentiment-explanation {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sentiment-explanation h4 {
        color: #FF69B4;
    }
    .sentiment-explanation p {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

#6. Authentication Function
def check_credentials():
    if "credentials_correct" not in st.session_state:
        st.title('Welcome to Stocket Dashboard')
        st.subheader('Please login to continue')
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin":
                st.session_state["credentials_correct"] = True
                st.rerun()  # Changed from st.experimental_rerun() to st.rerun()
            else:
                st.error("😕 Username or password incorrect")
        return False
    return st.session_state["credentials_correct"]

#7. Data Fetching Functions
@st.cache_data(ttl=300)
def get_live_market_data(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

@st.cache_data(ttl=3600)
def get_historical_data(symbol, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Download data
        df = yf.download(symbol, start=start_date, end=end_date)
        
        # Verify data is not empty
        if df.empty:
            st.error(f"No data available for {symbol}")
            return pd.DataFrame()
            
        # Convert the index to a regular column and rename it to 'Date'
        df = df.copy()  # Create a copy to avoid modifying the original
        df.index.name = 'Date'  # Name the index
        df = df.reset_index()  # Reset index to make Date a regular column
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

#8. News and Sentiment Analysis Functions
@st.cache_data(ttl=3600)
def fetch_news(symbol):
    news_api = NewsApiClient(api_key=NEWS_API_KEY)
    all_articles = news_api.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=10)
    return all_articles['articles']

def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(article['content']) for article in articles]

#9. Technical Analysis Functions
def perform_technical_analysis(data):
    """Calculate technical indicators with better error handling"""
    try:
        # Convert index to datetime if it's not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
        
        # Create indicators DataFrame
        indicators = pd.DataFrame(index=data.index)
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        indicators['Stochastic_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        indicators['Stochastic_D'] = indicators['Stochastic_K'].rolling(window=3).mean()
        
        # Bollinger Bands
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = sma + (std * 2)
        indicators['BB_Middle'] = sma
        indicators['BB_Lower'] = sma - (std * 2)
        
        return indicators
    
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

def plot_technical_indicators(data, indicators, selected_indicators):
    """Create technical analysis charts using Altair"""
    
    # Price Chart
    price_chart = alt.Chart(data.reset_index()).mark_line(color='#333333').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title='Price ($)'),
        tooltip=['Date:T', 'Close:Q']
    ).properties(
        height=300,
        title='Price'
    )
    
    # Indicators Chart
    indicator_df = indicators[selected_indicators].reset_index()
    indicator_df = indicator_df.melt('Date', var_name='Indicator', value_name='Value')
    
    indicator_chart = alt.Chart(indicator_df).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Value:Q', title='Value'),
        color=alt.Color('Indicator:N', legend=alt.Legend(
            orient='top',
            title=None,
            labelFontSize=12
        )),
        tooltip=['Date:T', 'Indicator:N', 'Value:Q']
    ).properties(
        height=200
    )
    
    return alt.vconcat(price_chart, indicator_chart).resolve_scale(
        x='shared'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )

def technical_analysis_page():
    st.title("🧠 AI-Powered Technical Analysis")
    
    # Stock selection with company names
    stock_options = {
        'AAPL': 'Apple (AAPL)',
        'GOOGL': 'Google (GOOGL)',
        'MSFT': 'Microsoft (MSFT)',
        'AMZN': 'Amazon (AMZN)',
        'TSLA': 'Tesla (TSLA)'
    }
    
    stock_symbol = st.selectbox(
        "Select Stock",
        options=list(stock_options.keys()),
        format_func=lambda x: stock_options[x]
    )
    
    # Time period selection
    periods = {
        '1 Month': 30,
        '3 Months': 90,
        '6 Months': 180,
        '1 Year': 365
    }
    
    selected_period = st.select_slider(
        "Select Time Period",
        options=list(periods.keys()),
        value='3 Months'
    )
    
    with st.spinner("Calculating technical indicators..."):
        # Get historical data
        data = get_historical_data(stock_symbol, days=periods[selected_period])
        
        if data is not None and not data.empty:
            # Calculate indicators
            indicators = perform_technical_analysis(data)
            
            if indicators is not None:
                # Available indicators
                indicator_groups = {
                    'Trend': ['MACD', 'MACD_Signal', 'MACD_Histogram'],
                    'Momentum': ['RSI', 'Stochastic_K', 'Stochastic_D'],
                    'Volatility': ['BB_Upper', 'BB_Middle', 'BB_Lower']
                }
                
                # Indicator selection
                st.subheader("Select Indicators")
                selected_group = st.selectbox("Indicator Group", list(indicator_groups.keys()))
                selected_indicators = st.multiselect(
                    "Select Indicators",
                    options=indicator_groups[selected_group],
                    default=indicator_groups[selected_group][:2]
                )
                
                if selected_indicators:
                    # Plot charts
                    chart = plot_technical_indicators(data, indicators, selected_indicators)
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Indicator Statistics")
                    stats = indicators[selected_indicators].describe().round(2)
                    st.dataframe(stats)
                    
                    # AI Analysis
                    st.subheader("AI Analysis")
                    
                    analysis_prompt = f"""
                    Analyze the technical indicators for {stock_options[stock_symbol]}:
                    
                    Current Values:
                    {indicators[selected_indicators].tail(1).to_string()}
                    
                    Recent Trends:
                    {indicators[selected_indicators].tail().to_string()}
                    
                    Provide a brief analysis of:
                    1. Current market position
                    2. Key indicator signals
                    3. Potential trend direction
                    4. Notable patterns or divergences
                    """
                    
                    with st.spinner("Generating analysis..."):
                        try:
                            analysis = model.generate_content(analysis_prompt)
                            st.write(analysis.text)
                        except Exception as e:
                            st.error(f"Error generating analysis: {str(e)}")
                
                else:
                    st.warning("Please select at least one indicator to display.")
        else:
            st.error("No data available for the selected stock and time period.")
            
        
#10. Page Functions
def get_historical_data(symbol, days=30):
    """Fetch historical stock data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None
            
        # Convert index to datetime column
        df = df.reset_index()
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_stock_chart(data, symbol, days):
    """Create an interactive stock price chart"""
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title='Price ($)'),
        tooltip=[
            alt.Tooltip('Date:T', title='Date'),
            alt.Tooltip('Close:Q', title='Close', format='$.2f'),
            alt.Tooltip('Open:Q', title='Open', format='$.2f'),
            alt.Tooltip('High:Q', title='High', format='$.2f'),
            alt.Tooltip('Low:Q', title='Low', format='$.2f'),
            alt.Tooltip('Volume:Q', title='Volume', format=',')
        ]
    ).properties(
        title=f'{symbol} Stock Price - Past {days} Days',
        height=400
    ).interactive()
    
    volume_chart = alt.Chart(data).mark_bar().encode(
        x='Date:T',
        y=alt.Y('Volume:Q', title='Volume'),
        color=alt.condition(
            'datum.Open <= datum.Close',
            alt.value('#33CC33'),  # Green for up
            alt.value('#FF4444')   # Red for down
        ),
        tooltip=[
            alt.Tooltip('Date:T'),
            alt.Tooltip('Volume:Q', format=',')
        ]
    ).properties(
        height=200
    ).interactive()
    
    return chart & volume_chart

def market_summary_page():
    st.title("📊 Market Summary")
    
    # Stock selection with company names
    stock_options = {
        'AAPL': 'Apple (AAPL)',
        'GOOGL': 'Google (GOOGL)',
        'MSFT': 'Microsoft (MSFT)',
        'AMZN': 'Amazon (AMZN)',
        'TSLA': 'Tesla (TSLA)'
    }
    
    stock_symbol = st.selectbox(
        "Select Stock",
        options=list(stock_options.keys()),
        format_func=lambda x: stock_options[x]
    )
    
    # Fetch current stock data
    stock_data = get_live_market_data(stock_symbol)
    
    if stock_data:
        # Market Overview Section
        st.subheader("Market Overview")
        
        # Create metrics with colored backgrounds
        st.markdown("""
        <style>
        .metric-up { background-color: #e6ffe6; padding: 10px; border-radius: 5px; }
        .metric-down { background-color: #ffe6e6; padding: 10px; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate price change and percentage
        price_change = stock_data['c'] - stock_data['pc']
        change_percent = (price_change / stock_data['pc']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Current Price",
                value=f"${stock_data['c']:.2f}",
                delta=f"{price_change:.2f} ({change_percent:.2f}%)"
            )
        with col2:
            st.metric(label="Open", value=f"${stock_data['o']:.2f}")
        with col3:
            st.metric(label="High", value=f"${stock_data['h']:.2f}")
        with col4:
            st.metric(label="Low", value=f"${stock_data['l']:.2f}")
        
        # Historical Data Section
        st.subheader("Historical Data")
        
        # Time range selection
        time_ranges = {
            '1W': 7,
            '1M': 30,
            '3M': 90,
            '6M': 180,
            '1Y': 365
        }
        
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_range = st.select_slider(
                "Select Time Range",
                options=list(time_ranges.keys()),
                value='1M'
            )
        
        days = time_ranges[selected_range]
        
        # Fetch and display historical data
        with st.spinner("Loading historical data..."):
            hist_data = get_historical_data(stock_symbol, days)
            
            if hist_data is not None and not hist_data.empty:
                # Create and display chart
                chart = create_stock_chart(hist_data, stock_symbol, days)
                st.altair_chart(chart, use_container_width=True)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                stats = hist_data.describe()
                stats = stats[['Open', 'High', 'Low', 'Close', 'Volume']]
                st.dataframe(stats.round(2))
                
            else:
                st.error("No historical data available for the selected period.")
                
                

def display_news_cards(articles):
    """Display news in a clean, simple card layout"""
    st.markdown("""
    <style>
    .news-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #FF69B4;
    }
    .news-metadata {
        color: #666;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .news-title {
        color: #FF69B4;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .news-description {
        color: #333;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .news-link {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #FF69B4;
        color: white !important;
        text-decoration: none;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Search and sort
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("Search articles", "").lower()
    with col2:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest"])

    # Filter articles
    filtered_articles = [
        article for article in articles
        if search_term in article.get('title', '').lower() or 
           search_term in article.get('description', '').lower()
    ]

    # Sort articles
    filtered_articles.sort(
        key=lambda x: x.get('publishedAt', ''),
        reverse=(sort_by == "Newest")
    )

    # Show results count
    st.markdown(f"Showing **{len(filtered_articles)}** articles")

    # Display articles
    for article in filtered_articles:
        try:
            # Format date
            date = datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00'))
            date_str = date.strftime("%B %d, %Y")
        except:
            date_str = "Date unavailable"

        card_html = f"""
        <div class="news-card">
            <div class="news-metadata">{date_str}</div>
            <div class="news-title">{article.get('title', 'No title')}</div>
            <div class="news-description">{article.get('description', 'No description available')}</div>
            <a href="{article.get('url', '#')}" target="_blank" class="news-link">Read More →</a>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

@st.cache_data(ttl=900)
def fetch_news(symbol):
    """Fetch news articles for a given stock symbol"""
    try:
        news_api = NewsApiClient(api_key=NEWS_API_KEY)
        response = news_api.get_everything(
            q=symbol,
            language='en',
            sort_by='publishedAt',
            page_size=15
        )
        return response.get('articles', []) if response else []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def news_page():
    st.title("📰 Latest Stock News")
    
    # Simple stock selection with company names
    stock_options = {
        'AAPL': 'Apple (AAPL)',
        'GOOGL': 'Google (GOOGL)',
        'MSFT': 'Microsoft (MSFT)',
        'AMZN': 'Amazon (AMZN)',
        'TSLA': 'Tesla (TSLA)'
    }
    
    selected_stock = st.selectbox(
        "Select Stock",
        options=list(stock_options.keys()),
        format_func=lambda x: stock_options[x]
    )
    
    with st.spinner("Loading news..."):
        articles = fetch_news(selected_stock)
        
        if articles:
            display_news_cards(articles)
        else:
            st.warning("No news articles found. Please try another stock symbol.")


# Assume this function exists elsewhere in your code
def format_published_date(published_date):
    # Parse the ISO 8601 formatted date string
    parsed_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
    # Format the date to a more user-friendly format
    return parsed_date.strftime("%b %d, %Y")
        
#11. Helper functions for the news_page()
def format_published_date(published_date):
    # Parse the ISO 8601 formatted date string
    parsed_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
    # Format the date to a more user-friendly format
    return parsed_date.strftime("%b %d, %Y")

def sentiment_analysis_page():
    st.title("🔍 Sentiment Analysis")
    stock_symbol = st.selectbox("Select Stock Symbol", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
    
    articles = fetch_news(stock_symbol)
    if articles:
        sentiments = analyze_sentiment(articles)
        
        # Create a DataFrame for better visualization
        df = pd.DataFrame({
            'Title': [article['title'] for article in articles],
            'Positive': [sentiment['pos'] for sentiment in sentiments],
            'Neutral': [sentiment['neu'] for sentiment in sentiments],
            'Negative': [sentiment['neg'] for sentiment in sentiments],
            'Compound': [sentiment['compound'] for sentiment in sentiments]
        })
        
        st.subheader("Sentiment Heatmap")
        
        # Create a more defined heatmap using Altair
        heatmap = alt.Chart(df.melt(id_vars=['Title'], var_name='Sentiment', value_name='Score')).mark_rect().encode(
            x=alt.X('Sentiment:N', title=None),
            y=alt.Y('Title:N', title=None, sort='-x'),
            color=alt.Color('Score:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Title', 'Sentiment', 'Score']
        ).properties(
            width=600,
            height=400,
            title=f"Sentiment Analysis Heatmap for {stock_symbol}"
        )
        
        st.altair_chart(heatmap, use_container_width=True)
        
        st.subheader("Detailed Sentiment Scores")
        st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['Positive', 'Neutral', 'Negative', 'Compound']))
        
        # Add an overall sentiment summary
        avg_sentiment = df['Compound'].mean()
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        st.subheader("Overall Sentiment Summary")
        st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}", sentiment_label)
        
        # Generate AI insights
        prompt = f"Analyze the following sentiment scores for {stock_symbol} news articles:\n{df.to_string()}\nProvide a brief, insightful summary of the overall sentiment and its potential impact on the stock."
        response = model.generate_content(prompt)
        st.subheader("AI Insights")
        st.write(response.text)

        # Sentiment Score Explanation
        st.markdown('<div class="explanation-container">', unsafe_allow_html=True)
        st.subheader("Sentiment Score Explanation")
        for i, article in enumerate(articles[:5]):
            with st.expander(f"**{article['title']}**"):
                st.write(f"Positive: {sentiments[i]['pos']}, Neutral: {sentiments[i]['neu']}, Negative: {sentiments[i]['neg']}, Compound: {sentiments[i]['compound']}")
                st.write(f"Key Phrases: {', '.join([word for word in article['content'].split() if word.lower() not in nltk.corpus.stopwords.words('english')])}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Sentiment Comparison
        st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
        st.subheader("Sentiment Comparison")
        comparison_symbols = st.multiselect("Select Stocks to Compare", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], default=[stock_symbol])
        comparison_data = {}
        for symbol in comparison_symbols:
            comparison_articles = fetch_news(symbol)
            comparison_sentiments = analyze_sentiment(comparison_articles)
            comparison_data[symbol] = pd.DataFrame({
                'Article': [a['title'] for a in comparison_articles],
                'Compound': [s['compound'] for s in comparison_sentiments]
            })
        comparison_df = pd.concat(comparison_data.values(), keys=comparison_data.keys())
        comparison_df.reset_index(inplace=True)
        comparison_df.rename(columns={'level_0': 'Symbol'}, inplace=True)

        comparison_chart = alt.Chart(comparison_df).mark_boxplot(extent='min-max').encode(
            x='Symbol:N',
            y='Compound:Q',
            color='Symbol:N',
            tooltip=['Symbol', 'Compound']
        ).properties(
            title='Sentiment Comparison',
            width=800,
            height=400
        ).interactive()

        st.altair_chart(comparison_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def chatbot_page():
    st.title("🤖 Stocket AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about stocks?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in model.generate_content(prompt, stream=True):
                full_response += response.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

#12. Main Dashboard Function
def main():
    if check_credentials():
        st.sidebar.title("Navigation")

        if st.sidebar.button("📊 Market Summary"):
            st.session_state.page = "market_summary"
        if st.sidebar.button("📰 News"):
            st.session_state.page = "news"
        if st.sidebar.button("🔍 Sentiment Analysis"):
            st.session_state.page = "sentiment_analysis"
        if st.sidebar.button("📊 AI-Powered Technical Analysis"):
            st.session_state.page = "technical_analysis"
        if st.sidebar.button("🤖 AI Assistant"):
            st.session_state.page = "ai_assistant"

        if "page" not in st.session_state:
            st.session_state.page = "market_summary"

        if st.session_state.page == "market_summary":
            market_summary_page()
        elif st.session_state.page == "news":
            news_page()
        elif st.session_state.page == "sentiment_analysis":
            sentiment_analysis_page()
        elif st.session_state.page == "technical_analysis":
            technical_analysis_page()
        elif st.session_state.page == "ai_assistant":
            chatbot_page()

if __name__ == "__main__":
    main()
