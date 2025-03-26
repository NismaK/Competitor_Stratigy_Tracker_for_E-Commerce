import streamlit as st
import pandas as pd
import plotly.express as px
from model import analyze_sentiment, load_competitor_data, load_reviews_data
st.title("E-commerce Competitor Strategy Dashboard")

# Load data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Sidebar for product selection
products = competitor_data["product_name"].unique().tolist()
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

# Filter data based on selection
product_data = competitor_data[competitor_data["product_name"] == selected_product]
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]

# Display competitor data
st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail())

st.title("Sentiment Analysis Dashboard")

# Sidebar for product selection
products = reviews_data["product_name"].unique().tolist()
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

# Filter data for the selected product
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]

# Sentiment Analysis Function
def get_sentiment(text):
    analysis = TextBlob(str(text))  # Convert to string and analyze
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
product_reviews["Sentiment"] = product_reviews["review"].apply(get_sentiment)
# product_reviews=pd.read_csv("sentiment_results.csv")

# Sentiment Distribution Visualization
st.subheader("Sentiment Distribution")
sentiment_counts = product_reviews["Sentiment"].value_counts()
fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
             labels={"x": "Sentiment", "y": "Count"}, title="Sentiment Distribution")
st.plotly_chart(fig)

# Word Cloud for Reviews
st.subheader("Word Cloud")
all_reviews = " ".join(str(review) for review in product_reviews["review"])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)


