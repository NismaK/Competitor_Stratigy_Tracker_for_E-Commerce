import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set page title
st.title("E-commerce Competitor Strategy Dashboard")

# Load data
competitor_data = pd.read_csv("Prices_Dataset.csv")
reviews_data = pd.read_csv("product_reviews.csv")

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


# Sentiment Analysis Function
from transformers import pipeline

# Load a pre-trained sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return "Neutral"  # Assign 'Neutral' for missing/empty values
    
    result = sentiment_pipeline(text)[0]  # Get sentiment prediction
    sentiment = result["label"]

    # Convert model output to simpler labels
    if sentiment == "POSITIVE":
        return "Positive"
    elif sentiment == "NEGATIVE":
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis using BERT
if "review" in product_reviews.columns:
    product_reviews["Sentiment"] = product_reviews["review"].astype(str).apply(get_sentiment)
else:
    st.error("The 'review' column is missing in the dataset.")

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
