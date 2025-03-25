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

# Sentiment Analysis on Reviews
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import pipeline

# Load dataset 
file_path = "product_reviews.csv" 
df = pd.read_csv(file_path)

# Streamlit UI
st.title("Product Review Sentiment Analysis")
st.write("This dashboard shows the sentiment analysis of product reviews.")

# Check if required columns exist
if "review" not in df.columns or "product_name" not in df.columns:
    st.error("Dataset must contain 'product_name' and 'review' columns!")
else:
    # Initialize sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Function to analyze sentiment
    def analyze_sentiment(review):
        result = sentiment_pipeline(review)[0]
        return result["label"]

    # Apply sentiment analysis
    df["Sentiment"] = df["review"].apply(analyze_sentiment)

    # Group data by product and sentiment
    sentiment_counts = df.groupby(["product_name", "Sentiment"]).size().unstack(fill_value=0)

    # Plot sentiment distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    sentiment_counts.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
    plt.title("Sentiment Analysis of Products")
    plt.xlabel("Product")
    plt.ylabel("Count of Sentiments")
    plt.legend(title="Sentiment")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Display plot in Streamlit
    st.pyplot(fig)

    # Show processed data
    st.write("Processed Data:")
    st.dataframe(df)
