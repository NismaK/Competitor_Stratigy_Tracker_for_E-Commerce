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
API_URL = "https://api.groq.com/v1/chat/completions"
HEADERS = {"Authorization": "Bearer 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", "Content-Type": "application/json"}
# Generate Strategic Recommendations using LLM
import requests
import json
import pandas as pd
latestP=competitor_data["price"]
prompt = f"""
You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest strategies to optimize pricing, promotions, and customer satisfaction for {products}:

1. Competitor Data (Prices, Discounts, Predicted Discounts):
{latestP.to_dict()}

2. Sentiment Analysis:
{sentiment_counts.to_dict()}

3. Customer Reviews:
{product_reviews.to_dict(orient="records")}

Current Date: {pd.Timestamp.today().strftime('%Y-%m-%d')}

### Task:
- Identify key pricing trends.
- Suggest strategies for optimizing pricing over the next 3 days.
- Recommend marketing campaigns aligned with sentiment trends.
- Provide actionable, realistic strategies to increase sales and customer satisfaction.

Structure the response as:
1. Pricing Strategy
2. Promotional Campaign Ideas
3. Customer Satisfaction Recommendations
"""
data = {"model": "mixtral-8x7b", "messages": [{"role": "system", "content": prompt}], "temperature": 0.7, "response_format": {"type": "json_object"}}
response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))

if response.status_code == 200:
    strategy_output = response.json().get("content", "No recommendations available.")
else:
    strategy_output = "Error: Unable to fetch recommendations. Check API Key and Internet."

st.subheader("Strategic Recommendations")
st.write(strategy_output)
