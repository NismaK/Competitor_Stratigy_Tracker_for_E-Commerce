import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

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
import requests
import streamlit as st
import pandas as pd

# Load price discount predictions
price_discount_prediction = pd.read_csv("price_discount_predictions.csv")

# Set your Groq API key here
GROQ_API_KEY = "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"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to call Groq API for strategic recommendations
def get_strategic_recommendation(products, competitor_data, sentiment_counts, predicted_discount):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Define LLM prompt
    prompt = f"""
    You are a highly skilled e-commerce strategist with expertise in pricing optimization, promotional campaigns, and customer satisfaction. Your task is to analyze the given data and provide **structured** recommendations.

    ### 📌 Strategic Recommendations

    #### 1️⃣ Pricing Strategy
    - Should we **increase, decrease, or maintain** our pricing? Justify based on competitor trends and sentiment analysis.
    - What is the **optimal price range**? Provide specific percentage adjustments.
    - How will this **affect customer demand and revenue**?

    #### 2️⃣ Promotional Campaign Ideas
    - Suggest **3-5 detailed campaign strategies** based on competitor promotions and trends.
    - Specify **execution tactics** (timing, platforms, target audience).
    - Include **projected impact** (reach, engagement, sales boost).

    #### 3️⃣ Customer Satisfaction Recommendations
    - Identify **pain points** from sentiment analysis (delivery delays, pricing complaints, support issues).
    - Recommend **actionable solutions**.

    #### 4️⃣ Actionable Recommendations
    - Based on the above insights, provide **clear, actionable strategies**.
    - Use **markdown formatting** (### for subheadings, bullet points, bold highlights).
    - Keep it **concise, structured, and practical**.

    **Product Name:** {products}

    **Competitor Insights:**
    - **Current Prices & Discounts:** {competitor_data}
    - **Predicted Discount Trends:** {predicted_discount}

    **Customer Sentiment Analysis:** {sentiment_counts}
    """

    # Make API Request
    payload = {
        "model": "llama-3.1-8b-instant",  
        "messages": [
            {"role": "system", "content": "You are an expert e-commerce strategist."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)

    # Debugging - Print Full Response
    print("Full API Response:", response.json())

    response_data = response.json()
    if "choices" in response_data and response_data["choices"]:
        return response_data["choices"][0]["message"]["content"]
    else:
        return "⚠️ No response from the API. Check API key and request format."

# Streamlit App UI
st.title("LLM-Powered Strategic Recommendations")

# Retrieve relevant data
competitor_data = "Competitor A: $50, Competitor B: $45, Our Price: $48"
sentiment_counts = "Positive: 60%, Neutral: 25%, Negative: 15%"
predicted_discount = price_discount_prediction[price_discount_prediction["product_name"] == selected_product]

# Generate and display recommendations
recommendations = get_strategic_recommendation(selected_product, competitor_data, sentiment_counts, predicted_discount)
st.subheader("📌 Strategic Recommendations")
st.write(recommendations)  # Display properly formatted output
