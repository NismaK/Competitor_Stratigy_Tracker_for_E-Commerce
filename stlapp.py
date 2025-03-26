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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
import re
# Function to extract top words
def extract_top_words(text, n=5):
    # Remove stopwords and clean text
    words = re.findall(r'\w+', text.lower())
    stopwords = set(STOPWORDS)
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort and get top words
    top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:n]
    return top_words

# Strategic Recommendation Generation Function
def generate_strategic_recommendations(product_name, sentiment_data, reviews_data):
    """
    Generate strategic recommendations using a text generation model
    
    Args:
        product_name (str): Name of the product
        sentiment_data (pd.Series): Sentiment distribution
        reviews_data (pd.DataFrame): Raw review data
    
    Returns:
        str: Strategic recommendations for promotional campaigns and improvements
    """
    # Analyze sentiment distribution
    total_reviews = len(reviews_data)
    positive_percentage = (sentiment_data.get('Positive', 0) / total_reviews) * 100
    negative_percentage = (sentiment_data.get('Negative', 0) / total_reviews) * 100
    
    # Extract key insights from reviews
    all_reviews_text = " ".join(str(review) for review in reviews_data["review"])
    key_positive_features = extract_top_words(all_reviews_text)
    
    # Construct prompt for recommendation generation
    prompt = f"""Based on product analysis, generate strategic recommendations:

Product: {product_name}
Sentiment Analysis:
- Total Reviews: {total_reviews}
- Positive Reviews: {positive_percentage:.2f}%
- Negative Reviews: {negative_percentage:.2f}%
Key Product Features: {', '.join(key_positive_features)}

Provide actionable recommendations for:
1. Promotional Campaign Strategy
2. Customer Satisfaction Improvement
3. Product Positioning
4. Marketing Messaging

Detail specific strategies based on the sentiment and feature analysis.

Recommendations:
"""
    
    # Use a lightweight, publicly available model
    try:
        # Switch to a smaller, more accessible model
        model_name = "distilgpt2"  # Lightweight version of GPT-2
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Prepare input
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate text
        output = model.generate(
            input_ids, 
            max_length=1024, 
            num_return_sequences=1, 
            temperature=0.7,  # Increased creativity
            no_repeat_ngram_size=2  # Reduce repetition
        )
        
        # Decode and clean up the generated text
        recommendations = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract recommendations section
        if "Recommendations:" in recommendations:
            recommendations = recommendations.split("Recommendations:")[1].strip()
        
        return recommendations
    
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# Strategic Recommendations Section
st.title("Strategic Recommendations")
if st.button("Generate Strategic Insights"):
    with st.spinner('Generating strategic recommendations...'):
        recommendations = generate_strategic_recommendations(
            selected_product, 
            sentiment_counts, 
            product_reviews
        )
        st.write(recommendations)
     
