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

