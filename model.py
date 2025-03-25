import pandas as pd

def load_competitor_data():
    return pd.read_csv("price_discount_predictions.csv")

def load_reviews_data():
    return pd.read_csv("product_reviews.csv")

def analyze_sentiment(reviews):
    return [{"label": r, "sentiment": "Positive"} for r in reviews]  # Dummy function
