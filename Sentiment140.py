import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from PIL import Image

st.title("Twitter Sentiment Analysis")
image=Image.open("Twitter_logo.png")
st.image(image,use_column_width=True)

# Ensure NLTK stopwords and tokenizer are available

nltk.download('punkt')
nltk.download('stopwords')

# Load the pickled logistic regression model
model_filename ="NaiveBayes_Tweets.pkl"
model = pickle.load(open(model_filename, 'rb'))

# Load the TF-IDF Vectorizer (ensure you save this correctly during training)
vectorizer_filename ="tfidf1_tweets.pkl"
tfidf_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

# Function to preprocess text (e.g., stemming or removing stopwords) to match the training preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


# Text input for the user to enter a tweet
user_input = st.text_area("Enter a Tweet:", "")

# Predict sentiment based on user input
if st.button("Analyze"):
    if user_input:
        # Preprocess the input text to match training preprocessing
        preprocessed_text = preprocess_text(user_input)
        
        # Transform the preprocessed input using the TF-IDF vectorizer
        input_data = [preprocessed_text]
        input_tfidf = tfidf_vectorizer.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_tfidf)
        prediction_prob = model.predict_proba(input_tfidf)

        # Display the sentiment
        if prediction == 1:
            st.success(f"The tweet is Positive with probability {prediction_prob[0][1]:.2f}")
        else:
            st.error(f"The tweet is Negative with probability {prediction_prob[0][0]:.2f}")
    else:
        st.warning("Please enter a tweet to analyze.")


