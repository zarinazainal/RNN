# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions  
# Function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st

## streamlit app
#streamlit app
##st.title("IMDB Movie Review Sentiment Analysis")
##st.write("Enter a movie review to predict its a positive or negative.")

# User input
##user_input = st.text_area('Movie Review')

#if st.button('Classify'):

  #  preprocess_input = preprocess_text(user_input)

    ## Make prediction
   # prediction = model.predict(preprocess_input)
   # sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display results
    
   # st.write(f"Sentiment: {sentiment}") 
   # st.write(f"Prediction Score: {prediction[0][0]:.4f}")
#else:
   # st.write('Please enter a movie review.')

   # ---------------- Streamlit UI ---------------- #
st.set_page_config(
    page_title="🎬 IMDB Review Sentiment Analyzer",
    page_icon="🍿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title Section
st.markdown("""
    <div style="text-align: center;">
        <h1 style='color: #FF4B4B;'>🎬 IMDB Review Sentiment Analyzer</h1>
        <p style='font-size: 18px;'>Enter a movie review below, and let's see if it's 🍭 *sweetly positive* or 🌧️ *bitterly negative*!</p>
    </div>
""", unsafe_allow_html=True)

# Input Section
user_input = st.text_area("📝 Your Movie Review Here:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Come on now... Type something in the box first! 🎈")
    else:
        with st.spinner("Running sentiment sensors... 🤖"):
            processed_input = preprocess_text(user_input)
            prediction = model.predict(processed_input, verbose=0)
            score = prediction[0][0]
            sentiment = "💖 Positive" if score > 0.5 else "💔 Negative"

        st.success("Here's what I think...")
        st.markdown(f"""
            <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;">
                <h3 style='color:#008080;'>Sentiment: {sentiment}</h3>
                <p style='font-size:16px;'>Prediction Score: <strong>{score:.4f}</strong></p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Paste your review and click the button to get started!")

# Footer
st.markdown("""<hr style="margin-top:50px;"/>""", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:14px;'>Made with ❤️ using Streamlit + TensorFlow</p>", unsafe_allow_html=True)