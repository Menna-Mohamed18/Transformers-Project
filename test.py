import streamlit as st
import torch
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pickle
import os

# Set NLTK data path
nltk.data.path.append('C:\\Users\\11\\AppData\\Roaming\\nltk_data')

# Download necessary NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # Added punkt_tab
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}")
    st.stop()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    try:
        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        if soup is None:
            st.error("BeautifulSoup failed to parse the input text.")
            return None
        text = soup.get_text()
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#', '', text)
        # Remove punctuations and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize text
        try:
            tokens = word_tokenize(text)
        except LookupError as e:
            st.error(f"Tokenization failed: {str(e)}")
            return None
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Rejoin tokens
        text = ' '.join(tokens)
        if not text or len(text.strip()) == 0:
            st.error("Input text is empty after preprocessing.")
            return None
        return text
    except Exception as e:
        st.error(f"Text preprocessing failed: {str(e)}")
        return None

# Prediction function
def predict_sentiment(text, model, tokenizer, device, label_encoder):
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        if not processed_text:
            return None, None
        
        # Tokenize
        encoded_dict = tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Move to device
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get prediction class
        prediction = torch.argmax(logits, dim=1).item()
        
        # Map prediction to sentiment
        sentiment = label_encoder.inverse_transform([prediction])[0]
        
        return sentiment, probs
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# Streamlit app
def main():
    st.title("Sentiment Analysis with DistilBERT")
    st.write("Enter a movie review to predict its sentiment (positive or negative).")

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check for required files
    model_dir = 'sentiment_model'
    weights_path = 'best_model.pt'
    label_encoder_path = 'label_encoder.pkl'

    if not os.path.exists(model_dir):
        st.error(f"Model directory '{model_dir}' not found. Please ensure 'model_setup.py' has been run.")
        st.stop()
    if not os.path.exists(weights_path):
        st.error(f"Model weights '{weights_path}' not found. Please ensure 'model_training.py' has been run.")
        st.stop()
    if not os.path.exists(label_encoder_path):
        st.error(f"Label encoder '{label_encoder_path}' not found. Please ensure 'data_preprocessing.py' has been run.")
        st.stop()

    # Load model, tokenizer, and label encoder
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        st.success("Model, tokenizer, and label encoder loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {str(e)}")
        st.stop()

    # Text input
    user_input = st.text_area("Enter your movie review:", height=150)

    # Predict button
    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a review before predicting.")
        else:
            with st.spinner("Predicting..."):
                sentiment, probs = predict_sentiment(user_input, model, tokenizer, device, label_encoder)
                if sentiment and probs is not None:
                    st.subheader("Prediction Result")
                    st.write(f"**Sentiment**: {sentiment}")
                    st.write(f"**Confidence**:")
                    st.write(f"- Positive: {probs[1]:.2%}")
                    st.write(f"- Negative: {probs[0]:.2%}")
                else:
                    st.error("Prediction failed. Please try again.")



if __name__ == "__main__":
    main()