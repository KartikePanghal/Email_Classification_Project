from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load pre-trained model and vectorizer
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logger.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or vectorizer: {e}")
    raise

def classify_email(email_body):
    try:
        features = vectorizer.transform([email_body])
        category = model.predict(features)[0]
        logger.info(f"Email classified successfully: {category}")
        return category
    except Exception as e:
        logger.error(f"Error during email classification: {e}")
        raise
