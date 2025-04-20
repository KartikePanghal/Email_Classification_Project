import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    logger.info("Downloading NLTK resources...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    logger.info("NLTK resources downloaded successfully.")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    raise

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def train_and_save_model(data_path, model_path, vectorizer_path):
    try:
        # Load dataset
        logger.info("Loading dataset...")
        data = pd.read_csv(data_path)
        data['email'] = data['email'].apply(preprocess_text)  # Preprocess text
        emails = data['email']
        labels = data['type']

        # Split dataset
        logger.info("Splitting dataset into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

        # Vectorize email text with n-grams
        logger.info("Vectorizing email text...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Handle class imbalance using SMOTE
        logger.info("Handling class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

        # Train model with hyperparameter tuning
        logger.info("Training classification model with hyperparameter tuning...")
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
        grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', random_state=42), param_grid, cv=3)
        grid_search.fit(X_train_tfidf, y_train)
        model = grid_search.best_estimator_

        # Evaluate model
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy}")
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Save model and vectorizer
        logger.info("Saving model and vectorizer...")
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(vectorizer_path, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

        logger.info("Model and vectorizer saved successfully.")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    # Define file paths
    DATA_PATH = "combined_emails_with_natural_pii.csv"  
    MODEL_PATH = "model.pkl"
    VECTORIZER_PATH = "vectorizer.pkl"

    # Train and save the model
    train_and_save_model(DATA_PATH, MODEL_PATH, VECTORIZER_PATH)
