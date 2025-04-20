# Email Classification API

This project is a FastAPI-based application for classifying emails into predefined categories. It includes functionality for masking Personally Identifiable Information (PII) in email content, classifying the email, and then demasking the PII.

## Features

- **PII Masking**: Automatically detects and masks sensitive information such as names, email addresses, phone numbers, etc.
- **Email Classification**: Classifies emails into categories using a pre-trained machine learning model.
- **PII Demasking**: Restores the original PII after classification.

## Project Structure

- `app.py`: Main FastAPI application with an endpoint for email classification.
- `utils.py`: Utility functions for masking and demasking PII.
- `models.py`: Loads the pre-trained model and vectorizer for email classification.
- `train_model.py`: Script for training the email classification model.
- `requirements.txt`: List of dependencies required for the project.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. Train the model (optional if `model.pkl` and `vectorizer.pkl` are already provided):
   ```bash
   python train_model.py
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

2. Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

### API Endpoint

- **POST** `/classify_email`: Classifies an email and returns the category along with masked and demasked content.

#### Request Body
```json
{
  "email_body": "Your email content here"
}
```

#### Response
```json
{
  "input_email_body": "Original email content",
  "list_of_masked_entities": [
    {
      "position": [start, end],
      "classification": "entity_type",
      "entity": "original_value"
    }
  ],
  "masked_email": "Masked email content",
  "category_of_the_email": "Category"
}
```

## Training the Model

To train the model, ensure you have a dataset in CSV format with the following columns:
- `email`: The email content.
- `type`: The category of the email.

Update the `DATA_PATH` in `train_model.py` to point to your dataset and run:
```bash
python train_model.py
```

## Dependencies

- FastAPI
- Uvicorn
- Scikit-learn
- Pydantic
- Pandas
- Imbalanced-learn
- NLTK

## Logging

The application uses Python's `logging` module to log important events and errors. Logs are displayed in the console.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
# Email Classification API

This project is a FastAPI-based application for classifying emails into predefined categories. It includes functionality for masking Personally Identifiable Information (PII) in email content, classifying the email, and then demasking the PII.

## Features

- **PII Masking**: Automatically detects and masks sensitive information such as names, email addresses, phone numbers, etc.
- **Email Classification**: Classifies emails into categories using a pre-trained machine learning model.
- **PII Demasking**: Restores the original PII after classification.

## Project Structure

- `app.py`: Main FastAPI application with an endpoint for email classification.
- `utils.py`: Utility functions for masking and demasking PII.
- `models.py`: Loads the pre-trained model and vectorizer for email classification.
- `train_model.py`: Script for training the email classification model.
- `requirements.txt`: List of dependencies required for the project.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. Train the model (optional if `model.pkl` and `vectorizer.pkl` are already provided):
   ```bash
   python train_model.py
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

2. Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

### API Endpoint

- **POST** `/classify_email`: Classifies an email and returns the category along with masked and demasked content.

#### Request Body
```json
{
  "email_body": "Your email content here"
}
```

#### Response
```json
{
  "input_email_body": "Original email content",
  "list_of_masked_entities": [
    {
      "position": [start, end],
      "classification": "entity_type",
      "entity": "original_value"
    }
  ],
  "masked_email": "Masked email content",
  "category_of_the_email": "Category"
}
```

## Training the Model

To train the model, ensure you have a dataset in CSV format with the following columns:
- `email`: The email content.
- `type`: The category of the email.

Update the `DATA_PATH` in `train_model.py` to point to your dataset and run:
```bash
python train_model.py
```

## Dependencies

- FastAPI
- Uvicorn
- Scikit-learn
- Pydantic
- Pandas
- Imbalanced-learn
- NLTK

## Logging

The application uses Python's `logging` module to log important events and errors. Logs are displayed in the console.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
