import pandas as pd
import re
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from data_preprocessing import load_dataset, clean_text, stem_text, vectorize_text
from model import split_data, train_logistic_regression, train_naive_bayes
from sklearn.metrics import accuracy_score, classification_report
from evaluate import evaluate_model, plot_confusion_matrix

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a CSV dataset into a DataFrame."""
    return pd.read_csv(filepath)

def clean_text(text: str) -> str:
    """Basic text cleaning: remove HTML tags, non-alphabetic characters, lowercasing."""
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-letters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing spaces
    return text

def stem_text(text: str) -> str:
    """Apply stemming to text using PorterStemmer."""
    ps = PorterStemmer()
    words = text.split()
    stemmed_words = [ps.stem(word) for word in words]
    return " ".join(stemmed_words)

def vectorize_text(text_series):
    """Vectorize text using TF-IDF and return feature matrix and vectorizer."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer

if __name__ == "__main__":
    # Example usage
    df = load_dataset("data/raw/IMDB Dataset.csv")  # Update filename if needed

    print("Original review sample:")
    print(df['review'].iloc[0])

    df['cleaned'] = df['review'].apply(clean_text)
    df['stemmed'] = df['cleaned'].apply(stem_text)

    print("\nCleaned review sample:")
    print(df['cleaned'].iloc[0])
    print("\nStemmed review sample:")
    print(df['stemmed'].iloc[0])

    X, vectorizer = vectorize_text(df['stemmed'])
    print(f"\nFeature matrix shape: {X.shape}")

    # 1. Load and preprocess data
    df = load_dataset("data/raw/IMDB Dataset.csv")
    df['cleaned'] = df['review'].apply(clean_text)
    df['stemmed'] = df['cleaned'].apply(stem_text)
    X, vectorizer = vectorize_text(df['stemmed'])
    y = df['sentiment']  # Make sure this column exists and is 'positive'/'negative'

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Train models
    logreg_model = train_logistic_regression(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)

    # 4. Evaluate models
    # Predict and evaluate Logistic Regression
    y_pred_logreg = logreg_model.predict(X_test)
    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
    print(classification_report(y_test, y_pred_logreg))
    print("\nLogistic Regression Evaluation:")
    evaluate_model(y_test, y_pred_logreg)
    plot_confusion_matrix(
        y_test, y_pred_logreg,
        labels=['negative', 'positive'],
        title='Logistic Regression Confusion Matrix',
        save_path='logreg_confusion.png'
    )

    # Predict and evaluate Naive Bayes
    y_pred_nb = nb_model.predict(X_test)
    print("\nNaive Bayes Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb))
    print("\nNaive Bayes Evaluation:")
    evaluate_model(y_test, y_pred_nb)
    plot_confusion_matrix(
        y_test, y_pred_nb,
        labels=['negative', 'positive'],
        title='Naive Bayes Confusion Matrix',
        save_path='nb_confusion.png'
    )