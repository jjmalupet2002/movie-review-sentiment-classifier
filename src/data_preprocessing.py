# preprocessing functions here 

import pandas as pd
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()

def load_dataset(filepath):
    """
    Loads a CSV file containing movie reviews and their labels.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with the dataset.
    """
    return pd.read_csv(filepath) 

def clean_text(text):
    """
    Cleans input text by lowercasing, removing punctuation, and stopwords.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words) 

def stem_text(text):
    """
    Applies stemming to each word in the text.

    Args:
        text (str): The text to stem.

    Returns:
        str: Stemmed text.
    """
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words) 

def vectorize_text(texts, max_features=10000):
    """
    Converts a list/series of text documents into TF-IDF feature vectors.

    Args:
        texts (list or pd.Series): The cleaned text data.
        max_features (int): Maximum number of features (vocabulary size).

    Returns:
        X (sparse matrix): TF-IDF feature matrix.
        vectorizer (TfidfVectorizer): Fitted vectorizer object.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer 