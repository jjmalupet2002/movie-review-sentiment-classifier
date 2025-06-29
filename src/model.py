# model training and saving functions here 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and labels into training and testing sets.

    Args:
        X: Feature matrix.
        y: Labels.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed.

    Returns:
        X_train, X_test, y_train, y_test: Split data.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression classifier.

    Returns:
        Trained LogisticRegression model.
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def train_naive_bayes(X_train, y_train):
    """
    Trains a Multinomial Naive Bayes classifier.

    Returns:
        Trained MultinomialNB model.
    """
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf 

def save_model(model, filename):
    """
    Saves the trained model to a file.
    """
    joblib.dump(model, filename) 