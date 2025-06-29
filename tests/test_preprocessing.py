# unit tests for preprocessing functions 
import unittest
from src.data_preprocessing import clean_text, stem_text, vectorize_text

class TestPreprocessing(unittest.TestCase):
    def test_clean_text(self):
        text = "Hello, World! This is a test."
        cleaned = clean_text(text)
        self.assertNotIn(",", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertTrue(cleaned.islower())
        self.assertNotIn("this", cleaned)  # 'this' is a stopword

    def test_stem_text(self):
        text = "running runner runs"
        stemmed = stem_text(text)
        self.assertIn("run", stemmed)
        self.assertIn("runner", stemmed)  # 'runner' is not stemmed to 'run'
        self.assertNotIn("running", stemmed)
        self.assertNotIn("runs", stemmed)

    def test_vectorize_text(self):
        texts = ["the cat sat", "the dog barked"]
        X, vectorizer = vectorize_text(texts, max_features=5)
        self.assertEqual(X.shape[0], 2)  # 2 documents
        self.assertLessEqual(X.shape[1], 5)  # max_features

if __name__ == "__main__":
    unittest.main() 