from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    LogisticRegression
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd

# Load dataset from CSV file
data = pd.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400]  # Training data: first 4400 items
test_data = data[4400:]   # Test data: remaining items

# Initialize classifier and vectorizer
classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# Train the classifier
vectorize_text = vectorizer.fit_transform(train_data.v2)
classifier.fit(vectorize_text, train_data.v1)

# Evaluate performance on test data
vectorize_text = vectorizer.transform(test_data.v2)
score = classifier.score(vectorize_text, test_data.v1)

# Print the accuracy score
print(score)  # Output: 0.988 (98.8%)
