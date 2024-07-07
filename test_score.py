from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
import csv

# Load data from CSV file
data = pd.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400]  # Training data: first 4400 items
test_data = data[4400:]   # Test data: remaining items

# Initialize the classifier and vectorizer
classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# Train the classifier
vectorize_text = vectorizer.fit_transform(train_data['v2'])
classifier.fit(vectorize_text, train_data['v1'])

# Score the classifier on test data
vectorize_text_test = vectorizer.transform(test_data['v2'])
score = classifier.score(vectorize_text_test, test_data['v1'])
print(f"Accuracy Score: {score * 100:.1f}%")  # Output: 98.8

# Prepare CSV array for detailed evaluation
csv_arr = []
for index, row in test_data.iterrows():
    answer = row['v1']
    text = row['v2']
    vectorize_text = vectorizer.transform([text])
    predict = classifier.predict(vectorize_text)[0]
    result = 'correct' if predict == answer else 'wrong'
    csv_arr.append([index, text, answer, predict, result])

# Write detailed evaluation results to CSV file
csv_filename = 'test_score.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', 'result'])
    for row in csv_arr:
        spamwriter.writerow(row)

print(f"Detailed evaluation results saved to {csv_filename}")
