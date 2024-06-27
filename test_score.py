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

# Load data
data = pd.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400]  # 4400 items
test_data = data[4400:]   # 1172 items

# Initialize the classifier and vectorizer
classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# Train the classifier
vectorize_text = vectorizer.fit_transform(train_data['v2'])
classifier.fit(vectorize_text, train_data['v1'])

# Score the classifier
vectorize_text_test = vectorizer.transform(test_data['v2'])
score = classifier.score(vectorize_text_test, test_data['v1'])
print(score)  # 98.8

# Prepare CSV array
csv_arr = []
for index, row in test_data.iterrows():
    answer = row['v1']
    text = row['v2']
    vectorize_text = vectorizer.transform([text])
    predict = classifier.predict(vectorize_text)[0]
    result = 'right' if predict == answer else 'wrong'
    csv_arr.append([len(csv_arr), text, answer, predict, result])

# Write CSV
with open('test_score.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', 'result'])
    for row in csv_arr:
        spamwriter.writerow(row)
