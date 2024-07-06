import os
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Global variables for classifier and vectorizer
Classifier = None
Vectorizer = None

# Load data from CSV file
data = pd.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400]  # Training data: first 4400 items
test_data = data[4400:]   # Test data: remaining items

# Train model on the training data
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Classifier.fit(vectorize_text, train_data.v1)

# Define route for the web application
@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')  # Get 'message' parameter from request URL
    error = ''
    predict_proba = ''
    predict = ''

    global Classifier, Vectorizer
    try:
        if len(message) > 0:
            # Vectorize the input message
            vectorize_message = Vectorizer.transform([message])
            # Predict the class label
            predict = Classifier.predict(vectorize_message)[0]
            # Get probabilities for each class
            predict_proba = Classifier.predict_proba(vectorize_message).tolist()
    except Exception as e:
        error = str(type(e).__name__) + ' ' + str(e)

    # Return JSON response
    return jsonify(
        message=message,
        predict_proba=predict_proba,
        predict=predict,
        error=error
    )

if __name__ == '__main__':
    # Set port for the application
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
