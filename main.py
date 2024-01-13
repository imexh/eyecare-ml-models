import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)


# 0 Female/No
# 1 Male/Yes

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)


# encoding the columns
def encode_categorical_columns(data, categorical_columns):
    label_encoder = LabelEncoder()

    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data


def split_data(data, test_size=0.2, random_state=42):
    X = data.drop(columns=['Status of the csv', 'Do you think you have Computer Vision Syndrome?'])
    y = data['Status of the csv']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# Train model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)


# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Classification report:", classification_rep)

    print("")

    # Create box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot([y_test, probabilities], labels=["Original Test Data", "Predicted Test Data"])
    plt.title("Distribution of Original and Predicted Test Data")
    plt.xlabel("Data Type")
    plt.ylabel("Target Variable")
    plt.show()


# Predict new data
def predict_new_data(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    print("New Data Prediction:", prediction)
    print("New data Probability:", probability)
    return probability


def execute_model(data_frame):
    data = load_data("cvs2.csv")

    categorical_columns = ['Gender', 'Average number of hours you spend in front of a screen a day?',
                           'Do you use contact lenses?', 'Do you have a history of eye disease and treatment?',
                           'Have you done previous eye surgeries?', 'Do you use monitor filters/blue light filters?',
                           'How often do you take breaks during the use of an electronic device?', 'Room illumination',
                           'Screen brightness', 'Headache', 'Burning eye sensation', 'Eye redness', 'Blurred vision',
                           'Dry eyes (tearing))', 'Neck and shoulder pain', 'Eye strain', 'Tired eyes', 'Sore eyes',
                           'Irritation', 'Poor focusing', 'Double Vision', 'Status of the csv']
    encoded_data = encode_categorical_columns(data, categorical_columns)

    X_train, X_test, y_train, y_test, = split_data(encoded_data)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Support Vector Machines", SVC(probability=True)),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
    ]

    for name, model in models:
        print(f"\n{name}:")
        train_model(model, X_train, y_train)
        # TODO: Uncomment to plot the graphs
        # evaluate_model(model, X_test, y_test)
        predicted_result = predict_new_data(model, data_frame)

    # TODO: Return result of the most accurate model only
    return predicted_result


@app.route("/")
def index():
    return "Welcome to Eye Care Model"


@app.route("/model/prediction", methods=["POST"])
def calculateProbability():
    inp = request.args.get("data")

    if inp is None:
        return jsonify({"error": "Data parameter is missing"}), 400

    inp = [int(float(num)) if float(num).is_integer() else float(num) for num in inp.split(',')]

    columns = ['Age', 'Gender', 'Average number of hours you spend in front of a screen a day?',
               'Do you use contact lenses?', 'Do you have a history of eye disease and treatment?',
               'Have you done previous eye surgeries?', 'Do you use monitor filters/blue light filters?',
               'How often do you take breaks during the use of an electronic device?', 'Room illumination',
               'Screen brightness', 'Average distance from monitor (in cm) ?', 'Headache', 'Burning eye sensation',
               'Eye redness', 'Blurred vision', 'Dry eyes (tearing))', 'Neck and shoulder pain', 'Eye strain',
               'Tired eyes',
               'Sore eyes', 'Irritation', 'Poor focusing', 'Double Vision']
    df = pd.DataFrame([inp], columns=columns)

    try:
        probability = execute_model(df)
        return jsonify({"probability": probability})
    except:
        print("Caught model errors!!!")
        return jsonify({"probability": 0.0})


# TODO: Comment this if running mail
if __name__ == "__main__":
    app.run(debug=True)

# TODO: Comment this if running flask
# if __name__ == "__main__":
#     inp = [27, 0, 3, 0, 0, 0, 1, 3, 3, 3, 60, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
#
#     columns = ['Age', 'Gender', 'Average number of hours you spend in front of a screen a day?',
#                'Do you use contact lenses?', 'Do you have a history of eye disease and treatment?',
#                'Have you done previous eye surgeries?', 'Do you use monitor filters/blue light filters?',
#                'How often do you take breaks during the use of an electronic device?', 'Room illumination',
#                'Screen brightness', 'Average distance from monitor (in cm) ?', 'Headache', 'Burning eye sensation',
#                'Eye redness', 'Blurred vision', 'Dry eyes (tearing))', 'Neck and shoulder pain', 'Eye strain',
#                'Tired eyes',
#                'Sore eyes', 'Irritation', 'Poor focusing', 'Double Vision']
#     df = pd.DataFrame([inp], columns=columns)
#     execute_model(df)
