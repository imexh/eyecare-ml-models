import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

loaded_model = load_model("models/CVS_Boostrap_Feedforward.h5")


@app.route("/")
def index():
    return "Welcome to Eye Care ML Models service!!!"


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
               'Tired eyes', 'Sore eyes', 'Irritation', 'Poor focusing', 'Double Vision']
    df = pd.DataFrame([inp], columns=columns)

    try:
        probability = loaded_model.predict(df)[0][0]
        return jsonify({"probability": float(probability)})
    except:
        print("Caught model errors!!!")
        return jsonify({"probability": 0.0})


# TODO: Comment this when running the main
# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=False, port=8081, host='0.0.0.0')

# TODO: Comment this when running flask
# if __name__ == "__main__":
#     inp = [27, 0, 3, 0, 0, 0, 1, 3, 3, 3, 60, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
#
#     columns = ['Age', 'Gender', 'Average number of hours you spend in front of a screen a day?',
#                'Do you use contact lenses?', 'Do you have a history of eye disease and treatment?',
#                'Have you done previous eye surgeries?', 'Do you use monitor filters/blue light filters?',
#                'How often do you take breaks during the use of an electronic device?', 'Room illumination',
#                'Screen brightness', 'Average distance from monitor (in cm) ?', 'Headache', 'Burning eye sensation',
#                'Eye redness', 'Blurred vision', 'Dry eyes (tearing))', 'Neck and shoulder pain', 'Eye strain',
#                'Tired eyes', 'Sore eyes', 'Irritation', 'Poor focusing', 'Double Vision']
#     df = pd.DataFrame([inp], columns=columns)
#     print(loaded_model.predict(df)[0][0])
