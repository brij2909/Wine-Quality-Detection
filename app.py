# Flask Libraries
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

# Home page for app
@app.route("/")
def home():
    return render_template("mlwine.html")

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        float_feature = [float(x) for x in request.form.values()]

        if len(float_feature) != 11:
            return render_template("mlwine.html", prediction_text="Invalid number of features. Please provide all required features.")

        features = [np.array(float_feature)]

        predictions = model.predict(features)

        prediction_text = "Bad Quality Wine." if predictions[0] == 0 else "Good Quality Wine."

        return render_template("mlwine.html", prediction_text=prediction_text)

    return render_template("mlwine.html")

if __name__ == "__main__":
    app.run(debug=True)
