import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create a Flask app
app = Flask(__name__, template_folder="/Users/krishnasaialvakonda/Downloads/LPD")

# Load pickle models
scalerp = pickle.load(open("scaler.pkl", "rb"))
knnmodel = pickle.load(open("knnmodel.pkl", "rb"))
adamodel = pickle.load(open("adamodel.pkl", "rb"))
annmodel = pickle.load(open("annmodel.pkl", "rb"))
recommendations = {
    "total_bilirubin": {
        "low": ["carrots", "spinach", "beets", "tomatoes"],
        "high": ["avoid alcohol", "avoid fatty foods", "avoid processed foods"]
    },
    "direct_bilirubin": {
        "low": ["eggs", "bananas", "avocado", "sweet potatoes"],
        "high": ["avoid alcohol", "avoid fatty foods", "avoid processed foods"]
    },
    "alk_phos": {
        "low": ["milk", "cheese", "yogurt", "broccoli"],
        "high": ["avoid alcohol", "avoid fatty foods", "avoid processed foods"]
    },
    "alt": {
        "low": ["oats", "walnuts", "eggs", "green leafy vegetables"],
        "high": ["avoid alcohol", "avoid fatty foods", "avoid processed foods"]
    },

    "total_protein": {
        "low": ["chicken breast", "fish", "lean beef", "greek yogurt"],
        "high": ["low-fat dairy products", "tofu", "beans", "leafy green vegetables"]
    },
    "albumin": {
        "low": ["eggs", "mushrooms", "cheese", "lean meat"],
        "high": ["leafy green vegetables", "citrus fruits", "nuts", "whole grains"]
    },
    "ag_ratio": {
        "low": ["seafood", "oats", "brown rice", "broccoli"],
        "high": ["turmeric", "garlic", "ginger", "spinach"]
    }
}


def recommendation(features, gender):
    s = ""
    if gender == "male":
        if features[2] < 0.3:
            s += "Total bilirubin level is low. You should consume more food such as:"
            s += ", ".join(recommendations["total_bilirubin"]["low"]) + ".\n\n"
        elif features[2] > 1.2:
            s += "Total bilirubin level is high. Avoid foods high in fat  such as "
            s += ", ".join(recommendations["total_bilirubin"]["high"]) + ".\n\n"

        if features[3] < 0.1:
            s += "Direct bilirubin level is low. You should consume more food such as:"
            s += ", ".join(recommendations["direct_bilirubin"]["low"]) + ".\n\n"
        elif features[3] > 0.3:
            s += "Direct bilirubin level is High. Avoid foods high in fat  and consult a doctor."
            s += ", ".join(recommendations["direct_bilirubin"]["high"]) + ".\n\n"

        if features[4] < 40:
            s += "Alkaline phosphotransferase level is low. Consume more dairy products, nuts and seeds, and green " \
                 "leafy vegetables. "
            s += ", ".join(recommendations["alk_phos"]["low"]) + ".\n\n"
        elif features[4] > 129:
            s += "Alkaline phosphotransferase level is high.\n Avoid alcohol and fatty foods, and consult a doctor."
            s += ", ".join(recommendations["alk_phos"]["high"]) + ".\n\n"

        if features[5] < 10:
            s += " Alamine Aminotransferase level is low\n. Consume more "
            s += ", ".join(recommendations["alt"]["low"]) + ".\n\n"
        elif features[5] > 40:
            s += " Alamine Aminotransferase level is high. Avoid alcohol and fatty foods, and consult a doctor."
            s += ", ".join(recommendations["alt"]["high"]) + "\n\n"

        if features[6] < 6.1:
            s += " Total proteins level is low. Consume more meat, fish, eggs, and dairy products. "
            s += ", ".join(recommendations["total_protein"]["low"]) + ".\n\n"
        elif features[6] > 8.1:
            s += " Total proteins level is high. \nConsume less meat and dairy products and consult a doctor."
            s += ", ".join(recommendations["total_protein"]["high"]) + ".\n\n"

        if features[7] < 3.9:
            s += " Albumin is low. Consume more "
            s += ", ".join(recommendations["albumin"]["low"]) + ".\n\n"
        elif features[7] > 5:
            s += "Albumin is high. Consume less"
            s += ", ".join(recommendations["albumin"]["high"]) + ".\n\n"

        if features[8] < 1.0:
            s += " Albumin and Globulin ratio level is low. Consume more "
            s += ", ".join(recommendations["ag_ratio"]["low"]) + ".\n\n"
        elif features[8] > 1.6:
            s += " Albumin and Globulin ratio level is high.Consume less "
            s += ", ".join(recommendations["ag_ratio"]["high"]) + ".\n\n"

    if gender == "female":
        if features[2] < 0.2:
            s += "Total bilirubin level is low. You should consume more food such as:"
            s += ", ".join(recommendations["total_bilirubin"]["low"]) + ".\n\n"
        elif features[2] > 1.2:
            s += "Total bilirubin level is high. Avoid foods high in fat  such as "
            s += ", ".join(recommendations["total_bilirubin"]["high"]) + ".\n\n"

        if features[3] < 0.1:
            s += "Direct bilirubin level is low. You should consume more food such as:"
            s += ", ".join(recommendations["direct_bilirubin"]["low"]) + ".\n\n"
        elif features[3] > 0.3:
            s += "Direct bilirubin level is High. Avoid foods high in fat  and consult a doctor."
            s += ", ".join(recommendations["direct_bilirubin"]["high"]) + ".\n\n"

        if features[4] < 35:
            s += "Alkaline phosphotransferase level is low. Consume more dairy products, nuts and seeds, and green " \
                 "leafy vegetables. "
            s += ", ".join(recommendations["alk_phos"]["low"]) + ".\n\n"
        elif features[4] > 105:
            s += "Alkaline phosphotransferase level is high. Avoid alcohol and fatty foods, and consult a doctor."
            s += ", ".join(recommendations["alk_phos"]["high"]) + ".\n\n"

        if features[5] < 7:
            s += " Alamine Aminotransferase level is low. Consume more "
            s += ", ".join(recommendations["alt"]["low"]) + ".\n\n"
        elif features[5] > 35:
            s += " Alamine Aminotransferase level is high. Avoid alcohol and fatty foods, and consult a doctor."
            s += ", ".join(recommendations["alt"]["high"]) + ".\n\n"

        if features[6] < 6.1:
            s += " Total proteins level is low. Consume more meat, fish, eggs, and dairy products. "
            s += ", ".join(recommendations["total_protein"]["low"]) + ".\n\n"
        elif features[6] > 8.1:
            s += " Total proteins level is high. Consume less meat and dairy products and consult a doctor."
            s += ", ".join(recommendations["total_protein"]["high"]) + ".\n\n"

        if features[7] < 3.9:
            s += " Albumin is low. Consume more "
            s += ", ".join(recommendations["albumin"]["low"]) + ".\n\n"
        elif features[7] > 5:
            s += "Albumin is high. Consume less"
            s += ", ".join(recommendations["albumin"]["high"]) + ".\n\n"

        if features[8] < 1.0:
            s += " Albumin and Globulin Ratio level is low. Consume more "
            s += ", ".join(recommendations["ag_ratio"]["low"]) + ".\n\n"
        elif features[8] > 1.6:
            s += " Albumin and Globulin Ratio level is high. Consume less "
            s += ", ".join(recommendations["ag_ratio"]["high"]) + ".\n\n"

    return s + "All other values are within normal range. We hope that you stay healthy and safe by following the " \
               "aforementioned guidelines. Wishing you good health. "


@app.route("/")
def Hello():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    features = [x for x in request.form.values()]
    gender = features.pop(1).lower()
    features = list(map(float, features))
    req_fea = np.array(
        [int(features[0]), features[1], int(features[3]), int(features[4]), int(features[5]), features[6]])
    req_fea = req_fea.reshape(1, 6)
    req_fea = scalerp.transform(req_fea)
    knn, ada, ann = knnmodel.predict(req_fea)[0], adamodel.predict(req_fea)[0], 1 if annmodel.predict(
        req_fea) == float(1) else 2
    print(knn, ada, ann)
    a, ans = knn, None
    if ada == a:
        ans = a
    elif ann == a:
        ans = a
    else:
        ans = ada
    b = recommendation(features, gender).replace('\n\n', '\n').split('\n')
    #print(ans)
    return render_template('predict.html', predict_content="You have Disease" if ans == 1 else "You don't have "
                                                                                               "Disease",
                           recommend=b)


if __name__ == "__main__":
    app.run(debug=True)
