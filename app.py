from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    prediction = model.predict(
        [
            [
                int(request.form.get("age")),
                request.form.get("workclass"),
                int(request.form.get("fnlwgt")),
                request.form.get("education"),
                int(request.form.get("educational-num")),
                request.form.get("marital-status"),
                request.form.get("occupation"),
                request.form.get("relationship"),
                request.form.get("race"),
                request.form.get("gender"),
                int(request.form.get("capital-gain")),
                int(request.form.get("capital-loss")),
                int(request.form.get("hours-per-week"))
            ]
        ]
    )
    return prediction
