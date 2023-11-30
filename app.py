# flask --app app run

from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

RF_model = pickle.load(open("RF_model.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    d = {
        "age": [int(request.form.get("age"))],
        "workclass": [request.form.get("workclass")],
        "fnlwgt": [int(request.form.get("fnlwgt"))],
        "education": [request.form.get("education")],
        "educational-num": [int(request.form.get("educational-num"))],
        "marital-status": [request.form.get("marital-status")],
        "occupation": [request.form.get("occupation")],
        "relationship": [request.form.get("relationship")],
        "race": [request.form.get("race")],
        "gender": [request.form.get("gender")],
        "capital-gain": [int(request.form.get("capital-gain"))],
        "capital-loss": [int(request.form.get("capital-loss"))],
        "hours-per-week": [int(request.form.get("hours-per-week"))],
    }
    df = pd.DataFrame(data=d, index=[0])
    columns_to_encode = ['education', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
    encoded_columns = ohe.transform(df[columns_to_encode])
    encoded_df = pd.DataFrame(encoded_columns, columns=ohe.get_feature_names_out(columns_to_encode))
    df_encoded = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)
    # scaled_input = scaler.transform(df_encoded)
    # print(scaled_input)
    pred = RF_model.predict(df_encoded)
    # print(pred)
    if pred == [1]:
        result =  '>50k'
    else:
        result = '<50k'

    #return result

    return render_template('result.html', prediction=result )
    

    