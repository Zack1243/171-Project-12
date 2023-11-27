from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    return request.form.get("workclass")