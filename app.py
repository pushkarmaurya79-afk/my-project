from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    cpu = float(request.form["cpu"])
    memory = float(request.form["memory"])
    disk = float(request.form["disk"])

    prediction = model.predict([[cpu, memory, disk]])

    return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)