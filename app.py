from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import SpamClassifier

app = Flask(__name__)

# Load model once when app starts
classifier = SpamClassifier()

@app.route("/", methods=["GET"])
def index():
    return render_template("home.html", prediction=None)

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    prediction = None
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            prediction = classifier.predict(message)
    return render_template("home.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

 