from flask import Flask, render_template, request, jsonify
import pickle

# Corrected file paths for loading vectorizer and model
cv = pickle.load(open("models/cv.pkl", "rb"))  # Ensure "models" folder exists and contains "cv.pkl"
model = pickle.load(open("models/clf.pkl", "rb"))  # Ensure "models" folder contains "clf.pkl"

app = Flask(__name__)

@app.route("/")
def home():
    # Fixed undefined variable `text` by setting a default value
    return render_template("index.html", text="")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        email_text = request.form.get("email_content")
    # Retrieve email content from form
    email_text = request.form.get("email-content")
    
    if not email_text:  # Handle case when input is empty
        return render_template("index.html", prediction="No content provided.", text="")

    # Corrected `tokenizer` to use `cv` (loaded CountVectorizer object)
    tokenized_email = cv.transform([email_text])  # Wrapped `email_text` in a list
    
    # Predict using the loaded model and handle prediction format
    predictions = model.predict(tokenized_email)[0]  # Get the single prediction
    
    # Adjust predictions to 1 or -1
    predictions = 1 if predictions == 1 else -1

    return render_template("index.html", prediction=predictions, text=email_text)

@app.route("/api/preict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    email_text = data["content"]
    tokenized_email = cv.transform([email_text])  # Wrapped `email_text` in a list
    
    # Predict using the loaded model and handle prediction format
    predictions = model.predict(tokenized_email)[0]  # Get the single prediction
    
    # Adjust predictions to 1 or -1
    predictions = 1 if predictions == 1 else -1

    return jsonify({prediction: prediction})

if __name__ == "__main__":
    app.run(debug=True)
