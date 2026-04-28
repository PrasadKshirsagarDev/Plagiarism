from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("marathi_ai_detector.pkl", "rb"))
vectorizer = pickle.load(open("marathi_vectorizer.pkl", "rb"))

def split_marathi_sentences(text):
    sentences = re.split(r'[।.!?]', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    paragraph = ""

    if request.method == "POST":
        paragraph = request.form["paragraph"]
        sentences = split_marathi_sentences(paragraph)

        for s in sentences:
            vec = vectorizer.transform([s])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][1]

            results.append({
                "sentence": s,
                "writer": "AI Written" if pred == 1 else "Human Written",
                "probability": round(prob, 3)
            })

    return render_template("index.html", results=results, paragraph=paragraph)

if __name__ == "__main__":
    app.run(debug=True)
