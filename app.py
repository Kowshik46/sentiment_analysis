from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the zero-shot classifier
classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    statement = data['statement']
    candidate_labels = ["positive", "negative", "neutral"]
    result = classifier(statement, candidate_labels)
    
    # Prepare the scores in a format suitable for the frontend
    scores = [
        {"label": label, "score": score * 100}  # Convert to percentage
        for label, score in zip(result['labels'], result['scores'])
    ]

    prediction = result['labels'][0]  # The label with the highest score
    return jsonify({'prediction': prediction, 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True)
    
