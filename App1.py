from flask import Flask, render_template, request

import joblib

# Load the model
model = joblib.load('reviews_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET','POST'])
def analyze():
    if request.method == 'POST':
         # Get the review from the form
        review = request.form['review']
        
        # Perform sentiment analysis using the loaded model
        sentiment = model.predict([review])[0]
        
        # Map the sentiment to a human-readable label
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
        
        return render_template('results.html', review=review, sentiment=sentiment_label)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
