from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests

app = Flask(__name__)

roberta_model = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta_model)
tokenizer = AutoTokenizer.from_pretrained(roberta_model)
labels = ['Negative', 'Neutral', 'Positive']

def get_ip_info(ip_address):
    try:
        response = requests.get(f"https://ipinfo.io/{ip_address}/json")
        return response.json()
    except Exception as e:
        return None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Get client IP address
        client_ip = request.remote_addr
        ip_info = get_ip_info(client_ip)

        # Preprocess the text
        tweet_words = []
        for word in text.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            tweet_words.append(word)
        processed_text = " ".join(tweet_words)

        # Sentiment analysis
        encoded_text = tokenizer(processed_text, return_tensors='pt')
        output = model(**encoded_text)

        scores = softmax(output[0][0].detach().numpy())

        # Display the probabilities for each sentiment class
        sentiment_probabilities = dict(zip(labels, scores))

        # Check if negative sentiment is more than positive and neutral
        if sentiment_probabilities['Negative'] > sentiment_probabilities['Positive'] and sentiment_probabilities['Negative'] > sentiment_probabilities['Neutral']:
            alert_message = "This is a negative sentiment!"
        else:
            alert_message = None

        return render_template('index.html', text=text, probabilities=sentiment_probabilities, alert_message=alert_message, ip_info=ip_info)

if __name__ == '__main__':
    app.run(debug=True)
