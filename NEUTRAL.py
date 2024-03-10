from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')
set(stopwords.words('english'))

sa = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/csv_file', methods=['POST'])  
def csv_file():  
    if request.method == 'POST':  
        f = request.files['file']
        print(f.filename)
        df = pd.read_csv(f.filename)
        reviews = df['Review'].to_list()

        sentiment_pos, sentiment_neg = [], []
        for rev in reviews:
            text_final = ''.join(c for c in rev if not c.isdigit())
            processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])
            dd = sa.polarity_scores(text=processed_doc1)
            if dd['compound'] > 0:
                sentiment_pos.append(rev)
            elif dd['compound'] < 0:
                sentiment_neg.append(rev)

        print("Positive Comments:")
        for comment in sentiment_pos:
            print(comment)
        
        print("\nNegative Comments:")
        for comment in sentiment_neg:
            print(comment)

        return render_template("dataprediction.html")

@app.route('/', methods=['POST'])
def my_form_post():
    # convert to lowercase
    text1 = request.form['text1'].lower()
    
    # remove digits
    text_final = ''.join(c for c in text1 if not c.isdigit())
        
    # remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)
    print(dd, compound)

    if compound > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return render_template('form.html', final=sentiment, text1=text_final, text2=dd['pos'], text5=dd['neg'], text4=compound)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5003, threaded=True)
