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

@app.route('/csv_file', methods = ['POST'])  
def csv_file():  
    if request.method == 'POST':  
        f = request.files['file']
        print(f.filename)
        df = pd.read_csv(f.filename)
        reviews = df['Review'].to_list()

        sentiment_pos, sentiment_neg, sentiment_neu = [], [], []
        for rev in reviews:
            text_final = ''.join(c for c in rev if not c.isdigit())
            processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])
            dd = sa.polarity_scores(text=processed_doc1)
            sentiment_pos.append(dd['pos'])
            sentiment_neg.append(dd['neg'])
            sentiment_neu.append(dd['neu'])

        df['Positive sentiment score'] = sentiment_pos
        df['Negative sentiment score'] = sentiment_neg
        df['Neutral sentiment score'] = sentiment_neu

        df = df[['Review', 'Positive sentiment score', 'Negative sentiment score', 'Neutral sentiment score']]
        print(df.shape)
        return render_template("dataprediction.html", data=df.to_html())  
        #return render_template("dataprediction.html", tables=[df.to_html(classes='data')], titles=df.columns.values)  


@app.route('/', methods=['POST'])
def my_form_post():
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())
    
    #remove punctuations
    #text3 = ''.join(c for c in text2 if c not in punctuation)
        
    #remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)
    print(dd, compound)
    return render_template('form.html', final=compound, text1=text_final,text2=dd['pos'],text5=dd['neg'],text4=compound,text3=dd['neu'])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
