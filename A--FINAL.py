#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_context('poster')


# In[27]:


df=pd.read_csv("ford.csv",lineterminator='\n')


# In[28]:


df


# In[29]:


df.columns


# In[30]:


dict = {'Unnamed: 0': 'sno',
        'Rating\r': 'Rating'}
 
# call rename () method
df.rename(columns=dict,
          inplace=True)
 
# print Data frame after rename columns
display(df)


# In[31]:


df = df.drop(['Review_Date', 'sno', 'Review_Title'], axis=1)


# In[32]:


df.dtypes


# In[33]:


df['Rating'] = df['Rating'].astype(int)


# In[34]:


df


# In[35]:


df['sentiment'] = 0
df.loc[df['Rating'] > 3, 'sentiment'] = 1
df.loc[df['Rating'] <= 3, 'sentiment'] = 0


# In[36]:


sentiments = ['Positive Sentiment', 'Negetive Sentiment']

slices = [(df['sentiment'] == 1).sum(), (df['sentiment'] == 0).sum()] 
colors = ['r', 'b'] 
plt.pie(slices, labels = sentiments, colors=colors, startangle=50, shadow = True,
        explode = (0,0), radius = 2.5, autopct = '%1.2f%%') 
plt.legend()
plt.show()


# In[37]:


sns.countplot(df['Rating'])    
df['Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Count')


# In[38]:


#NB MODEL
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(stemmed_tokens)
df['Review'] = df['Review'].apply(preprocess)
df
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Rating'], test_size=0.2, train_size=0.8, random_state=42, stratify=df['sentiment'])

# vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train the Multinomial Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# predict the sentiment of the text data in the testing set
y_pred = nb.predict(X_test_vec)

# compute accuracy and confusion matrix
accuracy_nb = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[39]:


print('Accuracy:', accuracy_nb)
print('Confusion matrix:')
print(conf_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

# create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# create heatmap with annotations
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

# set axis labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[40]:


positive_reviews = df[df['sentiment']==1].groupby('Vehicle_Title')['sentiment'].sum()

# Get the title of the vehicle with the most positive reviews
vehicle_with_most_positive_reviews = positive_reviews.idxmax()

# Print the result
print('The vehicle with the most positive reviews is', vehicle_with_most_positive_reviews)


# In[41]:


vehicle_title = r'2010 Ford Fusion Hybrid Sedan 4dr Sedan \(2\.5L 4cyl gas/electric hybrid CVT\)'

# Filter the dataset by the vehicle title using boolean indexing
df_vehicle = df[df['Vehicle_Title'].str.contains(vehicle_title)]


# In[42]:


df_vehicle 


# In[43]:


vehicle_of_interest = '2010 Ford Fusion Hybrid Sedan 4dr Sedan (2.5L 4cyl gas/electric hybrid CVT)'

# Filter the dataset by the vehicle title and count the number of rows
total_reviews = len(df[df['Vehicle_Title']==vehicle_of_interest])
positive_reviews = len(df[(df['Vehicle_Title']==vehicle_of_interest) & (df['sentiment']==1)])
negative_reviews = len(df[(df['Vehicle_Title']==vehicle_of_interest) & (df['sentiment']==0)])

print('Total number of reviews for', vehicle_of_interest + ':', total_reviews)
print('Number of positive reviews for', vehicle_of_interest + ':', positive_reviews)
print('Number of negative reviews for', vehicle_of_interest + ':', negative_reviews)


# In[19]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# Select the rows that correspond to the desired vehicle
df_vehicle = df[df['Vehicle_Title'].str.extract('2010 Ford Fusion Hybrid Sedan 4dr Sedan \((2.5L 4cyl gas/electric hybrid CVT)\)', expand=False).notnull()]

# Split the data into training and testing sets
train_data = df_vehicle.sample(frac=0.8, random_state=42)
test_data = df_vehicle.drop(train_data.index)

# Vectorize the review text using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data["Review"])
X_test = vectorizer.transform(test_data["Review"])

# Train a Naive Bayes classifier on the training data
clf = MultinomialNB()
y_train = train_data["Rating"]
clf.fit(X_train, y_train)

# Predict the accuracy on the test data
y_test = test_data["Rating"]
accuracy_Vnb = clf.score(X_test, y_test)
print("Accuracy:", accuracy_Vnb)


# In[20]:


# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(stemmed_tokens)
df['Review'] = df['Review'].apply(preprocess)
df

# Extract the features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Review"])
y = df["Rating"]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the SVM model
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)
# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy_svm = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy_svm )
print('Confusion matrix:')
print(conf_matrix)


# In[ ]:





# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Extract the reviews for the given vehicle
df_vehicle = df[df['Vehicle_Title'] == '2010 Ford Fusion Hybrid Sedan 4dr Sedan (2.5L 4cyl gas/electric hybrid CVT)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_vehicle['Review'], df_vehicle['Rating'], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert the text into a bag of words
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectorized, y_train)

# Test the model on the testing set
X_test_vectorized = vectorizer.transform(X_test)
y_pred = svm_model.predict(X_test_vectorized)

# Compute the accuracy and classification report
accuracy_Vsvm = accuracy_score(y_test, y_pred)


print("Accuracy:", accuracy_Vsvm)


# In[22]:


#VADER ALGORITHM
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Add a new column to store the sentiment scores
df['sentiment_score'] = 0

# Iterate over the reviews and calculate the sentiment score
for i, review in enumerate(df['Review']):
    sentiment_score = analyzer.polarity_scores(review)['compound']
    df.loc[i, 'sentiment_score'] = sentiment_score

# Convert the sentiment scores to sentiment labels
df.loc[:, 'sentiment_label'] = df['sentiment_score'].apply(lambda x: 1 if x >= 0.05 else (0 if x <= -0.05 else None))

# Remove rows with missing sentiment labels
df.dropna(subset=['sentiment_label'], inplace=True)

# Calculate the accuracy of the sentiment analysis for the dataset
accuracy_Vr = (df['sentiment_label'] == df['sentiment']).mean()

# Print the accuracy
print('Accuracy:', accuracy_Vr)
print(classification_report(df['sentiment'], df['sentiment_label']))

# Plot the distribution of sentiment labels
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='sentiment', data=df)
plt.show()


# In[44]:


import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Add a new column to store the sentiment scores
df['sentiment_score'] = 0

# Iterate over the reviews and calculate the sentiment score
for i, review in enumerate(df['Review']):
    sentiment_score = analyzer.polarity_scores(review)['compound']
    df.loc[i, 'sentiment_score'] = sentiment_score

# Filter the dataframe to only contain reviews for a specific vehicle model and trim
df_vehicle = df[df['Vehicle_Title'].str.contains('2010 Ford Fusion Hybrid Sedan 4dr Sedan \((2.5L 4cyl gas/electric hybrid CVT)\)')].copy()

# Convert the sentiment scores to sentiment labels
df_vehicle.loc[:, 'sentiment_label'] = df_vehicle['sentiment_score'].apply(lambda x: 1 if x >= 0.05 else (0 if x <= -0.05 else None))

# Remove rows with missing sentiment labels
df_vehicle.dropna(subset=['sentiment_label'], inplace=True)

# Calculate the accuracy of the sentiment analysis for the specific vehicle
accuracy_vVr = (df_vehicle['sentiment_label'] == df_vehicle['sentiment']).mean()

# Print the accuracy
print('Accuracy:', accuracy_vVr)
print(classification_report(df_vehicle['sentiment'], df_vehicle['sentiment_label']))

# Convert the 'sentiment' column to a categorical variable
df_vehicle['sentiment'] = df_vehicle['sentiment'].astype('category')

# Plot the distribution of sentiment labels
sns.countplot(x='sentiment', data=df_vehicle)
plt.show()


# In[45]:


import matplotlib.pyplot as plt
import numpy as np
 
# Data
algorithms = ['SVM', 'NB', 'VADER']
accuracy = [accuracy_svm, accuracy_nb, accuracy_Vr]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Round accuracy values to two decimal places
accuracy = [round(a, 2) for a in accuracy]
 
# Create bar plot
fig, ax = plt.subplots(figsize=(8,6))
bars = ax.bar(algorithms, accuracy, color=colors)

# Add text labels for each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=16)

# Set axis labels and title
ax.set_ylabel('Accuracy (%)', fontsize=18)
ax.set_xlabel('Algorithm', fontsize=18)
ax.set_title('Comparison of Sentiment Analysis Algorithms', fontsize=20)

# Set font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=16)

plt.show()


# In[46]:


import matplotlib.pyplot as plt
import numpy as np
 
# Data
algorithms = ['SVM', 'NB', 'VADER']
accuracy = [accuracy_Vsvm,accuracy_Vnb,accuracy_vVr]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

accuracy = [round(a, 2) for a in accuracy]
 
# Create bar plot
fig, ax = plt.subplots(figsize=(8,6))
bars = ax.bar(algorithms, accuracy, color=colors)

# Add text labels for each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=16)

# Set axis labels and title
ax.set_ylabel('Accuracy (%)', fontsize=18)
ax.set_xlabel('Algorithm', fontsize=18)
ax.set_title('Comparison of Specific Vehicles Sentiment Analysis Algorithms', fontsize=20)

# Set font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=16)

plt.show()


# In[103]:


from tabulate import tabulate

data = [['SVM', accuracy_svm],
        ['Naive Bayes', accuracy_nb],
        ['VADER', accuracy_Vr]]

title = 'Accuracy Scores for Algorithms'
print(title.center(30))
print(tabulate(data, headers=['Model', 'Testing Accuracy'], tablefmt='orgtbl'))


# In[99]:


from tabulate import tabulate

data = [['SVM', accuracy_Vsvm],
        ['Naive Bayes', accuracy_Vnb],
        ['VADER', accuracy_vVr]]

title = 'Accuracy Scores for most positve rated vehicle '
print(title.center(30))
print(tabulate(data, headers=['Model', 'Testing Accuracy'], tablefmt='orgtbl'))


# In[ ]:




