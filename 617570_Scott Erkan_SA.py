# Chapter 0 - sentiment driven demo

from transformers import pipeline

# results will be shown in the console 
# question-answering model
Scott_model1 = pipeline('question-answering')
question = 'What is my hobby?'
context = 'My name is Scott and my hobby is editing competitive music videos.'
Scott_model1(question = question, context = context)

#%%

# text summerization model 
classifier = pipeline('summarization')
classifier('Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.')

#%%

# text classification model
classifier = pipeline('text-classification', model = 'roberta-large-mnli')
classifier('A soccer game with multiple males playing. Some men are playing a sport.')

#%%

# text translation model
en_fr_translator = pipeline('translation_en_to_fr')
en_fr_translator('How old are you?')

#%%

# full sentiment analysis model
# Chapter 1 - Importing the necessary libraries (installed through anaconda)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import re
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.corpus import stopwords

#%%

# Chapter 2 - Preparing the data 

# Assigning dataset(s) to a dataframe 
train_df = pd.read_csv("C:/Users/scott/Desktop/kaggle/input/Corona_NLP_train.csv",encoding='latin-1')
vdf = pd.read_csv("C:/Users/scott/Desktop/kaggle/input/Corona_NLP_train.csv",encoding='latin-1') # temp visualization df
test_df = pd.read_csv("C:/Users/scott/Desktop/kaggle/input/Corona_NLP_test.csv",encoding='latin-1')

train_df.info()
test_df.info()

#%%

# visualizing the current dataframe to gain insights

vdf['TweetAt'] = pd.to_datetime(vdf['TweetAt'])
vdf.drop_duplicates(subset='OriginalTweet',inplace=True)

# number of tweets by date
tweets_per_day = vdf['TweetAt'].dt.strftime('%m-%d').value_counts().sort_index().reset_index(name='counts')
plt.figure(figsize=(20,5))
ax = sns.barplot(x='index', y='counts', data=tweets_per_day,edgecolor = 'black',errorbar=('ci', False), palette='Oranges')
plt.title('Number of tweets by date')
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()

#%%

# number of tweets by location
tweets_per_country = vdf['Location'].value_counts().loc[lambda x : x > 100].reset_index(name='counts')
plt.figure(figsize=(15,6))
ax = sns.barplot(x='index', y='counts', data=tweets_per_country,edgecolor = 'black',errorbar=('ci', False), palette='PRGn')
plt.title('Number of tweets by location')
plt.xticks(rotation=70)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()

#%%

# showing the five different sentiment labels/categories
plt.figure(figsize=(10,5))
sns.countplot(x=train_df["Sentiment labels"])

#%%

#  Chapter 3 - Preprocessing of the data
# encoding labels

train_inputs = train_df['OriginalTweet'].copy()
test_inputs = test_df['OriginalTweet'].copy()

train_labels = train_df['Sentiment'].copy()
test_labels = test_df['Sentiment'].copy()

sentiment_encoding = {
    'Extremely Negative': 0,
    'Negative': 0,
    'Neutral': 1,
    'Positive': 2,
    'Extremely Positive': 2
}

train_labels = train_labels.replace(sentiment_encoding)
test_labels = test_labels.replace(sentiment_encoding)

#%%

# defining a function for data cleaning

stop_words = set(stopwords.words('english'))

def process_tweet(tweet):
    
    # remove urls
    tweet = re.sub(r'http\S+', ' ', tweet)
    
    # remove html tags
    tweet = re.sub(r'<.*?>', ' ', tweet)
    
    # remove digits
    tweet = re.sub(r'\d+', ' ', tweet)
    
    # remove hashtags
    tweet = re.sub(r'#\w+', ' ', tweet)
    
    # remove mentions
    tweet = re.sub(r'@\w+', ' ', tweet)
    
    #removing stop words
    tweet = tweet.split()
    tweet = [word for word in tweet if word not in stopwords.words('english')]
    
    return tweet
#%%

# applying and tokenizing

train_inputs = train_inputs.apply(process_tweet)
test_inputs = test_inputs.apply(process_tweet)

max_seq_length = np.max(train_inputs.apply(lambda tweet: len(tweet)))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_inputs)

vocab_length = len(tokenizer.word_index) + 1

train_inputs = tokenizer.texts_to_sequences(train_inputs)
test_inputs = tokenizer.texts_to_sequences(test_inputs)

train_inputs = pad_sequences(train_inputs, maxlen=max_seq_length, padding='post')
test_inputs = pad_sequences(test_inputs, maxlen=max_seq_length, padding='post')

# check what kind of tweets we are working with
print("Vocab length:", vocab_length)
print("Max sequence length:", max_seq_length)

#%%

# Chapter 4 - Modeling

train_inputs.shape

embedding_dim = 16

inputs = tf.keras.Input(shape=(max_seq_length,), name='input_layer')

embedding = tf.keras.layers.Embedding(
    input_dim=vocab_length,
    output_dim=embedding_dim,
    input_length=max_seq_length,
    name='word_embedding'
)(inputs)

gru_layer = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(units=256, return_sequences=True, name='gru_layer'),
    name='bidirectional_layer'
)(embedding)

#%%

max_pooling = tf.keras.layers.GlobalMaxPool1D(name='max_pooling')(gru_layer)

dropout_1 = tf.keras.layers.Dropout(0.4, name='dropout_1')(max_pooling)

dense = tf.keras.layers.Dense(64, activation='relu', name='dense')(dropout_1)

dropout_2 = tf.keras.layers.Dropout(0.4, name='dropout_2')(dense)

outputs = tf.keras.layers.Dense(3, activation='softmax', name='output_layer')(dropout_2)

#%%

model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

tf.keras.utils.plot_model(model)

#%%

# Chapter 5 - Training model

# NOTE: this chuck of code might take an extremely long time to finish depening on your setup
# if it is not possible for you to run it succesfully I suggest reducing the batch size (accuracy trade-off)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

batch_size = 32
epochs = 2

history = model.fit(
    train_inputs,
    train_labels,
    validation_split=0.12,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2
)

#%%

# Chapter 6 - Results & conclusion

model.evaluate(test_inputs, test_labels)
#[0.3488548994064331, 0.8788836002349854]
# There is an accuracy of 88 percent rounded up
# The accuracy could potentially be improved by deep cleaning the data taking 
# emojis into account, and by testing other models such as BERT or RoBERTa