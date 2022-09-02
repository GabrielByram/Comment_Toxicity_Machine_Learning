import numpy as np
import pandas as pd
import csv

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import seaborn as sns

from wordcloud import WordCloud
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input,  Activation
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, optimizers, layers
from sklearn.metrics import roc_auc_score

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize 
print(stopwords.words('english'))
import re                                   # library for regular expression operations
import string                               # for string operations
from nltk.stem import PorterStemmer         # module for stemming
from nltk.tokenize import regexp_tokenize   # module for tokenizing strings
from nltk.tokenize import TreebankWordTokenizer

# define functions to clean the text
def cleanWords(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# define function to remove stop words
def removeStopWords(text): 
   tokenizer = TreebankWordTokenizer()
   comment_tokens = tokenizer.tokenize(text)

   newText = [word for word in comment_tokens if word not in stopWords]
   return newText
 
# displays sample of data before parsing
# then removes stop words and puncuation from training and test data

def parseData(train, test):
   # print out the first 10 elements in the training and test data
   print("First 10 training comments before parsing data: \n")
   print(train.head(10))
   print("First 10 testing comments before parsing data: \n")
   print(test.head(10))

   print("\n\n")

   train['comment_text'] = train['comment_text'].apply(lambda x: cleanWords(x))
   test['comment_text'] = test['comment_text'].apply(lambda x: cleanWords(x))

   # clean stop words from train and test
   train['comment_text'] = train['comment_text'].apply(lambda x: removeStopWords(x))
   test['comment_text'] = test['comment_text'].apply(lambda x: removeStopWords(x))

   # show updated values of training and testing data
   print("First 10 training comments after parsing data: \n")
   print(train.head(10))
   print("First 10 testing comments after parsing data: \n")
   print(test.head(10))

   print("\n\n")

# creates tensorflow model from training results and test the model
def trainModel(train, test):
   columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
   targets = train[columns].values

   trainingDF = train['comment_text']
   testingDF = test['comment_text']

   # define info for model

   maxFeatureNumbers = 22000

   # tokenize the training and testing data
   tokenizer = Tokenizer(num_words = maxFeatureNumbers)
   tokenizer.fit_on_texts(list(trainingDF))
   tokenizedTrain = tokenizer.texts_to_sequences(trainingDF)
   tokenizedTest = tokenizer.texts_to_sequences(testingDF)

   # prepare tokenized training and testing data for model
   maxLength = 200
   Xtrain = pad_sequences(tokenizedTrain, maxlen = maxLength)
   XTest = pad_sequences(tokenizedTest, maxlen = maxLength)

   embeddingSize = 128
   maxLength = 200
   maxFeatureNumbers = 22000

   # prepare input for model
   commentInput = Input(shape = (maxLength, ))
   x = Embedding(maxFeatureNumbers, embeddingSize)(commentInput)
   x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
   x = GlobalMaxPool1D()(x)
   x = Dropout(0.1)(x)
   x = Dense(50, activation="relu")(x)
   x = Dropout(0.1)(x)
   x = Dense(6, activation="sigmoid")(x)

   model = Model(inputs=commentInput, outputs=x)
   model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

   # notifies user of model info
   print(model.summary())

   print("\nNow we will train our model using Tensorflow. This will take a little while.\n")
   # trains model based on training input and testing data
   batchSize = 64
   epochs = 2
   print(model.fit(Xtrain, targets, batch_size=batchSize, epochs=epochs, validation_split=0.1))

# define function to display the bar chart
def displayBarplot(comments):
   # adjust matploplib display size
   plt.figure(figsize=(10,6))

   # make barplot in seaborn
   snsBarplot = sns.barplot(comments.index, comments.values, alpha=0.5)

   # adjust matplotlib display info
   plt.title("Comment Amount by Type")
   plt.xlabel("Comment Type")
   plt.ylabel("Number of Comments")

   barplotRectangles = snsBarplot.patches
   barplotCategories = comments.values
   for plot, category in zip(barplotRectangles, barplotCategories):
      plotHeight = plot.get_height()
      snsBarplot.text(plot.get_x() + plot.get_width()/3, plotHeight+7, category, ha="center", va="bottom")

   plt.show()

# importing the datasets
train=pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
test=pd.read_csv("jigsaw-toxic-comment-classification-challenge/test.csv")

# display stop words
print("Here is a list of all the generic words we will be cleaning from our data")
stopWords = stopwords.words('english')

# display a graph of all the different number of comment types
cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
comments = train[cols].sum()
print("Now displaying a barplot of the toxic comment categories in our training data.\n\n")
displayBarplot(comments)

print("\n\n")

# word clouds of each comment category in training data

# make a string of all the words in each toxic comment category
toxicWords = ' '.join([text for text in train[train["toxic"] == 1]["comment_text"]])
severeToxicWords = ' '.join([text for text in train[train["severe_toxic"] == 1]["comment_text"]])
obsceneWords = ' '.join([text for text in train[train["obscene"] == 1]["comment_text"]])
threatWords = ' '.join([text for text in train[train["threat"] == 1]["comment_text"]])
insultWords = ' '.join([text for text in train[train["insult"] == 1]["comment_text"]])
identityHateWords = ' '.join([text for text in train[train["identity_hate"] == 1]["comment_text"]])
nonToxicWords = ' '.join([text for text in train[train["toxic"] == 0]["comment_text"]])

print("Now we will process the data for our word clouds. This will take about one minute.\n")
# NON-TOXIC
word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       margin=0,
                       max_words=500, # maximum numbers of words we want to see 
                       min_word_length=3, # minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(nonToxicWords)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Non-Toxic Comments', fontsize = 40)
plt.axis("off")
print("Now displaying a word cloud for the non-toxic comments.\n")
plt.show()

# TOXIC
print("Now displaying a word cloud for the toxic comments.\n")
word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # maximum numbers of words we want to see 
                       min_word_length=3, # minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(toxicWords)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Toxic Comments', fontsize = 40)
plt.axis("off")
plt.show()

# SEVERE TOXIC
print("Now displaying a word cloud for the severe toxic comments.\n")
word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(severeToxicWords)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Severe Toxic Comments', fontsize = 40)
plt.axis("off")
plt.show()

# OBSCENE
print("Now displaying a word cloud for the obscene comments.\n")
word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(obsceneWords)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Obscene Comments', fontsize = 40)
plt.axis("off")
plt.show()

# THREAT
print("Now displaying a word cloud for the threat comments.\n")

word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # maximum numbers of words we want to see 
                       min_word_length=3, # minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(threatWords)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Threat Comments', fontsize = 40)
plt.axis("off")
plt.show()

# INSULT
print("Now displaying a word cloud for the insult comments.\n")
word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # maximum numbers of words we want to see 
                       min_word_length=3, # minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(insultWords)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Insult Comments', fontsize = 40)
plt.axis("off")
plt.show()

# IDENTITY HATE
print("Now displaying a word cloud for the identity hate comments.\n")
word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # maximum numbers of words we want to see 
                       min_word_length=3, # minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(identityHateWords)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Identity Hate Comments', fontsize = 40)
plt.axis("off")
plt.show()

# parses and cleans the training and testing data
print("Now parsing and cleaning the training and testing data.")
print("This will take around 5 minutes.")
parseData(train, test)

print("\n\n")
# train model based on the training data and test the model
model = trainModel(train, test)

# model info and accuracy given inside trainModel when run