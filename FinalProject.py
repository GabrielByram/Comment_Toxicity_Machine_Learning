import numpy as np
import pandas as pd
import csv

""" from google.colab import drive
drive.mount('/content/gdrive') """

#Global parameters
exclude_stop_words = True

stopWords = 

def formVocab():
  with open('/content/gdrive/My Drive/CptS315Project/train.csv')
  csv_reader = csv.reader(csv.file)
  lin_num = 0
  for row in csv_reader:
     if line_num > 0: # skip header
        #row.split(',')