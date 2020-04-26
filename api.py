from flask import Flask, jsonify, request, render_template
import pickle
# Import from app/features.py.
# from features import FEATURES
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import pandas as pd



# Initialize the app and set a secret_key.
app = Flask(__name__)
app.secret_key = '123'

import numpy as np
import glob
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Lambda 
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.models import load_model
import sys

np.random.seed(7)
X_train = []
Y_train = []
filenames_txt = glob.glob('*.txt') 
filenames_fna = glob.glob('*.fna')

def process_file(filename): 
  # input: str filename 
  
  # Sets y_i to 1 if it starts with MN and 0 otherwise. 
  y_i = 0
  if filename[:2] == 'MN':  
    y_i = 1

  file = open(filename,encoding="latin-1") # opens the file
  string = file.read() # this takes the file and reads it 
  string_split = string.split('\n') # separates it based on the newline 
  # we will only be concerned from string_split[1] onwards since the 
  # first line has no value to us. 
  string_split = string_split[1:]
  
  # join all of them together to form one long string 
  combined_sequence = ''.join(string_split) 
  
  # empty list 
  X_i = []
  for i in range(len(combined_sequence)): 
    if combined_sequence[i]=='A':
      X_i.append(1)
    elif combined_sequence[i]=='C':
      X_i.append(2)
    elif combined_sequence[i]=='T':
      X_i.append(3)
    elif combined_sequence[i] == 'G':
      X_i.append(4)
       
  return X_i, y_i

# now we enumerate over the files 

# create an empty list 
list_Xtrain = [] 
list_ytrain = []

filenames = filenames_txt + filenames_fna

for files in filenames: 
  Xi, yi = process_file(files)
  list_Xtrain.append(Xi)
  list_ytrain.append(yi)

X_train = np.array(list_Xtrain)
y_train = np.array(list_ytrain)

list_Xtrain = [] 
list_ytrain = []

filenames = filenames_txt + filenames_fna

for files in filenames: 
  Xi, yi = process_file(files)
  list_Xtrain.append(Xi)
  list_ytrain.append(yi)

X_train = np.array(list_Xtrain)
y_train = np.array(list_ytrain)


n_values = np.max(y_train) + 1
y_train = np.eye(n_values)[y_train.flatten()]
 
max_review_length = 31000
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

MODEL = load_model('my_model.h5')

@app.route('/api', methods=['GET'])
def api():
  estimate = MODEL.predict(X_train)
  return jsonify(estimate.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)