import numpy as np
import glob
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Lambda 
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt 
from sklearn import metrics


np.random.seed(7)
X_train = []
Y_train = []
filenames_txt = glob.glob('*.txt') 
filenames_fna = glob.glob('*.fna')

print(filenames_txt)

def process_file(filename): 
  # input: str filename 
  
  # Sets y_i to 1 if it starts with MN and 0 otherwise. 
  y_i = 0
  if filename[:2] == 'MT':  
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
  
print(X_train, y_train)

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

#print(X_train, y_train)
  
max_review_length = 31000
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vecor_length = 5
model = Sequential()
model.add(Embedding(5, embedding_vecor_length, input_length=max_review_length)) #embeddings are meant to downsize input
model.add(Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', strides= 6))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(250))
model.add(Dense(16, activation= "relu"))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=10, batch_size= 64)
#metrics.classification_report(X_train, y_train) hard to do this rn since the data is label-encoded

scores = model.evaluate(X_train, y_train, verbose=0)

print("Accuracy: %.2f%%" % (scores[0]*100))

#add accuracy and loss plots!!
#increase amount of other viral RNA sequences text files
#import OTHER metrics from sklearn!!
