# Importing necessary libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

# Initializing WordNetLemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()

# Loading intents data from JSON file
intents = json.loads(open('Chatbot\intents.json').read())

# Initializing lists and variables
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Processing each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizing words in patterns
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # Creating documents with words and intent tags
        documents.append((wordList, intent['tag']))
        # Adding intent tags to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatizing words and removing ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# Sorting and removing duplicates from words list
words = sorted(set(words))
# Sorting classes list
classes = sorted(set(classes))

# Saving words and classes lists using pickle for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initializing training data
training = []
outputEmpty = [0] * len(classes)

# Creating bag of words and output rows for training
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffling training data
random.shuffle(training)
# Converting training data to numpy array
training = np.array(training)

# Splitting training data into input and output
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Defining neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compiling the model with stochastic gradient descent optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Saving trained model
model.save('chatbot_model.h5', hist)
print('Done')
