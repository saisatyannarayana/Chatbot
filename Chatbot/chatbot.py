import random
import json
import pickle
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# Load necessary data and model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('Chatbot\intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Function to clean up user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create bag of words from user input
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict intent for user input
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Function to get response based on predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Evaluation function to calculate metrics
def evaluate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return accuracy, precision, recall, f1

# Sample test queries with their true intents
test_queries = [
    ("Hi there", "greeting"),
    ("What is your name?", "creator"),
    ("Where is the college located?", "location"),
    ("What is the college timing?", "hours"),
    ("Who is the head of the mechanical engineering department?", "hod"),
    ("What is the name of the college principal?", "principal"),
    ("How can I contact the college?", "number"),
    ("Do you have hostel facilities?", "hostel"),
    ("What events are organized in the college?", "event"),
    ("Is there any library in the college?", "library"),
    ("What are the fees for the courses?", "fees"),
    ("Who is the college principal?", "principal")
]

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Loop through test queries
for query, true_intent in test_queries:
    # Predict intent for the query using the chatbot model
    predicted_intent = predict_class(query)[0]['intent']
    
    # Append true and predicted labels
    true_labels.append(true_intent)
    predicted_labels.append(predicted_intent)

# Calculate evaluation metrics
accuracy, precision, recall, f1 = evaluate_metrics(true_labels, predicted_labels)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print("Hello! Bot here....")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
