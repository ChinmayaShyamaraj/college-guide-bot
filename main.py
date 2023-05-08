import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the intents JSON file
with open('intents.json') as file:
    data = json.load(file)
# Preprocess the data
lemmatizer = WordNetLemmatizer()
#nltk.download('punkt')
#nltk.download('wordnet')
#print("after download")
corpus = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        corpus.append(' '.join(words))
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Train the ML model
from sklearn.svm import LinearSVC

clf = LinearSVC()
clf.fit(X, tags)

# Define a function to generate a response to user input
def chatbot_response(user_input):
    user_input = user_input.lower()
    user_input = lemmatizer.lemmatize(user_input)
    user_input = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_input, X)
    closest_tag_index = similarity_scores.argmax()
    tag = tags[closest_tag_index]
    for intent in data['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response


query=""
print("Hello! I am your campus assistance bot. Ask me anything!\n")
while True:
    query=input(">")
    if query=='bye':
        print(chatbot_response(query))
        break
    response=chatbot_response(query)
    print(response)