import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
import time
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

import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill='both', expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.text_output = tk.Text(self, wrap='word', height=10, state='disabled')
        self.text_output.pack(fill='both', expand=True)

        self.prompt_label = tk.Label(self, text="Enter input:")
        self.prompt_label.pack()
        self.prompt_entry = tk.Entry(self)
        self.prompt_entry.pack(fill='x')

        self.prompt_entry.bind('<Return>', self.submit_prompt)
        self.prompt_entry.focus()

    def submit_prompt(self, event):
        input_text = self.prompt_entry.get()
        self.prompt_entry.delete(0, 'end')
        output_text = self.process_input(input_text)
        self.text_output.configure(state='normal')
        self.text_output.insert('end', f'\n\nYou: {input_text}\n{output_text}')
        self.text_output.configure(state='disabled')
        self.text_output.see('end')

            
        

    def process_input(self, input_text):
        # Do some processing on the input text here
        output_text = "guideBot> " + chatbot_response(input_text)
        return output_text

root = tk.Tk()
root.title("Campus guide bot")
app = Application(master=root)
app.mainloop()