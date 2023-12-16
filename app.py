import tkinter as tk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle
import re

port_stem = PorterStemmer()
with open('C:\\Users\\tejus\\Desktop\\6th Sem Project\\Sentiment Analysis - Twitter Data (NLP)\\tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# loading the saved model
loaded_model = pickle.load(open('C:\\Users\\tejus\\Desktop\\6th Sem Project\\Sentiment Analysis - Twitter Data (NLP)\\trained_model_another.sav', 'rb'))

def classify_message():
    # Get user input from the entry field
    user_input = entry.get()
    user_input_stemmed = stemming(user_input)

    user_input_vectorized = vectorizer.transform([user_input_stemmed])

    # Perform sentiment analysis using the loaded model
    prediction = loaded_model.predict(user_input_vectorized)

    # Display the result in the output label
    if prediction[0] == 1:
        result_label.config(text="It is a positive message", fg="green")
    else:
        result_label.config(text="It is a negative message", fg="red")

# Create the main application window
app = tk.Tk()
app.title("Sentiment Analysis App")
app.geometry("640x480")  # Set fixed size

# Create and place the label above the text entry box
label = tk.Label(app, text="Enter your message:", font=("Helvetica", 14))
label.pack(pady=10)

# Create and place the input field with a reduced width
entry = tk.Entry(app, width=40, font=("Helvetica", 12))
entry.pack(pady=10)

# Create and place the classify button
classify_button = tk.Button(app, text="Classify", command=classify_message, width=20, height=2)
classify_button.pack()

# Create and place the output label
result_label = tk.Label(app, text="", font=("Helvetica", 16), pady=20)
result_label.pack()

# Start the application
app.mainloop()
