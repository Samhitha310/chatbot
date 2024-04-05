import json
import tkinter as tk
from tkinter import PhotoImage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from PIL import Image, ImageTk
import threading
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, static_folder='static', template_folder='templates')

class MLChatBot:
    def __init__(self, dataset_path):
        # Load and preprocess the dataset
        with open(dataset_path, 'r') as file:
            data = json.load(file)

        questions = [item['question'] for item in data['questions']]
        answers = [item['answer'] for item in data['questions']]

        # Create a pipeline with TF-IDF vectorizer and SVM classifier
        self.model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
        self.model.fit(questions, answers)

    def generate_response(self, user_input):
        # Use the trained model to predict the response
        response = self.model.predict([user_input])
        return response[0]

class ChatGUI:
    def __init__(self, master, chatbot):
        self.master = master
        master.title("Conversation ChatBot")
        

        self.chatbot = chatbot

        self.conversation_text = tk.StringVar()

        # Load and configure the images
        img_you = Image.open("fconversation.png")  # Replace "fconversation.png" with your image file
        img_you = img_you.resize((50, 50))  # Adjust the size as needed
        self.photo_you = ImageTk.PhotoImage(img_you)

        img_chatbot = Image.open("mconversation.png")  # Replace "mconversation.png" with your image file
        img_chatbot = img_chatbot.resize((50, 50))  # Adjust the size as needed
        self.photo_chatbot = ImageTk.PhotoImage(img_chatbot)

        # Conversation label with images
        self.conversation_label = tk.Label(master, wraplength=150)
        self.conversation_label.pack(pady=10)

        # User input entry
        self.user_input_entry = tk.Entry(master)
        self.user_input_entry.pack(pady=10)

        # Send button
        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(pady=10)

    def send_message(self):
        # Handle user input and update conversation
        user_input = self.user_input_entry.get()
        print(f"User: {user_input}")  # Debug print

        if user_input.lower() == 'exit':
            self.conversation_text.set("Goodbye!")
            self.master.after(2000, self.master.destroy)  # Close the window after 2 seconds
            return

        # Get chatbot response and update conversation with image
        response = self.chatbot.generate_response(user_input)
        print(f"ChatBot: {response}")  # Debug print
        new_conversation = f"You: {user_input}\nChatBot: {response}"

        # Add the image for "You"
        user_image_label = tk.Label(self.master, image=self.photo_you)
        user_image_label.pack(anchor="w")

        # Add user input text
        user_input_label = tk.Label(self.master, text=f"You: {user_input}")
        user_input_label.pack(anchor="w")

        # Add the image for "ChatBot"
        chatbot_image_label = tk.Label(self.master, image=self.photo_chatbot)
        chatbot_image_label.pack(anchor="e")

        # Add chatbot response text
        chatbot_response_label = tk.Label(self.master, text=f"ChatBot: {response}")
        chatbot_response_label.pack(anchor="e")

        self.conversation_text.set(new_conversation)
        self.user_input_entry.delete(0, tk.END)  # Clear the user input

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/products')
def products():
    # Your view logic here
    return render_template('products.html')
@app.route('/account')
def account():
    # Your view logic here
    return render_template('account.html')
@app.route('/cart')
def cart():
    # Your view logic here
    return render_template('cart.html')
@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.form['user_input']
    response = ml_chatbot.generate_response(user_input)
    return jsonify({"response": response})

# Run the Flask app
def run_flask_app():
    ml_chatbot = MLChatBot("Ecommerce_FAQ_Chatbot_dataset.json")  # Replace with the actual path to your dataset
    app.run(debug=True)
    return ml_chatbot  # Return the chatbot instance
def flask_thread():
    thread = threading.Thread(target=run_flask_app)
    thread.start()

if __name__ == "__main__":
    ml_chatbot = run_flask_app()
    root = tk.Tk()
    chat_gui = ChatGUI(root, ml_chatbot)
    root.mainloop()
