import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import speech_recognition as sr
import re
#data set is from kaggle
data = pd.read_csv('labeled_data.csv')
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data['tweet'] = data['tweet'].apply(preprocess_text)
model = LogisticRegression(max_iter=1000)

X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['hate_speech'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)


def get_user_input():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            messagebox.showinfo("Message", "Speak")
            audio = r.listen(source)
        user_input = r.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        messagebox.showerror("Error", "Could not understand audio.")
    except sr.RequestError as e:
        messagebox.showerror("Error", "Could not request results; {0}".format(e))
    return None


def predict_hate_speech(user_input):
    user_input_processed = vectorizer.transform([user_input])
    prediction = model.predict(user_input_processed)
    return prediction[0]


def main():
    def handle_input():
        selected_method = method_combobox.get()
        if selected_method == 'Text':
            user_input = text_entry.get()
        elif selected_method == 'Voice':
            user_input = get_user_input()
            if user_input is None:
                return
        else:
            return

        if user_input.lower() == 'q':
            root.destroy()
            return

        prediction = predict_hate_speech(user_input)
        if prediction == 1:
            result_label.config(text="Hate speech detected.", foreground="red")
        else:
            result_label.config(text="No hate speech detected.", foreground="green")

    root = tk.Tk()
    root.title("Hate Speech Detection")
    root.bg_image = tk.PhotoImage(file="mic.png")

    root.bg_label = tk.Label(root, image=root.bg_image)
    root.bg_label.place(relwidth=1, relheight=1)

    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    method_label = ttk.Label(main_frame, text="Choose input method:")
    method_label.grid(column=0, row=0, sticky=(tk.W, tk.E))

    method_combobox = ttk.Combobox(main_frame, values=["Text", "Voice"])
    method_combobox.grid(column=1, row=0, sticky=(tk.W, tk.E))
    method_combobox.current(0)

    text_entry = ttk.Entry(main_frame)
    text_entry.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E))

    result_label = ttk.Label(main_frame, text="")
    result_label.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E))

    submit_button = ttk.Button(main_frame, text="Submit", command=handle_input)
    submit_button.grid(column=0, row=3, columnspan=2, sticky=(tk.W, tk.E))

    root.mainloop()


if __name__=="__main__":
    main()