import sentiment_analysis_processing_pipeline
import tkinter as tk
from tkinter import messagebox


def process_review():
    result_label.config(text="", fg="black")
    review_text = review_entry.get("1.0", "end").strip()
    sentiment = sentiment_analysis_processing_pipeline.processing(review_text)

    if sentiment == "Positive Review":
        result_label.config(text="Positive Review", fg="green")
    else:
        result_label.config(text="Negative Review", fg="red")



root = tk.Tk()
root.title("Movie Review Sentiment Analysis")

# Create widgets
review_label = tk.Label(root, text="Enter your movie review:")
review_label.pack(pady=10)

review_entry = tk.Text(root, height=5, width=50)
review_entry.pack()

process_button = tk.Button(root, text="Process", command=process_review)
process_button.pack(pady=10)

result_label = tk.Label(root, text="", fg="black")
result_label.pack()

root.mainloop()
