import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
try:
    data = pd.read_csv("spam.csv", encoding="latin-1")
except FileNotFoundError:
    print("Dataset file not found! Please ensure 'spam.csv' is in the same directory.")
    exit()

# Use the correct column names
data = data[["class", "message"]]
data.columns = ["label", "message"]

# Encode labels (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Check for missing values and remove them
data = data.dropna()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, random_state=42)

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Function for predicting user input
def predict_spam():
    print("\n--- SMS Spam Detector ---")
    while True:
        user_input = input("Enter a message (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        # Transform the user input using the trained vectorizer
        user_input_vec = vectorizer.transform([user_input])
        # Predict using the trained model
        prediction = model.predict(user_input_vec)[0]
        if prediction == 1:
            print("Prediction: SPAM\n")
        else:
            print("Prediction: HAM\n")

# Start the input prediction loop
predict_spam()
