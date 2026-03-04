import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
df.columns = ["label","text"]
df["label"] = df["label"].map({"ham":0,"spam":1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)

# Save model and vectorizer
pickle.dump(svm_model, open("svm_spam_model.pkl","wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl","wb"))

print("Model trained and saved successfully!")

