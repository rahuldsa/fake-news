import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine datasets
df = pd.concat([true_df, fake_df]).reset_index(drop=True)

# Split data
X = df["title"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Predict & evaluate
y_pred = model.predict(X_test_vec)
score = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {score * 100:.2f}%")
print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Try a custom prediction
def check_news(title):
    vec = vectorizer.transform([title])
    pred = model.predict(vec)
    return "📰 Real News" if pred[0] == 1 else "⚠️ Fake News"


# Test example
print("\n🔎 Example Check:")
print(
    check_news(
        "ISRO successfully launches Chandrayaan-3, begins new phase of lunar exploration."
    )
)
