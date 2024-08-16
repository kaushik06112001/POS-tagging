import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Read the data from a .csv file
file_path = 'POS_input.csv'
data = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # word, pos = line.strip().split(',')
        # data.append((word, pos))
        parts = line.strip().split(',')
        if len(parts) == 2:
            word, pos = parts
            data.append((word, pos))

# Convert to DataFrame
df = pd.DataFrame(data, columns=['word', 'pos'])

# Feature extraction
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))  # character-level n-grams
X = vectorizer.fit_transform(df['word'])

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['pos'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Save the trained model and necessary objects
joblib.dump(model, './model/xgboost_model.pkl')
joblib.dump(vectorizer, './model/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, './model/label_encoder.pkl')

print("Model, vectorizer, and label encoder saved successfully!")
