import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Read the data from a .txt file
file_path = 'POS.csv'
data = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        word, pos = line.strip().split(',')
        data.append((word, pos))

# Convert to DataFrame
df = pd.DataFrame(data, columns=['word', 'pos'])

# Feature extraction
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))  # character-level n-grams
X = vectorizer.fit_transform(df['word'])
print("vectorizer",vectorizer)
print("X",X)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['pos'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# For displaying purposes, also split the original word and pos columns
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(df['word'], df['pos'], test_size=0.3, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Decode the predictions and true labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create a DataFrame to show test data and predictions
results = pd.DataFrame({
    'Word': X_test_raw,
    'True POS': y_test_raw,
    'Predicted POS': y_pred_labels
})

# Print test data and predictions
print("Test Data and Predictions:")
print(results)

# Evaluate the model
accuracy = accuracy_score(y_test_raw, y_pred_labels)
report = classification_report(y_test_raw, y_pred_labels)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Save results to a file
# output_file_path = 'output.csv'
# results.to_csv(output_file_path, index=False, encoding='utf-8')
