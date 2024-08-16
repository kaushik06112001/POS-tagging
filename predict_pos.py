import pandas as pd
import joblib
import re

# Load the saved model, vectorizer, and label encoder
model = joblib.load('./model/xgboost_model.pkl')
vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')

# Read the new data from a .txt file
file_path = 'assamese.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text to extract words
# This uses a simple regex to split on spaces and punctuation
# words = re.findall(r'\b\w+\b', text)
# words = re.findall(r'\S+', text)
tokens = re.findall(r'\S+', text)
#words = re.findall(r'\b[^\W\d_]+\b', text, re.UNICODE)
words = [token for token in tokens if re.search(r'[^\W\d_]', token, re.UNICODE)]

# Convert the words to a DataFrame
df_new = pd.DataFrame(words, columns=['word'])

# Transform the words using the loaded TF-IDF vectorizer
X_new = vectorizer.transform(df_new['word'])

# Predict the POS tags
y_pred = model.predict(X_new)

# Decode the predicted labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create a DataFrame to show the predictions
results = pd.DataFrame({
    'Word': df_new['word'],
    'Predicted POS': y_pred_labels
})

# Print the predictions
print("Predictions:")
print(results)

# Optionally, save the predictions to a file
output_file_path = 'predicted_pos.csv'
results.to_csv(output_file_path, index=False, encoding='utf-8')

# Save the predictions to a text file
# output_txt_path = 'predicted_pos.txt'
# with open(output_txt_path, 'w', encoding='utf-8') as file:
#     for index, row in results.iterrows():
#         file.write(f"{row['Word']}\t{row['True POS']}\t{row['Predicted POS']}\n")



print(f"Predictions saved to {output_file_path}")
